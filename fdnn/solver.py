import copy
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from fdnn.config import SolverConfig
from fdnn._model import build_model
from fdnn._pml import build_pml_channels


class NN_solver:
    """Neural-network preconditioned GMRES solver for 3D Maxwell FDFD.

    Usage::

        import fdnn

        solver = fdnn.NN_solver(model_path="./my_model")
        solver.config({
            "sim_shape":   (64, 64, 64),
            "wavelength":  1.55,
            "dL":          0.05,
            "pml_layers":  (10, 10, 10, 10, 10, 10),
            "tol":         1e-5,
            "max_iter":    100,
            "restart":     10,
            "batch_size":  4,
            "device":      "cuda:0",
            # multi-GPU:
            # "multi_gpu": True,
            # "gpu_ids":   [0, 1, 2, 3],
        })

        # eps: (N, sx, sy, sz)  — permittivity, real-valued
        # src: (N, sx, sy, sz, 6) — source, real repr. of 3 complex components
        E, residuals = solver.solve(eps, src)
        # E:         (N, sx, sy, sz, 3), torch.complex64
        # residuals: list[float], one per mini-batch chunk

    Checkpoint format
    -----------------
    The solver expects ``<model_path>/models/fdnn_model.pt`` with structure::

        {
            "state_dict": <OrderedDict>,
            "meta": {
                "fdnn_version": "0.1.0",
                "model_type":   "<registered model key>",
                "model_kwargs": { ... },   # passed to model constructor
                "ln_R":         -16,
                "pml_ranges":   [...],     # informational
                "domain_sizes": [...],     # informational
                "residual_type": "...",    # informational
            }
        }

    Model path
    ----------
    Pass ``model_path`` directly or set the ``FDNN_MODEL_PATH`` env variable.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or os.environ.get("FDNN_MODEL_PATH")
        if not self.model_path:
            raise ValueError(
                "model_path must be provided or the FDNN_MODEL_PATH "
                "environment variable must be set."
            )
        self._cfg = SolverConfig()
        self._initialized = False
        # gpu_id -> {"model": nn.Module, "pml_channels": Tensor}
        self._gpu_contexts: Dict[Optional[int], dict] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def config(self, cfg: dict) -> "NN_solver":
        """Update solver configuration. Returns self for chaining.

        Accepts any subset of :class:`~fdnn.config.SolverConfig` field names.
        """
        valid_keys = set(self._cfg.__dataclass_fields__)
        unknown = set(cfg.keys()) - valid_keys
        if unknown:
            raise ValueError(
                f"Unknown config keys: {unknown}. "
                f"Valid keys: {sorted(valid_keys)}"
            )
        for k, v in cfg.items():
            setattr(self._cfg, k, v)
        self._initialized = False
        return self

    @torch.no_grad()
    def solve(
        self,
        eps: Union[torch.Tensor, np.ndarray],
        src: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, List[float]]:
        """Solve the Maxwell FDFD system.

        Parameters
        ----------
        eps:
            Permittivity.  Shape ``(N, sx, sy, sz)``, real float32.
            A single un-batched sample ``(sx, sy, sz)`` is also accepted.
        src:
            Source field, real representation of three complex E components.
            Shape ``(N, sx, sy, sz, 6)``.

        Returns
        -------
        E:
            Complex E-field, shape ``(N, sx, sy, sz, 3)``, ``torch.complex64``.
        residuals:
            List of final relative residuals (one float per mini-batch chunk).
        """
        self._init()

        eps = _to_tensor(eps, torch.float32)
        src = _to_tensor(src, torch.float32)

        squeeze = eps.dim() == 3
        if squeeze:
            eps = eps.unsqueeze(0)
            src = src.unsqueeze(0)

        gpu_ids = self._get_gpu_ids()
        if self._cfg.multi_gpu and len(gpu_ids) > 1:
            E, residuals = self._solve_multi_gpu(eps, src, gpu_ids)
        else:
            E, residuals = self._solve_batched(eps, src, gpu_ids[0])

        if squeeze:
            E = E.squeeze(0)

        return E, residuals

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init(self) -> None:
        if self._initialized:
            return
        self._cfg.validate()

        ckpt_path = os.path.join(self.model_path, "models", "fdnn_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Expected checkpoint at {ckpt_path}.\n"
                "Migrate your old checkpoint with scripts/migrate_checkpoint.py."
            )

        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        meta = ckpt["meta"]
        state_dict = ckpt["state_dict"]
        ln_R = meta["ln_R"]

        # Build one model on CPU, then copy to each GPU.
        model_cpu = build_model(meta["model_type"], meta["model_kwargs"])
        model_cpu.load_state_dict(state_dict)
        model_cpu.eval()

        gpu_ids = self._get_gpu_ids()
        self._gpu_contexts = {}

        for gpu_id in gpu_ids:
            dev = _device_str(gpu_id)
            model_dev = copy.deepcopy(model_cpu).to(dev)
            pml_ch = build_pml_channels(
                sim_shape=self._cfg.sim_shape,
                wavelength=self._cfg.wavelength,
                dL=self._cfg.dL,
                pml_layers=self._cfg.pml_layers,
                ln_R=ln_R,
                device=dev,
            )
            self._gpu_contexts[gpu_id] = {
                "model": model_dev,
                "pml_channels": pml_ch,  # (1, sx, sy, sz, n_pml_ch)
            }

        self._initialized = True

    # ── Dispatch helpers ──────────────────────────────────────────────────────

    def _get_gpu_ids(self) -> List[Optional[int]]:
        if self._cfg.multi_gpu and self._cfg.gpu_ids:
            return self._cfg.gpu_ids
        device = self._cfg.device
        if device.startswith("cuda:"):
            return [int(device.split(":")[-1])]
        if device == "cuda":
            return [0]
        return [None]  # CPU

    def _solve_batched(
        self,
        eps: torch.Tensor,
        src: torch.Tensor,
        gpu_id: Optional[int],
    ) -> Tuple[torch.Tensor, List[float]]:
        """Sequentially process chunks of batch_size on a single GPU."""
        N = eps.shape[0]
        bs = self._cfg.batch_size
        E_parts: List[torch.Tensor] = []
        residuals: List[float] = []

        for start in range(0, N, bs):
            end = min(start + bs, N)
            E_chunk, res = self._solve_chunk(eps[start:end], src[start:end], gpu_id)
            E_parts.append(E_chunk.cpu())
            residuals.append(res)

        return torch.cat(E_parts, dim=0), residuals

    def _solve_multi_gpu(
        self,
        eps: torch.Tensor,
        src: torch.Tensor,
        gpu_ids: List[int],
    ) -> Tuple[torch.Tensor, List[float]]:
        """Distribute batch across GPUs; each sub-batch runs in its own thread."""
        N = eps.shape[0]
        n_gpus = len(gpu_ids)
        chunk_size = (N + n_gpus - 1) // n_gpus

        def run(i: int, gpu_id: int):
            start = i * chunk_size
            end = min(start + chunk_size, N)
            if start >= N:
                return i, None, None
            E_chunk, r_chunk = self._solve_batched(
                eps[start:end], src[start:end], gpu_id
            )
            return i, E_chunk, r_chunk

        slot_E: List[Optional[torch.Tensor]] = [None] * n_gpus
        slot_r: List[Optional[List[float]]] = [None] * n_gpus

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = [pool.submit(run, i, gid) for i, gid in enumerate(gpu_ids)]
            for fut in futures:
                i, E_chunk, r_chunk = fut.result()
                if E_chunk is not None:
                    slot_E[i] = E_chunk
                    slot_r[i] = r_chunk

        E_parts = [e for e in slot_E if e is not None]
        residuals = [r for slot in slot_r if slot is not None for r in slot]
        return torch.cat(E_parts, dim=0), residuals

    # ── Core GMRES call ───────────────────────────────────────────────────────

    def _solve_chunk(
        self,
        eps_chunk: torch.Tensor,
        src_chunk: torch.Tensor,
        gpu_id: Optional[int],
    ) -> Tuple[torch.Tensor, float]:
        """Run one GMRES solve on a single mini-batch chunk."""
        from fdnn.solvers.gmres import mygmrestorch
        from fdnn.utils.physics import residue_E, src2rhs
        from fdnn.utils.utils import c2r, r2c

        cfg = self._cfg
        ctx = self._gpu_contexts[gpu_id]
        model = ctx["model"]
        pml_channels = ctx["pml_channels"]  # (1, sx, sy, sz, n_pml_ch)

        dev = _device_str(gpu_id)
        eps_chunk = eps_chunk.to(dev)
        src_chunk = src_chunk.to(dev)

        # Expand PML channels to batch size, concatenate with eps.
        # Shape: (bs, sx, sy, sz, 1 + n_pml_ch)
        n = eps_chunk.shape[0]
        batch_pml = pml_channels.expand(n, -1, -1, -1, -1)
        eps_with_pml = torch.cat([eps_chunk.unsqueeze(-1), batch_pml], dim=-1)

        Aop = lambda x: r2c(
            residue_E(
                c2r(x), eps_with_pml[..., 0], src_chunk,
                cfg.pml_layers, cfg.dL, cfg.wavelength,
                batched_compute=True, Aop=True,
            )
        )

        gmres = mygmrestorch(model, Aop, tol=cfg.tol, max_iter=cfg.max_iter)
        freq = torch.tensor(cfg.dL / cfg.wavelength).unsqueeze(0).to(dev)
        gmres.setup_eps(eps_with_pml, freq)

        complex_rhs = r2c(src2rhs(src_chunk, cfg.dL, cfg.wavelength))

        if cfg.restart == 0:
            x, relres_history, _, _ = gmres.solve(complex_rhs, cfg.verbose)
        else:
            x, relres_history = gmres.solve_with_restart(
                complex_rhs, cfg.tol, cfg.max_iter, cfg.restart, cfg.verbose
            )

        final_residual: float = relres_history[-1] if relres_history else float("nan")

        del complex_rhs, gmres, relres_history
        gc.collect()
        if gpu_id is not None:
            torch.cuda.empty_cache()

        return x, final_residual


# ── Utilities ─────────────────────────────────────────────────────────────────

def _device_str(gpu_id: Optional[int]) -> str:
    return f"cuda:{gpu_id}" if gpu_id is not None else "cpu"


def _to_tensor(x: Union[torch.Tensor, np.ndarray], dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if x.dtype != dtype else x
    return torch.as_tensor(x, dtype=dtype)
