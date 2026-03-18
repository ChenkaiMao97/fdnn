import copy
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from nnfd.config import SolverConfig
from nnfd._pml import build_pml_channels


class NN_solver:
    """Neural-network preconditioned GMRES solver for 3D Maxwell FDFD.

    Usage::

        import nnfd

        solver = nnfd.NN_solver(model_path="./my_model")
        solver.config({
            "tol":        1e-5,
            "max_iter":   100,
            "restart":    10,
            "batch_size": 4,
            "device":     "cuda:0",
            # multi-GPU:
            # "multi_gpu": True,
            # "gpu_ids":   [0, 1, 2, 3],
        })

        # Physics parameters are per-solve, not stored in config:
        E, residuals = solver.solve(
            eps, src,
            wavelength=1.55,
            dL=0.05,
            pml_layers=(10, 10, 10, 10, 10, 10),
        )
        # E:         (N, sx, sy, sz, 3), torch.complex64
        # residuals: list[float], one per mini-batch chunk

    Checkpoint format
    -----------------
    The solver expects ``<model_path>/models/nnfd_model.pt`` with structure::

        {
            "state_dict": <OrderedDict>,
            "meta": {
                "nnfd_version":  "0.1.0",
                "model_type":    "<registered model key>",
                "model_kwargs":  { ... },   # passed to model constructor
                "ln_R":          -16,
                "pml_ranges":    [...],     # informational
                "domain_sizes":  [...],     # informational
                "residual_type": "...",     # informational
            }
        }

    Model path
    ----------
    Pass ``model_path`` directly or set the ``NNFD_MODEL_PATH`` env variable.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or os.environ.get("NNFD_MODEL_PATH")
        if not self.model_path:
            raise ValueError(
                "model_path must be provided or the NNFD_MODEL_PATH "
                "environment variable must be set."
            )
        self._cfg = SolverConfig()
        self._initialized = False
        self._ln_R: Optional[float] = None
        # gpu_id -> {"model": nn.Module}
        self._gpu_contexts: Dict[Optional[int], dict] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def config(self, cfg: dict) -> "NN_solver":
        """Update solver configuration. Returns self for chaining.

        Accepts any subset of :class:`~nnfd.config.SolverConfig` field names.
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
        wavelength: float,
        dL: float,
        pml_layers: Tuple[int, int, int, int, int, int],
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
        wavelength:
            Free-space wavelength, same units as dL.
        dL:
            Grid spacing (isotropic).
        pml_layers:
            PML thickness in voxels: ``(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)``.

        Returns
        -------
        E:
            Complex E-field, shape ``(N, sx, sy, sz, 3)``, ``torch.complex64``.
        residuals:
            List of final relative residuals (one float per mini-batch chunk).
        """
        if len(pml_layers) != 6:
            raise ValueError(
                "pml_layers must have exactly 6 values: "
                "(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)"
            )

        self._init()

        eps = _to_tensor(eps, torch.float32)
        src = _to_tensor(src, torch.float32)

        squeeze = eps.dim() == 3
        if squeeze:
            eps = eps.unsqueeze(0)
            src = src.unsqueeze(0)

        gpu_ids = self._get_gpu_ids()
        kwargs = dict(wavelength=wavelength, dL=dL, pml_layers=pml_layers)

        if self._cfg.multi_gpu and len(gpu_ids) > 1:
            E, residuals = self._solve_multi_gpu(eps, src, gpu_ids, **kwargs)
        else:
            E, residuals = self._solve_batched(eps, src, gpu_ids[0], **kwargs)

        if squeeze:
            E = E.squeeze(0)

        return E, residuals

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init(self) -> None:
        """Load TorchScript model from checkpoint. Called once on first solve()."""
        if self._initialized:
            return
        self._cfg.validate()

        ckpt_path = os.path.join(self.model_path, "models", "nnfd_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Expected checkpoint at {ckpt_path}.\n"
                "Migrate your old checkpoint with scripts/migrate_checkpoint.py."
            )

        # Load TorchScript model; meta is embedded as an extra file.
        extra_files = {"meta.json": ""}
        model_cpu = torch.jit.load(
            ckpt_path, map_location="cpu", _extra_files=extra_files
        )
        meta = json.loads(extra_files["meta.json"])
        self._ln_R = meta["ln_R"]
        model_cpu.eval()

        # Copy to each target device.
        self._gpu_contexts = {}
        for gpu_id in self._get_gpu_ids():
            dev = _device_str(gpu_id)
            self._gpu_contexts[gpu_id] = {
                "model": copy.deepcopy(model_cpu).to(dev),
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
        wavelength: float,
        dL: float,
        pml_layers: tuple,
    ) -> Tuple[torch.Tensor, List[float]]:
        N = eps.shape[0]
        bs = self._cfg.batch_size
        E_parts: List[torch.Tensor] = []
        residuals: List[float] = []

        for start in range(0, N, bs):
            end = min(start + bs, N)
            E_chunk, res = self._solve_chunk(
                eps[start:end], src[start:end], gpu_id,
                wavelength, dL, pml_layers,
            )
            E_parts.append(E_chunk.cpu())
            residuals.append(res)

        return torch.cat(E_parts, dim=0), residuals

    def _solve_multi_gpu(
        self,
        eps: torch.Tensor,
        src: torch.Tensor,
        gpu_ids: List[int],
        wavelength: float,
        dL: float,
        pml_layers: tuple,
    ) -> Tuple[torch.Tensor, List[float]]:
        N = eps.shape[0]
        n_gpus = len(gpu_ids)
        chunk_size = (N + n_gpus - 1) // n_gpus

        def run(i: int, gpu_id: int):
            start = i * chunk_size
            end = min(start + chunk_size, N)
            if start >= N:
                return i, None, None
            E_chunk, r_chunk = self._solve_batched(
                eps[start:end], src[start:end], gpu_id,
                wavelength, dL, pml_layers,
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
        wavelength: float,
        dL: float,
        pml_layers: tuple,
    ) -> Tuple[torch.Tensor, float]:
        """Run one GMRES solve on a single mini-batch chunk."""
        from nnfd.solvers.gmres import mygmrestorch
        from nnfd.utils.physics import residue_E, src2rhs
        from nnfd.utils.utils import c2r, r2c

        cfg = self._cfg
        model = self._gpu_contexts[gpu_id]["model"]
        dev = _device_str(gpu_id)

        eps_chunk = eps_chunk.to(dev)
        src_chunk = src_chunk.to(dev)

        # Build PML channels for this specific (sim_shape, wl, dL, pmls).
        # Cheap to compute (pure numpy → tensor), so done per chunk.
        sim_shape = tuple(eps_chunk.shape[1:])  # (sx, sy, sz)
        pml_channels = build_pml_channels(
            sim_shape=sim_shape,
            wavelength=wavelength,
            dL=dL,
            pml_layers=pml_layers,
            ln_R=self._ln_R,
            device=dev,
        )  # (1, sx, sy, sz, n_pml_ch)

        # Concatenate eps with PML channels.
        # Shape: (bs, sx, sy, sz, 1 + n_pml_ch)
        n = eps_chunk.shape[0]
        eps_with_pml = torch.cat(
            [eps_chunk.unsqueeze(-1), pml_channels.expand(n, -1, -1, -1, -1)],
            dim=-1,
        )

        Aop = lambda x: r2c(
            residue_E(
                c2r(x), eps_with_pml[..., 0], src_chunk,
                pml_layers, dL, wavelength,
                batched_compute=True, Aop=True,
            )
        )

        gmres = mygmrestorch(model, Aop, tol=cfg.tol, max_iter=cfg.max_iter)
        freq = torch.tensor(dL / wavelength).unsqueeze(0).to(dev)
        gmres.setup_eps(eps_with_pml, freq)

        complex_rhs = r2c(src2rhs(src_chunk, dL, wavelength))

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
