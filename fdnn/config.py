from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SolverConfig:
    # ── Solver ────────────────────────────────────────────────────────────────
    tol: float = 1e-5
    """Relative residual convergence tolerance."""

    max_iter: int = 100
    """Maximum total GMRES iterations."""

    restart: int = 10
    """Restart interval for restarted GMRES. Set to 0 to disable restarts."""

    verbose: bool = False
    """Print per-iteration residual progress."""

    # ── Compute ───────────────────────────────────────────────────────────────
    batch_size: int = 1
    """Number of samples per GMRES call. Larger values use more GPU memory."""

    device: str = "cuda:0"
    """Target device for single-GPU mode, e.g. 'cuda:0' or 'cpu'."""

    multi_gpu: bool = False
    """If True, distribute the batch across gpu_ids in parallel threads."""

    gpu_ids: List[int] = field(default_factory=list)
    """GPU indices to use when multi_gpu=True, e.g. [0, 1, 2, 3]."""

    def validate(self) -> None:
        if self.multi_gpu and not self.gpu_ids:
            raise ValueError(
                "multi_gpu=True requires gpu_ids to be a non-empty list"
            )
