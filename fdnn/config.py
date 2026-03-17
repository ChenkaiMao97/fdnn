from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SolverConfig:
    # ── Physics (required before calling solve) ───────────────────────────────
    sim_shape: Optional[Tuple[int, int, int]] = None
    """Simulation domain size in voxels, e.g. (64, 64, 64)."""

    wavelength: Optional[float] = None
    """Free-space wavelength in the same units as dL."""

    dL: Optional[float] = None
    """Grid spacing (isotropic), same units as wavelength."""

    pml_layers: Optional[Tuple[int, int, int, int, int, int]] = None
    """PML thickness in voxels: (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)."""

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
        required = {"sim_shape", "wavelength", "dL", "pml_layers"}
        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            raise ValueError(
                f"SolverConfig is missing required fields: {missing}"
            )
        if len(self.pml_layers) != 6:
            raise ValueError(
                "pml_layers must have exactly 6 values: "
                "(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)"
            )
        if self.multi_gpu and not self.gpu_ids:
            raise ValueError(
                "multi_gpu=True requires gpu_ids to be a non-empty list"
            )
