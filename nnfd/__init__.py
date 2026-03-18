"""
nnfd — Finite Difference Neural Network solver for 3D Maxwell FDFD.

Quick start::

    import nnfd

    solver = nnfd.NN_solver(model_path="./my_model")  # or set NNFD_MODEL_PATH
    solver.config({
        "sim_shape":  (64, 64, 64),
        "wavelength": 1.55,
        "dL":         0.05,
        "pml_layers": (10, 10, 10, 10, 10, 10),
        "batch_size": 4,
        "device":     "cuda:0",
    })

    E, residuals = solver.solve(eps, src, wavelength=1.55, dL=0.05,
                                pml_layers=(10, 10, 10, 10, 10, 10))
    # E:         (N, sx, sy, sz, 3)  complex64 E-field
    # residuals: list[float]         one per mini-batch chunk
"""

from nnfd.solver import NN_solver
from nnfd.config import SolverConfig
from nnfd import hub

__all__ = ["NN_solver", "SolverConfig", "hub"]

__version__ = "0.1.0"
