# nnfd — Neural-Network Preconditioned FDFD Solver

`nnfd` solves 3-D Maxwell's equations (FDFD) using a neural network as a
preconditioner for GMRES iteration.

## Install

```bash
pip install nnfd
```

## Quick start

```python
import nnfd

# Download pre-trained model (once; cached in ~/.cache/nnfd/)
model_path = nnfd.hub.download(
    "your-org/nnfd-maxwell-3d-v1",
    token="hf_...",   # omit if the repo is public
)

solver = nnfd.NN_solver(model_path=model_path)
solver.config({"device": "cuda:0", "tol": 1e-4, "max_iter": 50})

# eps:  (N, sx, sy, sz)      float32  — relative permittivity
# src:  (N, sx, sy, sz, 6)   float32  — source (real repr of 3 complex E components)
E, residuals = solver.solve(
    eps, src,
    wavelength=1.55,               # free-space wavelength (same units as dL)
    dL=0.05,                       # grid spacing
    pml_layers=(10,10,10,10,10,10),
)
# E:         (N, sx, sy, sz, 3)  complex64
# residuals: list[float], one per mini-batch chunk
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
