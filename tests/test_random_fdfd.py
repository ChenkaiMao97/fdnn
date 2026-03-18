"""
Random-dielectric GMRES integration test.

Constructs a 3-D simulation domain with:
  - random dielectric blocks (eps ∈ [1, 12])
  - PML on all six faces
  - a single point-dipole source (Ex component)
then solves with NN_solver (GMRES + NN preconditioner) and plots the results.

Usage
-----
    # With a local model:
    python tests/test_random_fdfd.py --model_path /path/to/model_dir

    # Download model from HuggingFace first:
    python tests/test_random_fdfd.py \
        --model_path hf://your-org/nnfd-maxwell-3d-v1 \
        --hf_token hf_xxxxxxxx

    # Quick CPU sanity-check (small domain):
    python tests/test_random_fdfd.py --model_path /path/to/model_dir \
        --device cpu --size 32

Outputs
-------
Results are saved to --out_dir (default: ./test_outputs/):
  - eps.png          permittivity slices
  - src_Ex.png       source Ex (real part) slices
  - Ex_real.png      solved Ex field (real part)
  - Ex_imag.png      solved Ex field (imag part)
  - intensity.png    |E|^2 summed over components
  - residuals.txt    GMRES residual history
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless; remove if running interactively
import matplotlib.pyplot as plt

# Allow running from repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import nnfd
from nnfd.utils.plot_field3d import plot_3slices


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_random_eps(sim_shape, n_blocks=6, eps_bg=1.0, eps_max=10.0, seed=42):
    """Build a permittivity volume with random dielectric rectangular blocks.

    Parameters
    ----------
    sim_shape : (sx, sy, sz)
    n_blocks  : number of random rectangular inclusions
    eps_bg    : background permittivity
    eps_max   : maximum permittivity of any block

    Returns
    -------
    eps : torch.Tensor, shape (sx, sy, sz), float32
    """
    rng = np.random.default_rng(seed)
    sx, sy, sz = sim_shape
    eps = eps_bg * np.ones(sim_shape, dtype=np.float32)

    for _ in range(n_blocks):
        # Random block occupying 20–60 % of each axis
        bx = max(2, int(rng.uniform(0.20, 0.60) * sx))
        by = max(2, int(rng.uniform(0.20, 0.60) * sy))
        bz = max(2, int(rng.uniform(0.20, 0.60) * sz))

        x0 = rng.integers(0, sx - bx)
        y0 = rng.integers(0, sy - by)
        z0 = rng.integers(0, sz - bz)

        block_eps = rng.uniform(1.5, eps_max)
        eps[x0:x0+bx, y0:y0+by, z0:z0+bz] = block_eps

    return torch.from_numpy(eps)


def make_point_source(sim_shape, pml_layers, polarization="x"):
    """Place a point-dipole source just inside the PML on the z-low side.

    Parameters
    ----------
    sim_shape   : (sx, sy, sz)
    pml_layers  : (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    polarization: "x", "y", or "z"

    Returns
    -------
    src : torch.Tensor, shape (sx, sy, sz, 6), float32
          real representation: [Ex_r, Ex_i, Ey_r, Ey_i, Ez_r, Ez_i]
    """
    sx, sy, sz = sim_shape
    src = torch.zeros(*sim_shape, 6, dtype=torch.float32)

    # Source placed 2 cells inside the PML on the z-low face
    src_z = pml_layers[4] + 2
    src_x, src_y = sx // 2, sy // 2

    pol_index = {"x": 0, "y": 2, "z": 4}[polarization]   # real part of chosen component
    src[src_x, src_y, src_z, pol_index] = 1.0

    return src


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Random-dielectric GMRES integration test.")
    parser.add_argument("--model_path", required=True,
                        help="Local model directory OR 'hf://<repo_id>' to download from HuggingFace.")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace read token (for private repos).")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument("--size", type=int, default=128,
                        help="Cubic domain size (voxels per side). Use 32 for a quick CPU test.")
    parser.add_argument("--pml", type=int, default=30,
                        help="PML thickness on each face (voxels).")
    parser.add_argument("--wavelength", type=float, default=1.55,
                        help="Free-space wavelength (same units as --dL).")
    parser.add_argument("--dL", type=float, default=0.05,
                        help="Grid spacing (same units as --wavelength).")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="GMRES convergence tolerance.")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum GMRES iterations.")
    parser.add_argument("--restart", type=int, default=10,
                        help="GMRES restart period (0 = no restart).")
    parser.add_argument("--out_dir", default="test_outputs",
                        help="Directory to save plots and residuals.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dielectric structure.")
    args = parser.parse_args()

    # ── 0. Resolve model path ─────────────────────────────────────────────────
    model_path = args.model_path
    if model_path.startswith("hf://"):
        repo_id = model_path[len("hf://"):]
        print(f"Downloading model from HuggingFace: {repo_id}")
        model_path = nnfd.hub.download(repo_id, token=args.hf_token)
        print(f"Model downloaded to: {model_path}")

    # ── 1. Simulation parameters ──────────────────────────────────────────────
    S           = args.size
    P           = args.pml
    sim_shape   = (S, S, S)
    pml_layers  = (P, P, P, P, P, P)
    wavelength  = args.wavelength
    dL          = args.dL

    print(f"\nSimulation domain : {sim_shape}  (PML = {P} voxels on each face)")
    print(f"Wavelength / dL   : {wavelength} / {dL}")
    print(f"Device            : {args.device}")

    # ── 2. Build permittivity and source ──────────────────────────────────────
    print("\nBuilding random dielectric structure ...")
    eps = make_random_eps(sim_shape, n_blocks=8, seed=args.seed)   # (sx, sy, sz)
    src = make_point_source(sim_shape, pml_layers, polarization="x")  # (sx, sy, sz, 6)

    # Add batch dimension: (1, sx, sy, sz) and (1, sx, sy, sz, 6)
    eps_batch = eps.unsqueeze(0)
    src_batch = src.unsqueeze(0)

    print(f"  eps  shape : {tuple(eps_batch.shape)}  "
          f"range [{eps.min().item():.2f}, {eps.max().item():.2f}]")
    print(f"  src  shape : {tuple(src_batch.shape)}")

    # ── 3. Build solver ───────────────────────────────────────────────────────
    solver = nnfd.NN_solver(model_path=model_path)
    solver.config({
        "device":    args.device,
        "tol":       args.tol,
        "max_iter":  args.max_iter,
        "restart":   args.restart,
        "batch_size": 1,
        "verbose":   True,
    })

    # ── 4. Solve ──────────────────────────────────────────────────────────────
    print("\nSolving Maxwell FDFD with GMRES ...")
    E, residuals = solver.solve(
        eps_batch, src_batch,
        wavelength=wavelength,
        dL=dL,
        pml_layers=pml_layers,
    )
    # E: (1, sx, sy, sz, 3) complex64

    print(f"\nFinal GMRES residual(s): {residuals}")
    for i, r in enumerate(residuals):
        print(f"  chunk {i}: {r:.4e}")
        assert r < 0.01, f"GMRES did not converge for chunk {i}: residual = {r:.4e}"

    print("GMRES converged.")

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convenience plot functions
    plot_eps    = partial(plot_3slices, ticks=False, colorbar=True,
                          my_cmap=plt.cm.binary, cm_zero_center=False)
    plot_fields = partial(plot_3slices, ticks=False, colorbar=True,
                          my_cmap=plt.cm.seismic, cm_zero_center=True)

    # Permittivity
    eps_np = eps.numpy()   # (sx, sy, sz)
    plot_eps(eps_np, fname=str(out_dir / "eps.png"),
             title="Permittivity (ε)")
    plt.close("all")
    print(f"Saved: {out_dir}/eps.png")

    # Source Ex (real part)
    src_Ex_np = src[..., 0].numpy()  # (sx, sy, sz)
    plot_fields(src_Ex_np, fname=str(out_dir / "src_Ex.png"),
                title="Source  Ex  (real)")
    plt.close("all")
    print(f"Saved: {out_dir}/src_Ex.png")

    E0 = E[0]  # (sx, sy, sz, 3) complex64, remove batch dim

    # Ex real / imag
    Ex_real = E0[..., 0].real.cpu().numpy()
    plot_fields(Ex_real, fname=str(out_dir / "Ex_real.png"),
                title="Solved  Ex  (real part)")
    plt.close("all")

    Ex_imag = E0[..., 0].imag.cpu().numpy()
    plot_fields(Ex_imag, fname=str(out_dir / "Ex_imag.png"),
                title="Solved  Ex  (imag part)")
    plt.close("all")
    print(f"Saved: {out_dir}/Ex_real.png, {out_dir}/Ex_imag.png")

    # |E|^2 intensity
    intensity = torch.sum(torch.abs(E0) ** 2, dim=-1).cpu().numpy()  # (sx, sy, sz)
    plot_eps(intensity, fname=str(out_dir / "intensity.png"),
             title="|E|² intensity")
    plt.close("all")
    print(f"Saved: {out_dir}/intensity.png")

    # Residual log
    res_file = out_dir / "residuals.txt"
    with open(res_file, "w") as f:
        for i, r in enumerate(residuals):
            f.write(f"chunk {i}: {r:.6e}\n")
    print(f"Saved: {res_file}")

    print(f"\nAll outputs written to: {out_dir}/")
    print("Test PASSED.")


if __name__ == "__main__":
    main()
