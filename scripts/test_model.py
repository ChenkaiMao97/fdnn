"""
Quick sanity-check for the exported nnfd TorchScript model.

Run from anywhere:
    python /path/to/nnfd/scripts/test_model.py
"""

import json
import sys
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PT = "/media/ps3/chenkaim/checkpoints/copied_models/aperiodic_CondConv_30_pml_small-10_16_25T15_21_04/models/nnfd_model.pt"

# Spatial size to test with — pick something small and fast on CPU.
SX, SY, SZ = 32, 32, 32

# eps channels = 1 (permittivity) + number of PML channels your model was trained with.
# Check your gin config: input_channel_eps = ?
INPUT_CHANNEL_EPS = 4   # adjust to match your trained model

# Physics params (only used to compute freq = dL/wavelength)
DL         = 0.05
WAVELENGTH = 1.55

DEVICE = "cpu"          # change to "cuda:0" to test on GPU
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print(f"Loading: {MODEL_PT}")
    extra = {"meta.json": ""}
    model = torch.jit.load(MODEL_PT, map_location=DEVICE, _extra_files=extra)
    model.eval()

    meta = json.loads(extra["meta.json"])
    print(f"Meta:    {meta}\n")

    # ── Build dummy inputs ───────────────────────────────────────────────────
    # eps_with_pml:  (bs, sx, sy, sz, input_channel_eps)
    # rhs:           (bs, sx, sy, sz, 6)   — real repr of 3 complex E components
    # freq:          (1,)                  — dL / wavelength
    eps  = torch.rand(1, SX, SY, SZ, INPUT_CHANNEL_EPS, device=DEVICE)
    rhs  = torch.rand(1, SX, SY, SZ, 6,                 device=DEVICE)
    freq = torch.tensor([DL / WAVELENGTH],               device=DEVICE)

    print(f"eps  shape : {tuple(eps.shape)}")
    print(f"rhs  shape : {tuple(rhs.shape)}")
    print(f"freq value : {freq.item():.5f}\n")

    # ── Forward pass ────────────────────────────────────────────────────────
    with torch.no_grad():
        print("Running model.setup(eps, freq) ...")
        model.setup(eps, freq)

        print("Running model.forward(rhs, freq) ...")
        out = model(rhs, freq)

    print(f"\nOutput shape : {tuple(out.shape)}")
    print(f"Expected     : (1, {SX}, {SY}, {SZ}, 6)")
    assert out.shape == (1, SX, SY, SZ, 6), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(),  "Output contains NaN!"
    assert not torch.isinf(out).any(),  "Output contains Inf!"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
