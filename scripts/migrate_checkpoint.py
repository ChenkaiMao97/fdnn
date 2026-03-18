"""
Migrate an old-format checkpoint to the nnfd TorchScript format.

What this script does
---------------------
1. Loads your trained model weights.
2. Instantiates the model class (you need the source here — this is the
   LAST time the architecture code is needed).
3. Exports to TorchScript, which bundles the computation graph without
   exposing Python source to end users.
4. Embeds metadata (ln_R etc.) inside the .pt file as a JSON extra file.

Output: <model_path>/models/nnfd_model.pt
  - A single TorchScript file.  No separate config, no model class required
    to load it.  Users cannot read the architecture from this file.

Run this script from the repo root (where your model class is importable):

    cd /path/to/your/training/repo
    python /path/to/nnfd/scripts/migrate_checkpoint.py

The original checkpoint is NOT modified.
"""

import json
import os
import sys
import torch
from collections import OrderedDict
import re

# ────────────────────────────────────────────────────────────────────────────
# CONFIG — fill these in before running
# ────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "/media/ps3/chenkaim/checkpoints/copied_models/aperiodic_CondConv_30_pml_small-10_16_25T15_21_04"   # directory containing models/last_model.pt

# Metadata baked into the exported file.
# ln_R is the only value the solver actually uses at inference time.
# The rest are informational (training provenance).
META = {
    "nnfd_version":  "0.1.0",
    "ln_R":          -10,             # trainer.ln_R
    "pml_ranges":    [30, 30, 30, 30, 30, 30],         # trainer.pml_ranges  [pml_min, pml_max]
    "domain_sizes":  [128,256,128,256,96,128],  # trainer.domain_sizes [xmin,xmax, ...]
    "residual_type": "SC-PML",        # trainer.residual_type
}

# ── TorchScript export ───────────────────────────────────────────────────────
# We use torch.jit.script, which correctly handles all control flow (if/for).
#
# Only fall back to torch.jit.trace if script raises an error — this happens
# when the model uses Python constructs that TorchScript can't parse (e.g.
# certain third-party ops).  Trace is NOT preferred: it silently records only
# one execution path and will produce wrong results if the model has
# data-dependent branches.
#
# If you must use trace, set these to a valid example input pair:
TRACE_EXAMPLE_INPUT = None   # e.g. torch.randn(1, 64, 64, 64, 7)
TRACE_EXAMPLE_FREQ  = None   # e.g. torch.tensor([0.05 / 1.55])

# ────────────────────────────────────────────────────────────────────────────

def remap_state_dict(state_dict: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    """
    Remap checkpoint keys from the original MG_CondConv layout to the
    TorchScript-compatible layout after the following renames:

      DownSizedNetwork:
        downs.N.0.*  →  down_blocks.N.*   (the residual_blocks)
        downs.N.1.*  →  down_samples.N.*  (the Downsample)

      UpSizedNetwork:
        ups.N.0.*    →  up_samples.N.*    (the Upsample)
        ups.N.1.*    →  up_blocks.N.*     (the residual_blocks)

    This handles setup_down_net, solve_down_net, and solve_up_net.
    """
    new_sd: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in state_dict.items():
        # ── DownSizedNetwork: downs.N.0 → down_blocks.N
        k = re.sub(r'(\.downs\.)(\d+)\.0\.', r'.down_blocks.\2.', k)
        # ── DownSizedNetwork: downs.N.1 → down_samples.N
        k = re.sub(r'(\.downs\.)(\d+)\.1\.', r'.down_samples.\2.', k)
        # ── UpSizedNetwork: ups.N.0 → up_samples.N
        k = re.sub(r'(\.ups\.)(\d+)\.0\.', r'.up_samples.\2.', k)
        # ── UpSizedNetwork: ups.N.1 → up_blocks.N
        k = re.sub(r'(\.ups\.)(\d+)\.1\.', r'.up_blocks.\2.', k)
        new_sd[k] = v
    return new_sd

def load_model(model_path: str) -> torch.nn.Module:
    """
    Instantiate your model class and load the trained weights.

    Edit this function to match your model's constructor and checkpoint format.
    This is the only place where your model's Python class is needed.
    """

    import gin
    sys.path.insert(0, model_path)          # makes waveynet3d importable

    # parse gin config (same as NN_solver.init() did)
    for f in os.listdir(model_path):
        if f.endswith(".gin"):
            gin.parse_config_file(os.path.join(model_path, f))

    from waveynet3d.models import model_factory

    # instantiate model — match whatever your gin config sets for domain_sizes
    model = model_factory(domain_sizes=META["domain_sizes"], paddings=[0, 0, 0])


    # --- Load weights -------------------------------------------------------
    weights_path = os.path.join(model_path, "models", "last_model.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_path, "models", "best_model.pt")

    ckpt = torch.load(weights_path, weights_only=False, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Diagnostic: print old checkpoint keys vs new model keys for up_net
    ckpt_up_keys = sorted([k for k in state_dict.keys() if 'up' in k])
    model_up_keys = sorted([k for n, _ in model.named_parameters() if 'up' in n for k in [n]])

    print("=== CHECKPOINT keys (up) ===")
    for k in ckpt_up_keys[:20]: print(" ", k)
    print("=== MODEL keys (up) ===")
    for k in model_up_keys[:20]: print(" ", k)

    state_dict = remap_state_dict(state_dict)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print("Exporting with torch.jit.script ...")
    try:
        scripted = torch.jit.script(model)
    except Exception as e:
        if TRACE_EXAMPLE_INPUT is None:
            print(f"\ntorch.jit.script failed:\n  {e}\n")
            print(
                "If your model uses constructs incompatible with TorchScript, "
                "set TRACE_EXAMPLE_INPUT and TRACE_EXAMPLE_FREQ in this script "
                "to fall back to torch.jit.trace."
            )
            sys.exit(1)
        print(f"script failed ({e}), falling back to torch.jit.trace ...")
        scripted = torch.jit.trace(
            model,
            example_inputs=(TRACE_EXAMPLE_INPUT, TRACE_EXAMPLE_FREQ),
        )

    out_path = os.path.join(MODEL_PATH, "models", "nnfd_model.pt")
    torch.jit.save(
        scripted,
        out_path,
        _extra_files={"meta.json": json.dumps(META)},
    )
    print(f"Saved: {out_path}")
    print(f"Meta:  {META}")

    # Quick sanity check: reload and verify meta round-trips correctly.
    extra = {"meta.json": ""}
    torch.jit.load(out_path, _extra_files=extra)
    assert json.loads(extra["meta.json"]) == META, "meta round-trip failed!"
    print("Sanity check passed.")

if __name__ == "__main__":
    main()
