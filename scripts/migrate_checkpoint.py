"""
Migrate an old-format checkpoint to the fdnn TorchScript format.

What this script does
---------------------
1. Loads your trained model weights.
2. Instantiates the model class (you need the source here — this is the
   LAST time the architecture code is needed).
3. Exports to TorchScript, which bundles the computation graph without
   exposing Python source to end users.
4. Embeds metadata (ln_R etc.) inside the .pt file as a JSON extra file.

Output: <model_path>/models/fdnn_model.pt
  - A single TorchScript file.  No separate config, no model class required
    to load it.  Users cannot read the architecture from this file.

Run this script from the repo root (where your model class is importable):

    cd /path/to/your/training/repo
    python /path/to/fdnn/scripts/migrate_checkpoint.py

The original checkpoint is NOT modified.
"""

import json
import os
import sys
import torch

# ────────────────────────────────────────────────────────────────────────────
# CONFIG — fill these in before running
# ────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "/path/to/your/model"   # directory containing models/last_model.pt

# Metadata baked into the exported file.
# ln_R is the only value the solver actually uses at inference time.
# The rest are informational (training provenance).
META = {
    "fdnn_version":  "0.1.0",
    "ln_R":          -16,             # trainer.ln_R
    "pml_ranges":    [8, 12],         # trainer.pml_ranges  [pml_min, pml_max]
    "domain_sizes":  [32, 96, 32, 96, 32, 96],  # trainer.domain_sizes [xmin,xmax, ...]
    "residual_type": "sc_pml",        # trainer.residual_type
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

def load_model(model_path: str) -> torch.nn.Module:
    """
    Instantiate your model class and load the trained weights.

    Edit this function to match your model's constructor and checkpoint format.
    This is the only place where your model's Python class is needed.
    """
    # --- Edit below this line -------------------------------------------

    # Example (replace with your actual imports and constructor):
    #
    #   sys.path.insert(0, "/path/to/waveynet3d")
    #   from waveynet3d.models import model_factory
    #   import gin
    #   gin.parse_config_file(os.path.join(model_path, "config.gin"))
    #   model = model_factory(domain_sizes=[64, 64, 64], paddings=[0, 0, 0])

    raise NotImplementedError(
        "Edit load_model() in migrate_checkpoint.py to instantiate your model."
    )

    # --- Load weights -------------------------------------------------------
    weights_path = os.path.join(model_path, "models", "last_model.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_path, "models", "best_model.pt")

    ckpt = torch.load(weights_path, weights_only=False, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
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

    out_path = os.path.join(MODEL_PATH, "models", "fdnn_model.pt")
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
