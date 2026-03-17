"""
Migrate an old-format checkpoint to the fdnn checkpoint format.

Old format (from the training repo):
    <model_path>/models/last_model.pt   →  {"state_dict": ..., ...}
    <model_path>/*.gin                   →  gin config file

New format (fdnn):
    <model_path>/models/fdnn_model.pt   →  {"state_dict": ..., "meta": {...}}

Usage
-----
Before running, set the values in the CONFIG block below, then:

    python scripts/migrate_checkpoint.py

The original checkpoint is NOT modified.  The new file is saved as
``fdnn_model.pt`` alongside it.
"""

import os
import sys
import torch

# ────────────────────────────────────────────────────────────────────────────
# CONFIG — fill these in before running
# ────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "/path/to/your/model"   # directory that contains models/ and *.gin

# Values you extract from your training setup (IterativeTrainer / gin config):
MODEL_TYPE    = "YourModelClassName"   # the string you'll use with @register_model
MODEL_KWARGS  = {                      # kwargs passed to model.__init__
    "domain_sizes": [64, 64, 64],      # example — match your gin config
    "paddings":     [0, 0, 0],
    # add any other constructor args here
}
LN_R          = -16                    # trainer.ln_R
PML_RANGES    = [8, 12]               # trainer.pml_ranges (informational)
DOMAIN_SIZES  = [32, 96, 32, 96, 32, 96]  # trainer.domain_sizes (informational)
RESIDUAL_TYPE = "sc_pml"              # trainer.residual_type (informational)

# ────────────────────────────────────────────────────────────────────────────

def main():
    # Load old checkpoint
    old_path = os.path.join(MODEL_PATH, "models", "last_model.pt")
    if not os.path.exists(old_path):
        old_path = os.path.join(MODEL_PATH, "models", "best_model.pt")
    if not os.path.exists(old_path):
        print(f"ERROR: no checkpoint found in {MODEL_PATH}/models/")
        sys.exit(1)

    print(f"Loading: {old_path}")
    old_ckpt = torch.load(old_path, weights_only=False, map_location="cpu")

    # The old checkpoint may store the state dict directly or under a key
    if isinstance(old_ckpt, dict) and "state_dict" in old_ckpt:
        state_dict = old_ckpt["state_dict"]
    else:
        state_dict = old_ckpt   # raw state dict

    new_ckpt = {
        "state_dict": state_dict,
        "meta": {
            "fdnn_version":  "0.1.0",
            "model_type":    MODEL_TYPE,
            "model_kwargs":  MODEL_KWARGS,
            "ln_R":          LN_R,
            "pml_ranges":    PML_RANGES,
            "domain_sizes":  DOMAIN_SIZES,
            "residual_type": RESIDUAL_TYPE,
        },
    }

    out_path = os.path.join(MODEL_PATH, "models", "fdnn_model.pt")
    torch.save(new_ckpt, out_path)
    print(f"Saved new checkpoint: {out_path}")
    print(f"Meta: {new_ckpt['meta']}")


if __name__ == "__main__":
    main()
