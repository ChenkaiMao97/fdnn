"""
Upload an fdnn TorchScript model to HuggingFace Hub.

Usage
-----
    python scripts/upload_to_hub.py \
        --model_path /path/to/model_dir \
        --repo_id    your-hf-username/fdnn-maxwell-3d-v1 \
        --token      hf_xxxxxxxxxxxxxxxx \
        [--public]

The model directory must contain:
    <model_path>/models/fdnn_model.pt

By default the repository is created as PRIVATE.  Pass --public to make it
public.

Access control (private repos)
--------------------------------
Private repos are only accessible to you (the owner) and collaborators you
invite.  To invite someone:

1.  Go to https://huggingface.co/<repo_id>/settings/members
2.  Add their HuggingFace username with "read" or "write" role.

Collaborators then download with:

    import fdnn
    model_path = fdnn.hub.download("your-org/fdnn-maxwell-3d-v1",
                                   token="hf_<their_read_token>")
    solver = fdnn.NN_solver(model_path=model_path)

Gated access (alternative)
---------------------------
If you want open-but-moderated access (users request access and you approve),
enable the "gated" feature via the HF web UI after upload:
    https://huggingface.co/<repo_id>/settings  →  "Enable gating"
"""

import argparse
import sys
import os

# Allow running from repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fdnn import hub


def main():
    parser = argparse.ArgumentParser(
        description="Upload fdnn TorchScript model to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Local directory containing models/fdnn_model.pt",
    )
    parser.add_argument(
        "--repo_id", required=True,
        help="HuggingFace repository ID, e.g. 'your-name/fdnn-maxwell-3d-v1'",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HuggingFace write token.  Defaults to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--public", action="store_true", default=False,
        help="Make the repository public (default: private).",
    )
    parser.add_argument(
        "--message", default="Upload fdnn TorchScript model",
        help="Commit message for the upload.",
    )
    args = parser.parse_args()

    if not args.token:
        parser.error(
            "HuggingFace token is required.  Pass --token or set HF_TOKEN env var."
        )

    hub.upload(
        model_path=args.model_path,
        repo_id=args.repo_id,
        token=args.token,
        private=not args.public,
        commit_message=args.message,
    )


if __name__ == "__main__":
    main()
