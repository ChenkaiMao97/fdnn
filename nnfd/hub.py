"""
Model hub utilities for nnfd — backed by HuggingFace Hub.

Download a pre-trained model
-----------------------------
    import nnfd

    model_path = nnfd.hub.download("your-org/nnfd-maxwell-3d-v1", token="hf_...")
    solver = nnfd.NN_solver(model_path=model_path)

Access control
--------------
The repository can be **private** on HuggingFace Hub, meaning only people you
explicitly grant access to can download it:

* **Private repo** — only you (the owner) and collaborators you invite via the
  HF website (Settings → Members) can read the repo.  Everyone else will get
  an HTTP 401.
* **Gated repo** — if you want open but moderated access, HF supports a "gated"
  model feature where users must request access and you approve them.  This can
  be enabled per-model in the HF model card settings.

In both cases, downloaders pass their personal HF token:

    model_path = nnfd.hub.download("your-org/nnfd-maxwell-3d-v1",
                                   token="hf_<user_read_token>")  # token omittable if public

Upload a model (owner only)
----------------------------
    import nnfd

    nnfd.hub.upload(
        model_path="./my_model",     # directory containing models/nnfd_model.pt
        repo_id="your-org/nnfd-maxwell-3d-v1",
        token="hf_<your_write_token>",
        private=True,                # keep the repo private
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_DEFAULT_CACHE = Path.home() / ".cache" / "nnfd"
_MODEL_FILENAME = "nnfd_model.pt"
_MODEL_SUBDIR   = "models"


# ── Public API ────────────────────────────────────────────────────────────────

def download(
    model_name: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    force: bool = False,
) -> str:
    """Download a pre-trained nnfd model from HuggingFace Hub.

    The model is cached locally so subsequent calls are instant (unless
    ``force=True``).

    Parameters
    ----------
    model_name:
        HuggingFace repository ID, e.g. ``"your-org/nnfd-maxwell-3d-v1"``.
    cache_dir:
        Root directory for cached models.
        Defaults to ``~/.cache/nnfd/``.
    token:
        HuggingFace user access token (read).  Required for private or gated
        repositories.  Can also be set via the ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``
        environment variable.
    force:
        Re-download even if a cached copy already exists.

    Returns
    -------
    str
        Path to the local model directory (contains ``models/nnfd_model.pt``),
        suitable for passing directly to :class:`~nnfd.solver.NN_solver`.
    """
    from huggingface_hub import hf_hub_download

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache_root = Path(cache_dir) if cache_dir else _DEFAULT_CACHE

    # Local destination: <cache_root>/<repo_id_flattened>/models/nnfd_model.pt
    safe_name  = model_name.replace("/", "--")
    local_dir  = cache_root / safe_name
    model_path = local_dir / _MODEL_SUBDIR / _MODEL_FILENAME

    if model_path.exists() and not force:
        return str(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / _MODEL_SUBDIR).mkdir(exist_ok=True)

    print(f"Downloading {model_name}/{_MODEL_SUBDIR}/{_MODEL_FILENAME} from HuggingFace Hub ...")
    hf_hub_download(
        repo_id=model_name,
        filename=f"{_MODEL_SUBDIR}/{_MODEL_FILENAME}",
        local_dir=str(local_dir),
        token=token,
        force_download=force,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Download appeared to succeed but {model_path} was not found. "
            "Check that the repository contains "
            f"'{_MODEL_SUBDIR}/{_MODEL_FILENAME}'."
        )

    print(f"Model cached at: {local_dir}")
    return str(local_dir)


def upload(
    model_path: str,
    repo_id: str,
    token: str,
    private: bool = True,
    commit_message: str = "Upload nnfd TorchScript model",
) -> str:
    """Upload a TorchScript nnfd model to HuggingFace Hub.

    Creates the repository if it does not exist.

    Parameters
    ----------
    model_path:
        Local directory containing ``models/nnfd_model.pt``
        (the standard nnfd checkpoint layout).
    repo_id:
        Target HuggingFace repository, e.g. ``"your-org/nnfd-maxwell-3d-v1"``.
    token:
        HuggingFace user access token with **write** permission.
    private:
        Create the repository as private (default: ``True``).
        Private repos are only accessible to the owner and collaborators.
        You can change this later in the HF web UI.
    commit_message:
        Commit message for the upload.

    Returns
    -------
    str
        URL of the repository on HuggingFace Hub.
    """
    from huggingface_hub import HfApi

    model_pt = Path(model_path) / _MODEL_SUBDIR / _MODEL_FILENAME
    if not model_pt.exists():
        raise FileNotFoundError(
            f"Expected checkpoint at {model_pt}.\n"
            "Run scripts/migrate_checkpoint.py first to create the TorchScript model."
        )

    api = HfApi(token=token)

    # Create repo (no-op if it already exists).
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    print(f"Repository: {repo_url}")

    # Upload the single model file, preserving the subdirectory structure.
    print(f"Uploading {model_pt} → {repo_id}/{_MODEL_SUBDIR}/{_MODEL_FILENAME} ...")
    api.upload_file(
        path_or_fileobj=str(model_pt),
        path_in_repo=f"{_MODEL_SUBDIR}/{_MODEL_FILENAME}",
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    print(f"Upload complete.  Repo URL: {repo_url}")
    print()
    print("Access control:")
    if private:
        print("  - Repository is PRIVATE.  Only you and invited collaborators can download it.")
        print("  - Invite collaborators at: https://huggingface.co/{repo_id}/settings/members")
        print("  - Collaborators download with:  nnfd.hub.download(repo_id, token='hf_...')")
    else:
        print("  - Repository is PUBLIC.")
    return str(repo_url)


def list_models():
    """List available pre-trained nnfd models (placeholder)."""
    raise NotImplementedError(
        "list_models() is not yet implemented. "
        "Browse available models at https://huggingface.co/models?search=nnfd"
    )
