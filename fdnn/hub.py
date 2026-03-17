"""
Model hub utilities for fdnn.

Current status
--------------
Models are loaded from a local directory.  Point to one via the
``model_path`` argument of :class:`~fdnn.solver.NN_solver` or by setting
the ``FDNN_MODEL_PATH`` environment variable.

Planned
-------
When models are published, this module will support:

* ``fdnn.hub.list_models()``   — list available pre-trained models.
* ``fdnn.hub.download(name)``  — fetch a model from HuggingFace Hub and
  cache it locally (default: ``~/.cache/fdnn/``).

Example (future API)::

    import fdnn

    model_path = fdnn.hub.download("fdnn/maxwell-3d-v1")
    solver = fdnn.NN_solver(model_path=model_path)
"""

from pathlib import Path
from typing import Optional


_DEFAULT_CACHE = Path.home() / ".cache" / "fdnn"


def list_models():
    """List available pre-trained models on the remote registry.

    .. note:: Not yet implemented.
    """
    raise NotImplementedError(
        "Remote model registry is not yet available. "
        "Use a local model_path or set FDNN_MODEL_PATH."
    )


def download(
    model_name: str,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """Download a pre-trained model from HuggingFace Hub.

    Parameters
    ----------
    model_name:
        Repository identifier, e.g. ``"fdnn/maxwell-3d-v1"``.
    cache_dir:
        Local directory for downloaded models.
        Defaults to ``~/.cache/fdnn/``.
    force:
        Re-download even if a cached copy exists.

    Returns
    -------
    str
        Local path to the downloaded model directory, ready to pass to
        :class:`~fdnn.solver.NN_solver`.

    .. note:: Not yet implemented.
    """
    raise NotImplementedError(
        "Remote model download is not yet available. "
        "Use a local model_path or set FDNN_MODEL_PATH."
    )
