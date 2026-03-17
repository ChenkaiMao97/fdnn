"""
Model registry for fdnn.

To register your model class so the package can instantiate it from a
checkpoint's "model_type" string, use the @register_model decorator:

    from fdnn._model import register_model
    import torch.nn as nn

    @register_model("MyModel")
    class MyModel(nn.Module):
        def __init__(self, **kwargs):   # match your model's actual __init__ signature
            super().__init__()
            ...

        def setup(self, eps, freq):
            # called once before each GMRES solve
            ...

        def forward(self, x, freq):
            # x: (bs, sx, sy, sz, 6) real residual
            # returns: (bs, sx, sy, sz, 6) preconditioned residual
            ...

The model class must implement:
  - __init__(self, domain_sizes, paddings, **model_kwargs)
  - setup(self, eps, freq)   — cache eps/freq before the GMRES loop
  - forward(self, x, freq)   — apply the preconditioner
"""

from typing import Any, Dict, Type
import torch.nn as nn

_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class under a string key."""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _REGISTRY:
            raise ValueError(
                f"A model named '{name}' is already registered. "
                "Choose a different name or remove the existing registration."
            )
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_model(model_type: str, model_kwargs: Dict[str, Any]) -> nn.Module:
    """Instantiate a registered model from its type string and kwargs.

    Args:
        model_type:   String key used when the model was registered.
        model_kwargs: Dict passed as keyword arguments to the constructor.

    Returns:
        An nn.Module instance (weights not yet loaded).
    """
    if model_type not in _REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Registered models: {sorted(_REGISTRY.keys())}.\n"
            "Register your model class with @fdnn._model.register_model('<name>')."
        )
    return _REGISTRY[model_type](**model_kwargs)
