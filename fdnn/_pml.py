"""
PML feature channel builder.

``build_pml_channels`` precomputes the PML feature tensor that gets
concatenated to the permittivity (eps) before being fed to the neural
network preconditioner.  Its output mirrors what
``SyntheticDataset.build_complex_eps()`` produced during training.

TODO
----
Paste the relevant logic from
``waveynet3d/data/simulation_dataset.py :: build_complex_eps`` here.

The function must satisfy:

    pml_channels = build_pml_channels(sim_shape, wavelength, dL,
                                      pml_layers, ln_R, device)

    # pml_channels: torch.Tensor, shape (1, sx, sy, sz, n_pml_channels)
    #               dtype: float32, device: as requested
    #               channel 0 of the *concatenated* tensor is eps itself;
    #               these are channels 1..n.

Once filled in, the solver will call:

    eps_with_pml = torch.cat([eps_chunk.unsqueeze(-1),
                               pml_channels.expand(bs, -1,-1,-1,-1)], dim=-1)
    # shape: (bs, sx, sy, sz, 1 + n_pml_channels)
"""

import torch
from typing import Optional, Tuple


def build_pml_channels(
    sim_shape: Tuple[int, int, int],
    wavelength: float,
    dL: float,
    pml_layers: Tuple[int, int, int, int, int, int],
    ln_R: float,
    device: str,
) -> torch.Tensor:
    """Precompute the PML feature channels for a given simulation geometry.

    Parameters
    ----------
    sim_shape:   (sx, sy, sz) voxel dimensions of the simulation domain.
    wavelength:  Free-space wavelength (same units as dL).
    dL:          Grid spacing.
    pml_layers:  PML thickness on each face: (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi).
    ln_R:        Natural log of the desired PML reflectance (e.g. -16).
    device:      Torch device string, e.g. 'cuda:0' or 'cpu'.

    Returns
    -------
    torch.Tensor of shape (1, sx, sy, sz, n_pml_channels), dtype float32.
    """
    # ------------------------------------------------------------------ #
    # TODO: paste the body of build_complex_eps (channels 1..) here.     #
    # Remove the NotImplementedError once filled in.                      #
    # ------------------------------------------------------------------ #
    raise NotImplementedError(
        "build_pml_channels is not yet implemented.\n"
        "Open fdnn/_pml.py and paste the logic from "
        "waveynet3d/data/simulation_dataset.py :: build_complex_eps "
        "(the part that produces channels 1..n)."
    )
