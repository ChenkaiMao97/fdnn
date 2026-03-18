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
import numpy as np
from typing import Optional, Tuple

from fdnn.PML_utils import apply_scpml

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
    
    omega = 2 * np.pi / wavelength  # natural unit with C_0 = 1
    dxes = (
        [np.array([dL] * sim_shape[0]), np.array([dL] * sim_shape[1]), np.array([dL] * sim_shape[2])],
        [np.array([dL] * sim_shape[0]), np.array([dL] * sim_shape[1]), np.array([dL] * sim_shape[2])],
    )

    dxes = apply_scpml(dxes, pml_layers, omega, ln_R=ln_R)

    masks = []
    if pml_layers[0] > 0 or pml_layers[1] > 0:
        masks.append(
            torch.from_numpy(dxes[0][0][:, None, None] / dL)
            .repeat(1, sim_shape[1], sim_shape[2]).imag.float()
        )
    if pml_layers[2] > 0 or pml_layers[3] > 0:
        masks.append(
            torch.from_numpy(dxes[0][1][None, :, None] / dL)
            .repeat(sim_shape[0], 1, sim_shape[2]).imag.float()
        )
    if pml_layers[4] > 0 or pml_layers[5] > 0:
        masks.append(
            torch.from_numpy(dxes[0][2][None, None, :] / dL)
            .repeat(sim_shape[0], sim_shape[1], 1).imag.float()
        )

    pml_channel = torch.stack(masks, dim=-1)  # (sx, sy, sz, n_pml_ch)
    return pml_channel.unsqueeze(0).to(device)  # (1, sx, sy, sz, n_pml_ch)
