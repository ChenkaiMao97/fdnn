import numpy as np
import torch
from typing import Callable, List, Optional, Tuple, Union
import itertools

s_function_type = Callable[[float], float]

GridSpacing = Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[
    np.ndarray, np.ndarray, np.ndarray]]

PmlLayers = Tuple[int, int, int, int, int, int]

def ln_R_exp_schedule(epoch: int, max_epoch: int, ln_R_start: float = -1, ln_R_end: float = -16, mult=10):
    # apply a exponential schedule to ln_R
    x = mult * epoch/(max_epoch-1)

    ln_R = np.exp(-(x-np.log(ln_R_start-ln_R_end))) + ln_R_end
    # when x = 0, ln_R = ln_R_start
    # when x = max_epoch-1, ln_R ~ ln_R_end
    return ln_R


def prepare_s_function(ln_R: float = -16, m: float = 4) -> s_function_type:
    """Create an s_function to pass to the SCPML functions.

    This is used when you would like to customize the PML parameters.

    Args:
        ln_R: Natural logarithm of the desired reflectance.
        m: Polynomial order for the PML (imaginary part increases as
            `distance**m`).

    Returns:
        An s_function, which takes an ndarray (distances) and returns an ndarray
        (complex part of the cell width; needs to be divided by
         `sqrt(epilon_effective) * real(omega)` before use).
    """

    def s_factor(distance: np.ndarray) -> np.ndarray:
        s_max = (m + 1) * ln_R / 2  # / 2 because we assume periodic boundaries
        return s_max * (distance**m)

    return s_factor

def stretch_with_scpml(
        dxes: GridSpacing,
        axis: int,
        polarity: int,
        omega: float,
        epsilon_effective: float = 1.0,
        thickness: int = 10,
        s_function: s_function_type = None,
        ln_R: float = -16,
        m: float = 4,
) -> GridSpacing:
    """
        Stretch dxes to contain a stretched-coordinate PML (SCPML) in one direction along one axis.

        :param dxes: dx_tuple with coordinates to stretch
        :param axis: axis to stretch (0=x, 1=y, 2=z)
        :param polarity: direction to stretch (-1 for -ve, +1 for +ve)
        :param omega: Angular frequency for the simulation
        :param epsilon_effective: Effective epsilon of the PML. Match this to the material at the
            edge of your grid. Default 1.
        :param thickness: number of cells to use for pml (default 10)
        :param s_function: s_function created by prepare_s_function(...), allowing customization
            of pml parameters. Default uses prepare_s_function() with no parameters.
        :return: Complex cell widths
    """
    if s_function is None:
        s_function = prepare_s_function(ln_R=ln_R, m=m)

    dx_ai = dxes[0][axis].astype(complex)
    dx_bi = dxes[1][axis].astype(complex)

    pos = np.hstack((0, dx_ai.cumsum()))
    pos_a = (pos[:-1] + pos[1:]) / 2
    pos_b = pos[:-1]

    # divide by this to adjust for epsilon_effective and omega
    s_correction = np.sqrt(epsilon_effective) * np.real(omega)

    if polarity > 0:
        # front pml
        bound = pos[thickness]
        d = bound - pos[0]

        def l_d(x):
            return (bound - x) / (bound - pos[0])

        slc = slice(thickness)

    else:
        # back pml
        bound = pos[-thickness - 1]
        d = pos[-1] - bound

        def l_d(x):
            return (x - bound) / (pos[-1] - bound)

        if thickness == 0:
            slc = slice(None)
        else:
            slc = slice(-thickness, None)

    dx_ai[slc] *= 1 + 1j * s_function(l_d(pos_a[slc])) / d / s_correction
    dx_bi[slc] *= 1 + 1j * s_function(l_d(pos_b[slc])) / d / s_correction

    dxes[0][axis] = dx_ai
    dxes[1][axis] = dx_bi

    return dxes

def apply_scpml(dxes: GridSpacing,
                pml_layers: Optional[Union[int, PmlLayers]],
                omega: float,
                ln_R: float = -16,
                m: float = 4) -> GridSpacing:
    """Applies PMLs to the grid spacing.

    This function implements SC-PMLs by modifying the grid spacing based on
    the PML layers.

    Args:
        dxes: Grid spacing to modify.
        pml_layers: Indicates number of PML layers to apply on each side. If
            `None`, no PMLs are applied. If this is a scalar, the same number of
            PML layers are applied to each side.
        omega: Frequency of PML operation.

    Returns:
        A new grid spacing with SC-PML applied.
    """
    # Make a copy of `dxes`. We write this out so that we have an array of
    # array of numpy arrays.
    dxes = [
        [np.array(dxes[grid_num][i]) for i in range(3)] for grid_num in range(2)
    ]

    if isinstance(pml_layers, torch.Tensor):
        assert pml_layers.shape == (6,) or pml_layers.shape[0] == 1, f"unexpected pml_layers shape: {pml_layers.shape}"
        pml_layers = pml_layers.cpu().squeeze().numpy()

    if np.sum(np.abs(np.array(pml_layers))) == 0:
        return dxes

    if isinstance(pml_layers, int):
        pml_layers = [pml_layers] * 6

    for pml, (axis, polarity) in zip(pml_layers,
                                     itertools.product(range(3), [1, -1])):
        if pml > 0:
            dxes = stretch_with_scpml(
                dxes, omega=omega, axis=axis, polarity=polarity, thickness=pml, ln_R=ln_R, m=m)

    return dxes