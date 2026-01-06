from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, List, Dict, Any, Union, Optional
from typing import TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from Network_State import Network_State
    from Big_Class import Big_Class

import colors

# ================================
# functions for plots
# ================================


def importants() -> None:

    # Set the custom color cycle globally without cycler
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors_lst)

    # # Create main grid: 3 rows (loss, buckles, input) with buckle region smaller with custom ratios
    # fig = plt.figure(figsize=(4.6, 2.2 + 1.2*n_springs))
    # gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, n_springs*0.75, 1.2], figure=fig)

    plt.tight_layout()
    plt.show()


def calibration_forces(voltages_vec, forces_vec, stds_vec, fit_params) -> None:
    # Set the custom color cycle globally without cycler
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors_lst)

    a, b = fit_params[0], fit_params[1]

    V = np.asarray(voltages_vec, dtype=float)
    F = np.asarray(forces_vec, dtype=float)
    sV = np.asarray(stds_vec, dtype=float)
    # Convert voltage std -> force std (propagate uncertainty through F = aV + b)
    sF = np.abs(a) * sV

    V_line = np.linspace(V.min(), V.max(), 200)
    F_line = a * V_line + b

    # plt.plot(V, F, ".", label="data")
    plt.errorbar(V, F, yerr=sF, fmt=".", capsize=3, label="data (±1σ)")
    plt.plot(V_line, F_line, "-", label=f"fit: F = {a:.4g}·V + {b:.4g}")
    # data with y-error bars
    plt.xlabel(r"$V\,[\mathrm{V}]$")
    plt.ylabel(r"$F\,[\mathrm{N}]$")
    plt.legend()
    plt.tight_layout()
    plt.show()
