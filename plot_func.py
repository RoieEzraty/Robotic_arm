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


def calibration_forces(voltages_arr, forces_arr, stds_arr, fit_params) -> None:
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors_lst)

    V = np.asarray(voltages_arr, dtype=float)   # (N,3)
    F = np.asarray(forces_arr, dtype=float)     # (N,3)
    sV = np.asarray(stds_arr, dtype=float)      # (N,3)

    a = np.asarray(fit_params[1, :], dtype=float)  # (3,)
    b = np.asarray(fit_params[0, :], dtype=float)  # (3,)

    # propagate uncertainty: σ_F ≈ |a| σ_V
    sF = np.abs(a)[None, :] * sV                # (N,3)

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8), constrained_layout=True)

    axis_lbls = ["x", "y", "z"]

    for k, ax in enumerate(axs):
        V_line = np.linspace(V[:, k].min(), V[:, k].max(), 300)
        F_line = a[k] * V_line + b[k]

        ax.errorbar(V[:, k], F[:, k], yerr=sF[:, k], fmt=".", capsize=3, label="data (±1σ)")
        ax.plot(V_line, F_line, "-", label=fr"fit: $F={a[k]:.4g}V+{b[k]:.4g}$")

        ax.set_ylabel(fr"$F_{axis_lbls[k]}\,[\mathrm{{N}}]$")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    axs[-1].set_xlabel(r"$V\,[\mathrm{V}]$")

    plt.show()
