from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from typing import Tuple, List, Dict, Any, Union, Optional
from typing import TYPE_CHECKING
from numpy.typing import NDArray

import colors

# ================================
# functions for plots
# ================================

# Set the custom color cycle globally without cycler
colors_lst, red, custom_cmap = colors.color_scheme()
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors_lst)


def importants_from_file(file_path: str, init_t: int = 0, final_t: int = None,
                         save: bool = False) -> None:

    # Set the custom color cycle globally without cycler
    colors_lst, red, custom_cmap = colors.color_scheme()
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors_lst)

    # ------ Pandas read dataframe ------
    df = pd.read_csv(file_path)

    # ------ Extract sizes ------
    F_measured = np.vstack([df["F_x_meas"].to_numpy(), df["F_y_meas"].to_numpy()])  # shape (2, T)
    F_desired = np.vstack([df["F_x_des"].to_numpy(), df["F_y_des"].to_numpy()])  # shape (2, T)
    loss_MSE = df["loss_MSE"].to_numpy()

    T = F_measured.shape[1]
    if final_t is None:
        final_t = T
    sl = slice(init_t, final_t)
    # t = np.arange(T)
    t = np.arange(T, dtype=int)

    fig, axs = plt.subplots(
        nrows=2, ncols=1, sharex=True,
        figsize=(6, 6),
        gridspec_kw={"height_ratios": [1, 1]}
    )

    # ===== top: forces =====
    axs[0].plot(t[sl], F_measured[0, sl], color=colors_lst[1], label="Fx measured")
    axs[0].plot(t[sl], F_desired[0, sl],  color=colors_lst[1], linestyle="--", label="Fx desired")

    axs[0].plot(t[sl], F_measured[1, sl], color=colors_lst[2], label="Fy measured")
    axs[0].plot(t[sl], F_desired[1, sl],  color=colors_lst[2], linestyle="--", label="Fy desired")

    axs[0].set_ylabel("Force [mN]")
    axs[0].set_ylim([-130, 330])
    axs[0].legend(loc="best")

    # ===== bottom: loss from file =====
    axs[1].plot(t[sl], loss_MSE[sl], color=colors_lst[0], label="loss x")

    # axs[1].axhline(0.0, linewidth=1)
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("Loss")
    axs[1].set_ylim([-0.02, 2.2])
    axs[1].legend(loc="best")

    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save:
        plt.savefig("importants.png", dpi=300, bbox_inches="tight")
    plt.show()


def force_global_during_measurement(t: NDArray, Fx_in_t: NDArray, Fy_in_t: NDArray) -> None:
    plt.plot(t, Fx_in_t)
    plt.plot(t, Fy_in_t)
    plt.xlabel(r"$t\,[\mathrm{s}]$")
    plt.ylabel(r"$F\,[\mathrm{mN}]$")
    plt.legend(['Fx', 'Fy'])
    plt.ylim([0, 150])
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
