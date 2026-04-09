from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

import colors

# ================================
# functions for plots
# ================================

# Set the custom color cycle globally
colors_lst, red, custom_cmap = colors.color_scheme()
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors_lst)


def importants_from_file(file_path: str, init_t: int = 0, final_t: Optional[int] = None,
                         save: bool = False) -> None:
    """Plot measured/designed forces and loss from an exported training CSV.

    Parameters
    ----------
    file_path : str. Path to CSV file, typically created by ``file_helpers.export_training_csv()``.
    init_t    : int, optional. First plotted time index.
    final_t   : int | None, optional. Final plotted time index (exclusive). If omitted, plots until the end.
    save      : bool, optional. If ``True``, save figure as ``importants.png``.
    """
    # read dataframe
    df = pd.read_csv(file_path)

    F_measured = np.vstack([df["F_x_meas"].to_numpy(dtype=float),
                            df["F_y_meas"].to_numpy(dtype=float)])  # shape (2, T)
    F_desired = np.vstack([df["F_x_des"].to_numpy(dtype=float),
                           df["F_y_des"].to_numpy(dtype=float)])  # shape (2, T)
    loss_MSE = df["loss_MSE"].to_numpy(dtype=float)

    # time steps
    T = int(F_measured.shape[1])
    if final_t is None:
        final_t = T
    sl = slice(init_t, final_t)
    t = np.arange(T, dtype=int)

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 6), gridspec_kw={"height_ratios": [1, 1]})

    # ===== top: forces =====
    axs[0].plot(t[sl], F_measured[0, sl], color=colors_lst[1], label="Fx measured")
    axs[0].plot(t[sl], F_desired[0, sl], color=colors_lst[1], linestyle="--", label="Fx desired")

    axs[0].plot(t[sl], F_measured[1, sl], color=colors_lst[2], label="Fy measured")
    axs[0].plot(t[sl], F_desired[1, sl], color=colors_lst[2], linestyle="--", label="Fy desired")

    axs[0].set_ylabel("Force [mN]")
    axs[0].set_ylim([-130, 330])
    axs[0].legend(loc="best")

    # ===== bottom: MSE loss =====
    axs[1].plot(t[sl], loss_MSE[sl], color=colors_lst[0], label="loss MSE")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("Loss")
    axs[1].set_ylim([-0.02, 2.2])
    axs[1].legend(loc="best")

    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save:
        plt.savefig("importants.png", dpi=300, bbox_inches="tight")
    plt.show()


def force_global_during_measurement(t: NDArray[np.float64],
                                    Fx_in_t: NDArray[np.float64], Fy_in_t: NDArray[np.float64]) -> None:
    """Plot global-frame force components during one measurement.
    Used inside Supervisor.force_global_during_measurement()

    Parameters
    ----------
    t       : NDArray[np.float64]. Measurement time vector [s], shape ``(T_meas,)``.
    Fx_in_t : NDArray[np.float64]. Global x-force trace [mN], shape ``(T_meas,)``.
    Fy_in_t : NDArray[np.float64]. Global y-force trace [mN], shape ``(T_meas,)``.
    """
    t_arr = np.asarray(t, dtype=float)
    Fx_arr = np.asarray(Fx_in_t, dtype=float)
    Fy_arr = np.asarray(Fy_in_t, dtype=float)

    if t_arr.ndim != 1 or Fx_arr.ndim != 1 or Fy_arr.ndim != 1:
        raise ValueError("t, Fx_in_t, and Fy_in_t must all be 1D arrays.")
    if not (t_arr.shape == Fx_arr.shape == Fy_arr.shape):
        raise ValueError(f"t, Fx_in_t, and Fy_in_t must have the same shape. "
                         f"Got {t_arr.shape}, {Fx_arr.shape}, {Fy_arr.shape}.")

    plt.plot(t_arr, Fx_arr)
    plt.plot(t_arr, Fy_arr)
    plt.xlabel(r"$t\,[\mathrm{s}]$")
    plt.ylabel(r"$F\,[\mathrm{mN}]$")
    plt.legend(["Fx", "Fy"])
    plt.show()


def calibration_forces(voltages_arr: NDArray[np.float64], forces_arr: NDArray[np.float64],
                       stds_arr: NDArray[np.float64], fit_params: NDArray[np.float64]) -> None:
    """Plot tri-axial calibration data and fitted force-voltage lines.
    Used inside ForsentekClass.calibrate_daily()

    Parameters
    ----------
    voltages_arr : array. Mean voltage samples [V], shape ``(N, 3)``.
    forces_arr   : array. Force samples [N], shape ``(N, 3)``.
    stds_arr     : array. Voltage standard deviations [V], shape ``(N, 3)``.
    fit_params   : array. Linear fit parameters of shape ``(2, 3)``,
                          with row 0 = intercepts and row 1 = slopes.

    Notes
    -----
    Uncertainty is propagated approximately as ``σ_F ≈ |a| σ_V``, where
    ``a`` is the fitted slope for each axis.
    """
    # caution and raises
    V = np.asarray(voltages_arr, dtype=float)
    F = np.asarray(forces_arr, dtype=float)
    sV = np.asarray(stds_arr, dtype=float)
    params = np.asarray(fit_params, dtype=float)

    if V.shape != F.shape or V.shape != sV.shape:
        raise ValueError(f"voltages_arr, forces_arr, and stds_arr must have the same shape. "
                         f"Got {V.shape}, {F.shape}, {sV.shape}.")
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"Input arrays must have shape (N, 3). Got {V.shape}.")
    if params.shape != (2, 3):
        raise ValueError(f"fit_params must have shape (2, 3). Got {params.shape}.")

    a = np.asarray(params[1, :], dtype=float)  # slopes
    b = np.asarray(params[0, :], dtype=float)  # intercepts

    # propagate uncertainty: σ_F ≈ |a| σ_V
    sF = np.abs(a)[None, :] * sV

    # create figure and labels
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8), constrained_layout=True)
    axis_lbls = ["x", "y", "z"]

    # plot 3 plots for 3 axes
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
