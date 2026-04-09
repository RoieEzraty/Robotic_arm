from __future__ import annotations

import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.typing import NDArray


def fit_force_vs_voltage(voltages_array: NDArray[np.float64], forces_array: NDArray[np.float64],
                         std_array: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
    """Fit a linear force-voltage relation independently for x, y, and z: ``F = slope * V + intercept``
    used inside Experiments.calibrate_forces_all_axes

    Parameters
    ----------
    voltages_array : NDArray[np.float64]. Voltage samples of shape ``(N, 3)``.
    forces_array   : NDArray[np.float64]. Force samples of shape ``(N, 3)``.
    std_array      : NDArray[np.float64] | None, optional. Standard deviation of voltage samples, shape ``(N, 3)``.
                     If provided, performs weighted least squares with weights ``w = 1 / std^2``.
                     If omitted, performs ordinary least squares.

    Returns
    -------
     Array of shape ``(2, 3)`` containing fitted parameters for each axis:
        row 0 = intercepts, row 1 = slopes.

    Notes
    -----
    - ``fit_params[0, :] = intercept``
      ``fit_params[1, :] = slope``
    - Non-positive standard deviations are replaced by median of positive values in the same axis. 
      If no positive values exist, unit weights are used for that axis.
    """
    # initialize arrays and caution raises
    voltages = np.asarray(voltages_array, dtype=float)
    forces = np.asarray(forces_array, dtype=float)

    if voltages.shape != forces.shape:
        raise ValueError(f"voltages_array and forces_array must have the same shape. "
                         f"Got {voltages.shape} and {forces.shape}.")
    if voltages.ndim != 2 or voltages.shape[1] != 3:
        raise ValueError(f"voltages_array and forces_array must both be of shape (N, 3). "f"Got {voltages.shape}.")

    if std_array is not None:
        stds = np.asarray(std_array, dtype=float)
        if stds.shape != voltages.shape:
            raise ValueError(f"std_array must have the same shape as voltages_array. "
                             f"Got {stds.shape} and {voltages.shape}.")
    else:
        stds = None

    fit_params = np.zeros((2, 3), dtype=float)

    # polynomial fit for every axis separately
    for dim in range(3):
        if stds is None:
            intercept, slope = poly.polyfit(voltages[:, dim], forces[:, dim], deg=1)
        else:
            std_dim = np.asarray(stds[:, dim], dtype=float)
            positive_std = std_dim[std_dim > 0]

            if positive_std.size == 0:
                weights = np.ones_like(std_dim, dtype=float)
            else:
                replacement_std = float(np.median(positive_std))
                std_dim = np.where(std_dim <= 0, replacement_std, std_dim)
                weights = 1.0 / (std_dim ** 2)

            intercept, slope = poly.polyfit(voltages[:, dim], forces[:, dim], deg=1, w=weights)

        fit_params[:, dim] = np.array([intercept, slope], dtype=float)

    return fit_params


def get_total_angle(L: float, tip_pos: NDArray[np.float64], prev_total_angle: float) -> float:
    """Compute unwrapped total chain angle from current tip position.
    Used inside MecaClass.clamp_to_circle_xy()

    Angle measured using the vector from tip position to end of base facet ``(L, 0)``, 
    with zero aligned along negative x direction. 
    Then unwrapped relative to ``prev_total_angle`` so it can exceed ``±180`` and ``±360`` degrees continuously.

    Parameters
    ----------
    L                : float. Edge length [mm], used to define the reference point ``(L, 0)``.
    tip_pos          : NDArray[np.float64], Current tip position of shape ``(2,)`` as ``[x, y]`` [mm].
    prev_total_angle : float. Previously accumulated unwrapped angle [deg].

    Returns
    -------
    New unwrapped total angle [deg].
    """
    # caution raises
    tip_pos_arr = np.asarray(tip_pos, dtype=float)
    if tip_pos_arr.shape != (2,):
        raise ValueError(f"tip_pos must have shape (2,), got {tip_pos_arr.shape}.")

    # previous total angle
    prev_total_angle_rads = np.deg2rad(prev_total_angle)

    # origin from which to calculate angle
    angle_origin = np.array([L, 0.0], dtype=float)
    dx, dy = angle_origin - tip_pos_arr

    total_angle = np.arctan2(dy, dx) - np.pi  # angle addition
    total_angle = (total_angle + np.pi) % (2.0 * np.pi) - np.pi  # unwrap so angle can exceed 180

    # unwrap
    prev_theta_wrapped = ((prev_total_angle_rads + np.pi) % (2.0 * np.pi)) - np.pi
    delta = total_angle - prev_theta_wrapped

    if delta > np.pi:
        delta -= 2.0 * np.pi
    elif delta < -np.pi:
        delta += 2.0 * np.pi

    total_angle = prev_total_angle_rads + delta
    return float(np.rad2deg(total_angle))


def fit_circle_xy(points_xy: NDArray[np.float64]) -> float:
    """Fit a circle to planar points and return only its radius. 
    Uses least-squares Kåsa-style fit on points ``(x, y)``.
    Used once upon init of MecaClass to calculate R_robot

    Parameters
    ----------
    points_xy : NDArray[np.float64]. Array of shape ``(N, 2)``, columns are ``[x, y]``.

    Returns
    -------
    float, Fitted circle radius ``R`` [same length units as input].
    """
    # caution raises
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xy must have shape (N, 2), got {points.shape}.")
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a circle.")

    x = points[:, 0]
    y = points[:, 1]

    # circle geometry
    A = np.c_[x, y, np.ones_like(x)]
    b = -(x**2 + y**2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coef, c = sol

    cx = -a / 2.0
    cy = -b_coef / 2.0
    radius = np.sqrt(max(0.0, cx**2 + cy**2 - c))
    return float(radius)


def effective_radius(R: float, L: float,  total_angle: float, tip_angle: float, margin: float = 0.0) -> float:
    """Compute the remaining effective chain radius after winding.
    shrink model is composed of:
    - ``2 * L`` shrink per each full tip revolution.
    - ``L * (1 - cos(rem))`` for the remaining partial revolution.
    Used inside MecaClass.clamp_to_circle_xy

    Parameters
    ----------
    R           : float. Maximal chain length [mm].
    L           : float. Chain link length [mm].
    total_angle : float. Unwrapped total chain angle [deg].
    tip_angle   : float. Current tip angle [deg].
    margin      : float, optional. Additional safety margin subtracted from the radius [mm].

    Returns
    -------
    float, Effective allowed radius [mm], clipped at zero.
    """
    # caution, prints and prelims.
    print(f"total_angle={total_angle:.2f}, tip_angle={tip_angle:.2f}")
    two_pi = 2.0 * np.pi

    # change in angle
    delta = float(np.abs(np.deg2rad(total_angle) - np.deg2rad(tip_angle)))

    # number of full revolutions
    n_rev = int(np.floor(delta / two_pi))
    rem = delta - n_rev * two_pi

    # effective chain radius shrink due to full tip revolutions
    shrink_full = (2.0 * L) * n_rev
    print("shrink due to revolutions", shrink_full)

    # effective chain radius shrink due to partial tip revolutions
    shrink_partial = L * (1.0 - np.cos(rem))
    print("shrink remainder in [mm]", shrink_partial)

    # sum both shrinks
    shrink = shrink_full + shrink_partial
    return float(max(0.0, (R - margin) - shrink))


def rotate_force_frame(force_in_t: NDArray[np.float64], tip_angle: float,) -> tuple[NDArray[np.float64],
                                                                                    NDArray[np.float64]]:
    """Rotate measured 2D force traces into  global frame.
    Used inside SupervisorClass.global_force()

    Parameters
    ----------
    force_in_t : array, Local-frame force samples of shape ``(T_meas, 2)``, taken from sensor
    tip_angle  : float, Tip angle relative to the origin [deg].

    Returns
    -------
    Fx_global_in_t,  Fy_global_in_t: Global-frame forces, each shaped ``(T_meas,)``.
    """
    # caution and raises
    forces = np.asarray(force_in_t, dtype=float)
    if forces.ndim != 2 or forces.shape[1] < 2:
        raise ValueError(f"force_in_t must have shape (T_meas, 2) or larger in second axis, got {forces.shape}.")

    tip_angle_rad = np.deg2rad(tip_angle)

    # rotation with sines and cosines (instead of matrix multip)
    Fx_global_in_t = -(forces[:, 0] * np.cos(tip_angle_rad) +
                       forces[:, 1] * np.sin(tip_angle_rad))
    Fy_global_in_t = -(-forces[:, 0] * np.sin(tip_angle_rad) +
                       forces[:, 1] * np.cos(tip_angle_rad))
    return Fx_global_in_t, Fy_global_in_t


def TRF_to_robot_tip(x: float, y: float, theta_z: float, tx: float) -> tuple[float, float]:
    """Convert chain-tip planar coordinates to robot-tip planar coordinates.
    Used inside MecaClass.clamp_to_circle_xy()

    Parameters
    ----------
    x, y, theta_z : floats, Chain-tip coordinates and orientation [mm], [mm], [deg].
    tx            : float, Tool-frame x offset between robot tip and chain tip [mm].

    Returns
    -------
    float, Robot-tip planar coordinates ``(x_robot, y_robot)`` [mm].
    """
    dx = float(np.cos(np.deg2rad(theta_z)) * tx)
    dy = float(np.sin(np.deg2rad(theta_z)) * tx)
    return float(x - dx), float(y - dy)
