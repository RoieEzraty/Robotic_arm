import ast

import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.typing import NDArray


def fit_force_vs_voltage(voltages_array: np.ndarray, forces_array: np.ndarray,
                         std_array: np.ndarray | None = None) -> NDArray[np.float64]:
    """
    Fit F = a*V + b separately for x/y/z.

    Inputs:
        voltages_array: (N,3)
        forces_array:   (N,3)
        stds_voltage_array: (N,3) or None

    Returns:
        fit_params_array: (2,3) where:
            fit_params_array[0, :] = [ax, ay, az]
            fit_params_array[1, :] = [bx, by, bz]
    """
    fit_params = np.zeros((2, 3), dtype=float)
    for dim in range(3):
        if std_array is None:
            # ordinary least squares
            a, b = poly.polyfit(voltages_array[:, dim], forces_array[:, dim], deg=1)           
        else:
            sV = np.asarray(std_array[:, dim], dtype=float)
            # avoid divide-by-zero
            sV = np.where(sV <= 0, np.nanmedian(sV[sV > 0]), sV)
            w = 1.0 / (sV**2)
            a, b = poly.polyfit(voltages_array[:, dim], forces_array[:, dim], deg=1, w=w)
        fit_params[:, dim] = np.asarray(np.array([a, b]), dtype=float).reshape(2,)
    return fit_params


def get_total_angle(L: float, tip_pos: NDArray, prev_total_angle: float) -> float:
    """
    angle between tip and last fixed node, CCW, [deg]
    pos_arr: array of shape (H, 2)
    Returns: angle [deg] in [-180, 180], measured from -x axis
    
    Parameters
    ----------
    L                - float, Edge length (used to define reference point at (L,0)).
    tip_pos          - (2,) array, Current tip position.
    prev_total_angle - float, deg, The accumulated unwrapped angle up to the previous timestep.
    

    Returns
    -------
    new_total_angle : float, deg, The unwrapped angle  (can exceed ±180, ±360, ...).
    """
    # This calculation is in radians, robot thinks in degs
    prev_total_angle_rads = np.deg2rad(prev_total_angle)  # [rad]
    
    # ------ total angle [-pi/2, pi/2] ------
    # angle_origin = origin + np.array([L, 0])  # total angle measured from end of first link
    angle_origin = np.array([L, 0])  # total angle measured from end of first link
    dx, dy = angle_origin - tip_pos  # displacement vector

    # shift so that 0 is along -x
    total_angle = np.arctan2(dy, dx) - np.pi
    # normalize back to [-pi, pi]
    total_angle = (total_angle + np.pi) % (2*np.pi) - np.pi

    # ------ correct for wrapping around center ------
    prev_theta_wrapped = ((prev_total_angle_rads + np.pi) % (2*np.pi)) - np.pi  # [rad]
    delta = total_angle - prev_theta_wrapped  # [rad]

    # correct jumpt across -x axis - adding or subtracting 2π
    if delta > np.pi:
        delta -= 2*np.pi
    elif delta < -np.pi:
        delta += 2*np.pi

    # Update cumulative angle
    total_angle = prev_total_angle_rads + delta

    return np.rad2deg(total_angle)  # [deg]


def fit_circle_xy(points_xy: np.ndarray):
    """
    Least-squares circle fit (Kåsa-style).
    points_xy: (N,2) array with columns [x,y]
    returns: (cx, cy, R)
    """
    x = points_xy[:, 0].astype(float)
    y = points_xy[:, 1].astype(float)

    # Solve: x^2 + y^2 + A x + B y + C = 0
    A = np.c_[x, y, np.ones_like(x)]
    b = -(x**2 + y**2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = sol

    cx = -a / 2.0
    cy = -b_ / 2.0
    R = np.sqrt(max(0.0, cx**2 + cy**2 - c))
    return float(R)


def effective_radius(R, L, total_angle, tip_angle, margin=0.0) -> float:
    # total_angle should be *unwrapped* (can exceed 360)
    print(f'total_angle={total_angle:.2f}, tip_angle={tip_angle:.2f}')
    delta = float(np.abs(np.deg2rad(total_angle) - np.deg2rad(tip_angle)))  # radians, unwrapped
    two_pi = 2.0 * np.pi
    n_rev = int(np.floor(delta / two_pi))
    rem = delta - n_rev * two_pi  # in [0, 2π)

    shrink_full = (2.0 * L) * n_rev
    print('shrink due to revolutions', shrink_full)
    shrink_partial = L * (1.0 - np.cos(rem / 2.0))  # in [0, 2L)
    # shrink_partial = L * (1.0 - np.cos(rem))  # in [0, 2L)
    print('shrink remainder in [mm]', shrink_partial)

    shrink = shrink_full + shrink_partial
    return max(0.0, (R - margin) - shrink)


def rotate_force_frame(force_in_t, tip_angle):
    """
    force_in_t: NDArray (T_meas, 2), of 2d forces during sensor measurement
    tip_angle : float, deg, tip_angle of robot relative to origin
    """
    tip_angle_rad = np.deg2rad(tip_angle)
    Fx_global_in_t = -(force_in_t[:, 0]*np.cos(tip_angle_rad) + force_in_t[:, 1]*np.sin(tip_angle_rad))
    Fy_global_in_t = -(-force_in_t[:, 0]*np.sin(tip_angle_rad) + force_in_t[:, 1]*np.cos(tip_angle_rad))
    # Fx_global_in_t = force_in_t[:, 0]*np.cos(tip_angle_rad) - force_in_t[:, 1]*np.sin(tip_angle_rad)
    # Fy_global_in_t = force_in_t[:, 0]*np.sin(tip_angle_rad) + force_in_t[:, 1]*np.cos(tip_angle_rad)
    return Fx_global_in_t, Fy_global_in_t


def TRF_to_robot_tip(x, y, theta_z, tx):
    # rotate tool offset into world/WRF
    dx = np.cos(np.deg2rad(theta_z))*tx
    dy = np.sin(np.deg2rad(theta_z))*tx
    return x - dx, y - dy


def cfg_get_vec2(cfg, section, key, fallback=None):
    s = cfg.get(section, key, fallback=None)
    if s is None:
        return fallback
    v = ast.literal_eval(s)          # parses "[83.2, -11.0]" -> list
    v = np.asarray(v, dtype=float)
    if v.shape != (2,):
        raise ValueError(f"{section}.{key} must be length-2, got {v}")
    return v


def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


def wrap_deg(a):
    # wrap to [-180, 180)
    return float(((a + 180.0) % 360.0) - 180.0)
