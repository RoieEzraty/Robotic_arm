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


def get_total_angle(origin, tip_pos, prev_total_angle):
    """
    angle between tip and last fixed node, CCW
    pos_arr: array of shape (H, 2)
    Returns: angle (radians) in [-pi, pi], measured from -x axis
    
    Parameters
    ----------
    tip_pos          - (2,) array, Current tip position.
    prev_total_angle - float, The accumulated unwrapped angle up to the previous timestep.
    L                - float, Edge length (used to define reference point at (L,0)).

    Returns
    -------
    new_total_angle : float, The unwrapped angle (can exceed ±pi, ±2pi, ±3pi, ...).
    """
    # This calculation is in radians, robot thinks in degs
    prev_total_angle_rads = np.deg2rad(prev_total_angle)
    
    # ------ total angle [-pi/2, pi/2] ------
    dx, dy = origin - tip_pos  # displacement vector

    # shift so that 0 is along -x
    total_angle = np.arctan2(dy, dx) - np.pi
    # normalize back to [-pi, pi]
    total_angle = (total_angle + np.pi) % (2*np.pi) - np.pi

    # ------ correct for wrapping around center ------
    prev_theta_wrapped = ((prev_total_angle_rads + np.pi) % (2*np.pi)) - np.pi
    delta = total_angle - prev_theta_wrapped

    # correct jumpt across -x axis - adding or subtracting 2π
    if delta > np.pi:
        delta -= 2*np.pi
    elif delta < -np.pi:
        delta += 2*np.pi

    # Update cumulative angle
    total_angle = prev_total_angle_rads + delta

    return total_angle


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
    return float(cx), float(cy), float(R)


def effective_radius(R, L, total_angle, tip_angle, margin=0.0) -> float:
    # total_angle_deg should be *unwrapped* (can exceed 360)
    print(f'L={L:.2f}, R={R:.2f}, total_angle={total_angle:.2f}, tip_angle={tip_angle:.2f}')
    # shrink = L * np.abs((total_angle - tip_angle) / 180.0)
    shrink = L * np.cos(np.abs(total_angle - tip_angle))
    return max(0.0, (R - margin) - shrink)


def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


def wrap_deg(a):
    # wrap to [-180, 180)
    return float(((a + 180.0) % 360.0) - 180.0)
