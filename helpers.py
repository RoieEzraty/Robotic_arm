from __future__ import annotations

import numpy as np
from numpy import array, zeros
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


def coil(angle: float, revolutions: float = 1.5, units='deg'):
    """
    return boolean whether the tip coiled too much

    Parameters:
    -----------
    angle       - float, angle during update state after corrections 
    revolutions - float, how many 2pi revolution allowed before angle is considered as too much coiled

    Returns:
    --------
    boolean - True=coiled too much, correct inside SupervisorClass.calc_update
    """
    if units == 'deg':
        return np.abs(np.deg2rad(angle)) > revolutions * 2*np.pi
    else:
        return np.abs(angle) > revolutions * 2*np.pi


def swept_last_edge_crosses_first_edge(tip_prev: np.ndarray, angle_prev: float, tip_new: np.ndarray,
                                       angle_new: float, L: float, *, n_samples: int = 101, eps: float = 1e-12,
                                       include_endpoints: bool = False) -> bool:
    """
    Return whether the quadrilateral swept by the last edge crosses the first edge.

    The swept quadrilateral is taken as:
        before_prev -> tip_prev -> tip_new -> before_new

    The first edge is:
        [ [0,0], [L,0] ]

    Parameters
    ----------
    before_prev, tip_prev, before_new, tip_new
        Endpoints of the old and new last edge, each shape (2,).
    L
        Edge length of the first segment.
    eps
        Numerical tolerance.
    include_endpoints
        Whether mere touching counts as crossing.

    Returns
    -------
    bool
        True if the first edge intersects the swept quadrilateral.
    """
    first_a = np.array([0.0, 0.0], dtype=float)
    first_b = np.array([float(L), 0.0], dtype=float)
    for s in np.linspace(0.0, 1.0, n_samples):
        tip_s = (1.0 - s) * tip_prev + s * tip_new
        angle_s = (1.0 - s) * angle_prev + s * angle_new
        before_s = _get_before_tip(tip_s, angle_s, L)

        if _segments_intersect(first_a, first_b, before_s, tip_s, eps=eps, include_endpoints=include_endpoints):
            return True
    return False


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


# -----------------------------------
# Geometrical helpers inside helpers
# -----------------------------------
def _get_before_tip(tip_pos: NDArray, tip_angle: float, L: float, *, dtype=None):
    """
    Return coordinates of the node that is one before the tip given the position of the last node and tip angle

    Notes:
    ------
    Works with numpy (xp=np) or jax.numpy (xp=jnp).

    Parameters:
    -----------
    tip_pos   - np.array(float), (2,), position of last node
    tip_angle - float
    L         - float, edge/facet length [mm]

    Returns:
    --------
    np.array(float), (2,), position of before the last node
    """
    if dtype is None:
        tip_pos = array(tip_pos).reshape((2,))
    else:
        tip_pos = array(tip_pos, dtype=dtype).reshape((2,))

    dx = L * np.cos(tip_angle)
    dy = L * np.sin(tip_angle)

    if dtype is None:
        return tip_pos - array([dx, dy])
    return tip_pos - array([dx, dy], dtype=dtype)


def _segments_intersect(a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64],
                        d: NDArray[np.float64], *, eps: float = 1e-12, include_endpoints: bool = True) -> bool:
    """
    Return whether the closed segments [a,b] and [c,d] intersect.

    Parameters
    ----------
    a, b, c, d
        Segment endpoints, each shape (2,).
    eps
        Numerical tolerance.
    include_endpoints
        Whether touching at endpoints / collinear touching counts as intersection.

    Returns
    -------
    bool
        True if the segments intersect.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    # Proper crossing
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    if not include_endpoints:
        return False

    # Touching / collinear cases
    if abs(o1) <= eps and _on_segment(a, c, b, eps=eps):
        return True
    if abs(o2) <= eps and _on_segment(a, d, b, eps=eps):
        return True
    if abs(o3) <= eps and _on_segment(c, a, d, eps=eps):
        return True
    if abs(o4) <= eps and _on_segment(c, b, d, eps=eps):
        return True

    return False


def _orient(p: NDArray[np.float64], q: NDArray[np.float64], r: NDArray[np.float64]) -> float:
    """
    Signed 2D orientation / cross product of pq with pr.

    Returns
    -------
    float
        > 0 for counter-clockwise turn,
        < 0 for clockwise turn,
        = 0 for collinear points.
    """
    return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def _on_segment(p: NDArray[np.float64], q: NDArray[np.float64], r: NDArray[np.float64],
                *, eps: float = 1e-12) -> bool:
    """
    Return whether q lies on the closed segment [p, r].
    """
    return (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and
            min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps)
