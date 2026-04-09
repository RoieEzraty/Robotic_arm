from __future__ import annotations
import copy
import csv
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional
from datetime import datetime   

import plot_func, file_helpers, helpers

if TYPE_CHECKING:
    from Logger import Logger
    from ForsentekClass import ForsentekClass
    from SupervisorClass import SupervisorClass
    from MecaClass import MecaClass

"""Experiment protocols for the origami-arm setup.

Methods
-------
sweep_measurement_fixed_origami(m, Snsr, Sprvsr, x_range=None, y_range=None,
                                theta_range=None, N=None, path=None)
    Measure forces over sequence of tip poses, either loaded from file or generated programmatically.
stress_strain(m, Snsr, theta_max, theta_ss, N, y_step, x_step,
              connect_hinge=True, reverse=True)
    Run a two-branch angular sweep for single-hinge stress-strain measurements.
calibrate_forces_all_axes(m, Snsr, weights_gr)
    Collect tri-axial voltage-force calibration data using known weights. fit linear force-voltage model.
calibrate_forces_1axis(Snsr, weights_gr, axis)
    Measure one sensor axis under gravity and sequence of known loads. Used inside calibrate_forces_all_axes()
"""


def sweep_measurement_fixed_origami(m: "MecaClass", Snsr: "ForsentekClass", Sprvsr: "SupervisorClass",
                                    x_range: Optional[float] = None, y_range: Optional[float] = None,
                                    theta_range: Optional[float] = None, N: Optional[int] = None,
                                    path: Optional[str] = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Measure force over a set of tip poses.

    Parameters
    ----------
    x_range, y_range, theta_range : floats, optional. Sweep parameters, used when ``path`` not provided.
    N    : int, optional. Number of sweep points when generating trajectory programmatically.
    path : str, optional. Dataset path. When provided, positions are loaded from file.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]] 
        ``x_y_theta_vec`` of shape ``(N, 3)`` and measured global force array ``F_vec`` of shape ``(N, 2)``.
    """
    m.set_frames(mod="training")

    # ------ load positions from file if path provided and generate tip positions ------
    x_y_theta_vec = _build_sweep_positions(x_range=x_range, y_range=y_range, theta_range=theta_range,
                                           N=N, path=path)
    n_points = int(x_y_theta_vec.shape[0])

    # ------ sweep and measure forces ------
    # empty force vector along sweep trajectory
    F_vec = np.zeros((n_points, 2), dtype=float)
    for i, pos in enumerate(x_y_theta_vec):
        # move arm
        print(f"moving robot to pos={pos}")
        m.move_pos_w_mid(pos, Sprvsr)

        # record force
        print("recording force")
        force_in_t, _ = Snsr.measure()
        Sprvsr.global_force(Snsr, m)  # rotate to global frame
        F_vec[i, :] = np.array([Sprvsr.Fx, Sprvsr.Fy], dtype=float)  # insert in vector

    print("finished logging force measurements")
    return x_y_theta_vec, F_vec


def stress_strain(m: "MecaClass", Snsr: "ForsentekClass", theta_max: float, theta_ss: float, N: int,
                  y_step: float, x_step: float, reverse: bool = True) -> tuple[NDArray[np.float64], 
                                                                               NDArray[np.float64],
                                                                               NDArray[np.float64]]:
    """Run two-sided-sweep stress-strain measurement for a single hinge.

    Protocol
    --------
    1. translate behind arm
    2. Sweep from ``-theta_ss`` to ``+theta_max`` and back to ``+theta_ss``.
    3. Translate behind arm in the other direction.
    4 Sweep from ``+theta_ss`` to ``-theta_max`` and back to ``-theta_ss``.

    Parameters
    ----------
    theta_max : float. Maximal sweep angle [deg].
    theta_ss  : float. Approximate steady state angle, from which to start measurement [deg].
    N         : int. Number of angle samples per sweep. Two sweeps are incorporated here.
    y_step    : float. Lateral bypass translation [mm].
    x_step    : float. Backward bypass translation [mm].
    reverse   : bool. True = flip the sign of the whole protocol.

    Returns
    -------
    Concatenated commanded angles and measured local-frame forces: ``thetas_all`` [deg], ``Fx_all``, ``Fy_all`` [N]

    Notes
    -----
    - The pole should be connected to the robot, not the clamp.
    - Hinge is at center, robot performs circular motion with constant radius from hinge.
    """
    # set frames for stress-strain experiment
    m.set_frames(mod="stress_strain")

    # acquire initial robot position.
    init_pos_6 = np.asarray(m.robot.GetPose(), dtype=float)
    x0, y0, z0, rx0, ry0, _rz0 = init_pos_6

    # all experiment done in reverse
    if reverse:
        y_step = -y_step
        x_step = -x_step
        theta_max = -theta_max
        theta_ss = -theta_ss

    def run_sweep_at_xy(theta_end: float, theta_ss_local: float) -> tuple[NDArray[np.float64], NDArray[np.float64],
                                                                          NDArray[np.float64]]:
        """Run sweep from -theta_ss_local -> theta_end -> +theta_ss_local

        Protocol
        --------
        1. Move robot incrementally
        2. Measure forces. Relevant force is in x axis.

        Parameters
        ----------
        theta_end      : float. Maximal sweep angle [deg].
        theta_ss_local : float. Approximate steady state angle, from which to start measurement [deg].
        x              : float. Number of angle samples per sweep.
        y              : float. Lateral bypass translation [mm].

        Returns
        -------
        Concatenated commanded angles and measured local-frame forces: ``thetas`` [deg], ``Fx``, ``Fy_all`` [N]
        """
        # insert thetas and initiate force vectors
        thetas = _theta_sweep(theta_end=theta_end, theta_ss=theta_ss_local, N=N)
        Fx = np.zeros(thetas.size, dtype=float)
        Fy = np.zeros(thetas.size, dtype=float)

        # move, measure, insert into vectors
        for i, th in enumerate(thetas):
            # move
            target6 = np.array([x0, y0, z0, rx0, ry0, th], dtype=float)
            m.move_pos_w_mid(target6, Sprvsr=None)

            # measure
            force_in_t, _ = Snsr.measure()

            # insert
            Fx[i] = float(np.mean(force_in_t[:, 0]))
            Fy[i] = float(np.mean(force_in_t[:, 1]))

        return thetas, Fx, Fy
 
    # translate behind arm 1st time
    _bypass_arm(m, x_mid=x0 - x_step, y_mid=y0 - y_step, z=z0, rx=rx0, ry=ry0, x_return=x0, y_return=y0)

    # sweep -theta_ss_local -> theta_end -> +theta_ss_local
    th1, Fx1, Fy1 = run_sweep_at_xy(theta_end=+theta_max, theta_ss_local=+theta_ss)

    # translate behind arm 2nd time
    m.move_pos_w_mid(np.array([x0, y0, z0, rx0, ry0, 0.0], dtype=float), Sprvsr=None)
    _bypass_arm(m, x_mid=x0 - x_step, y_mid=y0 + y_step, z=z0, rx=rx0, ry=ry0, x_return=x0, y_return=y0)

    # sweep theta_ss_local -> -theta_end -> -theta_ss_local
    th2, Fx2, Fy2 = run_sweep_at_xy(theta_end=-theta_max, theta_ss_local=-theta_ss)

    # return to rest position
    m.robot.MoveLin(x0, y0, z0, rx0, ry0, 0.0)
    m.robot.WaitIdle()

    # convert angles due to pole-chain contact being off-center
    delta_theta = np.rad2deg(np.arctan(m.pole_rad / (-m.x_TRF)))
    th1 = th1 + delta_theta
    th2 = th2 - delta_theta

    # assemble into full vectors
    thetas_all = np.concatenate([th1, th2])
    Fx_all = np.concatenate([Fx1, Fx2])
    Fy_all = np.concatenate([Fy1, Fy2])
    return thetas_all, Fx_all, Fy_all


def calibrate_forces_all_axes(m: "MecaClass", Snsr: "ForsentekClass",
                              weights_gr: list[float],) -> tuple[NDArray[np.float64], NDArray[np.float64],
                                                                 NDArray[np.float64], NDArray[np.float64]]:
    """Calibrate all three sensor axes using known weights. The user puts them on as asked by code.

    Parameters
    ----------
    weights_gr : list[float]. Calibration masses in grams.

    Returns
    -------
    voltages_array   : Voltage means [V] size (N weights + 1, 3) 
    forces_array     : applied forces [N] size (N weights + 1, 3) 
    stds_array       : voltage standard deviations [V] size (N weights + 1, 3)
    fit_params_array : fitted linear parameters [a,b] for F=a*V+b size (2,3)
    """
    # initialize
    n_weights = len(weights_gr) + 1  # include gravity-only baseline
    voltages_array = np.zeros((n_weights, 3), dtype=float)
    forces_array = np.zeros((n_weights, 3), dtype=float)
    stds_array = np.zeros((n_weights, 3), dtype=float)

    axes = ["x", "y", "z"]
    # positions of tip to put weights on
    joints = np.array([[0.0, 60.0, 30.0, -90.0, -90.0, 90.0],
                       [0.0, 60.0, 30.0, -90.0, -90.0, 180.0],
                       [0.0, 60.0, -60.0, 0.0, -90.0, 180.0],], dtype=float)

    # perform 3 axis calibration
    for i, axis in enumerate(axes):
        m.move_joints(joints[i, :])  # move to joint position, put weight and measure
        voltages_vec, forces_vec, stds_vec = calibrate_forces_1axis(Snsr, weights_gr, axis)
        # insert into 3d array
        voltages_array[:, i] = voltages_vec
        forces_array[:, i] = forces_vec
        stds_array[:, i] = stds_vec

    print("voltages_array", voltages_array)
    print("forces_array", forces_array)
    print("stds_array", stds_array)

    # save to csv file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"calibration_triaxial_{timestamp}.csv"
    file_helpers.save_calibration_csv(filename, voltages_array, forces_array, stds_array)

    # fit parameters Fit F = a*V + b separately for x/y/z, NDArray(2,3)
    fit_params_array = helpers.fit_force_vs_voltage(voltages_array, forces_array, stds_array)
    return voltages_array, forces_array, stds_array, fit_params_array


def calibrate_forces_1axis(Snsr: "ForsentekClass", weights_gr: list[float],
                           axis: str) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calibrate one sensor axis using known masses.

    Parameters
    ----------
    weights_gr : list[float]. Calibration masses in grams.
    axis       : {"x", "y", "z"}. Sensor axis to calibrate.

    Returns
    -------
    voltages_vec : Mean voltages [V] size (N_weight + 1, )
    forces_vec   : applied forces [N] size (N_weight + 1, )
    stds_vec     : voltage standard deviations [V] size (N_weight + 1, )
    """
    # initialize sizes
    dim2 = _axis_to_index(axis)  # convert string to int
    weights_kg = np.asarray(weights_gr, dtype=float) * 1e-3  # convert to kg
    voltages = np.zeros(weights_kg.size, dtype=float)
    stds = np.zeros(weights_kg.size, dtype=float)
    g = 9.81  # gravity in m/s^2

    # interact with user - no weight yet
    input("Place sensor so only g acts and press Enter...")
    # measure under g
    voltage_data, _ = Snsr.measure(1.0, mode="V")
    gravity_trace = voltage_data[:, dim2]
    voltage_g = float(np.mean(gravity_trace))
    stds_g = float(np.std(gravity_trace))
    print(f"voltage_g = {voltage_g:.6f} V")

    # go over all weights
    for i, weight in enumerate(weights_kg):
        # interact with user
        input(f"place weight = {weight:.6f} and press Enter")
        # measure force under weight
        voltage_data, _ = Snsr.measure(1.0, mode="V")
        axis_trace = voltage_data[:, dim2]
        voltages[i] = float(np.mean(axis_trace))
        stds[i] = float(np.std(axis_trace))

    # append to voltages under gravity
    voltages_vec = np.append(voltage_g, voltages)
    stds_vec = np.append(stds_g, stds)
    forces_vec = g * np.append(0.0, weights_kg)

    # print and plot
    print("v", voltages_vec)
    print("f", forces_vec)
    print("std", stds_vec)
    plt.errorbar(voltages_vec, forces_vec, yerr=stds_vec, fmt=".", capsize=3, label="data (±1σ)")
    plt.xlabel(r"$V\,[\mathrm{V}]$")
    plt.ylabel(r"$F\,[\mathrm{N}]$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return voltages_vec, forces_vec, stds_vec


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _build_sweep_positions(x_range: Optional[float], y_range: Optional[float], theta_range: Optional[float],
                           N: Optional[int], path: Optional[str]) -> NDArray[np.float64]:
    """
    Build measurement positions either from a dataset file or from simple sweep parameters.
    Used inside experiemtns.sweep_measurement_fixed_origami

    Parameters
    ----------
    x_range     : float | None. Fixed x-position for the sweep [mm]. Required if ``path``==None.
    y_range     : float | None. Half-range for y sweep [mm]. Generates values in [-y_range, +y_range].
    theta_range : float | None. Fixed orientation angle [deg].
    N           : int | None. Number of points in sweep. Must be positive.
    path        : str | None. Path to dataset file. If provided, overrides sweep generation.

    Returns
    -------
    Array of shape (N, 3) with columns [x, y, theta_deg].

    Notes
    -----
    If ``path`` is provided, positions are loaded using ``file_helpers.load_pos_force(path)``.
    If ``path`` is None, positions are generated as:
           - x is fixed at ``x_range``
           - y is swept linearly over [-y_range, +y_range]
           - theta is fixed at ``theta_range``
    """
    # ------ if position from file ------
    if path is not None:
        rows = file_helpers.load_pos_force(path)
        if not rows:
            raise ValueError("Dataset file is empty.")

        if "pos" in rows[0]:
            return np.array([r["pos"] for r in rows], dtype=float)
        if "pos_meas" in rows[0]:
            return np.array([r["pos_meas"] for r in rows], dtype=float)
        raise KeyError("Neither 'pos' nor 'pos_meas' found in dataset")

    if None in (x_range, y_range, theta_range, N):
        raise ValueError("If path is None, must provide x_range, y_range, theta_range, and N.")

    # ------ if position from range function inputs ------
    n_points = int(N)
    if n_points <= 0:
        raise ValueError("N must be positive.")

    if n_points == 1:
        x_vec = np.array([float(x_range)], dtype=float)
        y_vec = np.array([float(y_range)], dtype=float)
        theta_vec = np.array([float(theta_range)], dtype=float)
    else:
        x_vec = np.full(n_points, float(x_range), dtype=float)
        y_vec = np.linspace(-float(y_range), float(y_range), n_points, dtype=float)
        theta_vec = np.full(n_points, float(theta_range), dtype=float)
    return np.stack([x_vec, y_vec, theta_vec], axis=1)


def _theta_sweep(theta_end: float, theta_ss: float, N: int) -> NDArray[np.float64]:
    """Build angle vector ``-theta_ss -> theta_end -> +theta_ss``

    Parameters
    ----------
    theta_end : float. Highest angle, hinge almost reaching contact [deg]
    theta_ss  : float. Approximated rest (steady-state) angle [deg]
    N         : int. Number of angles to measure through. 

    Notes
    -----
    - No duplicated endpoints.
    - The N points are not evenly spaces in angle space. The way from theta_end to theta_ss has smaller angle step.
    """

    if N < 2:
        return np.array([0.0], dtype=float)

    n_up = N // 2 + 1
    n_down = N - n_up + 1

    up = np.linspace(-theta_ss, theta_end, n_up, dtype=float)
    down = np.linspace(theta_end, theta_ss, n_down, dtype=float)[1:]
    return np.concatenate([up, down])


def _bypass_arm(m: "MecaClass", x_mid: float, y_mid: float, z: float, rx: float,
                ry: float, x_return: float, y_return: float) -> None:
    """Perform short linear bypass move used between angular sweeps.

    Parameters
    ----------
    x_mid, y_mid       : floats. Far x, y positions [mm] so the pole is detached from hinge and goes opposite way
    z, rx, ry          : floats. z position [mm] and x and y angles [deg]. Constant all the way.
    x_return, y_return : float. x and y positions [mm] to return to at other side of hinge facet

    Notes
    -----
    - No force measurement. This step is done just to switch side of pole on hinge facet
    """
    m.robot.MoveLin(x_mid, y_mid, z, rx, ry, 0.0)
    m.robot.WaitIdle()
    m.robot.MoveLin(x_return, y_return, z, rx, ry, 0.0)
    m.robot.WaitIdle()


def _axis_to_index(axis: str) -> int:
    """Map axis name to sensor-column index.

    Parameters
    ----------
    axis : {"x", "y", "z"}. Sensor axis to calibrate.

    Returns
    -------
    int, axis number
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError("axis has to be 'x', 'y' or 'z'")
    return axis_map[axis]


# -------------------------------------------
# NOT IN USE
# -------------------------------------------
# def _connect_hinge_protocol(m: "MecaClass") -> None:
#     """Move above the hinge, wait for manual mounting, then lower to connect."""
#     m.robot.MoveLin(129.5, -12.5, 41.0, -180.0, 0.0, 0.0)
#     m.robot.WaitIdle()

#     input("place hinge below tip and press Enter to connect")

#     m.robot.MoveLin(129.5, -12.5, 30.0, -180.0, 0.0, 0.0)
#     m.robot.WaitIdle()
