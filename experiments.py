from __future__ import annotations
import configparser
import copy
import csv
import pathlib
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING, Callable, Union, Optional
import numpy as np
import csv
from datetime import datetime	

import plot_func, file_helpers, helpers

if TYPE_CHECKING:
	from Logger import Logger
	from ForsentekClass import ForsentekClass
	from SupervisorClass import SupervisorClass
	from MecaClass import MecaClass


def sweep_measurement_fixed_origami(
        m: MecaClass,
        Snsr: ForsentekClass,
        Sprvsr: SupervisorClass,
        x_range: Optional[float] = None,
        y_range: Optional[float] = None,
        theta_range: Optional[float] = None,
        N: Optional[int] = None,
        path: Optional[str] = None):

    m.set_frames(mod="training")

    # -----------------------------------------
    # Load positions from file if path provided
    # -----------------------------------------
    if path is not None:
        rows = file_helpers.load_pos_force(path)  # uses your CSV format
        x_y_theta_vec = np.array([r["pos"] for r in rows], dtype=float)  # [mm, mm, deg]
        N = x_y_theta_vec.shape[0]

    # -----------------------------------------
    # Generate sweep programmatically if path not provided
    # -----------------------------------------
    else:
        if None in (x_range, y_range, theta_range, N):
            raise ValueError("If path is None, must provide x_range, y_range, theta_range, N")

        if N == 1:
            x_vec = np.array([x_range])
            y_vec = np.array([y_range])
            theta_vec = np.array([theta_range])
        else:
            x_vec = np.ones(N) * x_range
            y_vec = np.linspace(-y_range, y_range, N)
            theta_vec = np.ones(N) * theta_range

        x_y_theta_vec = np.stack([x_vec, y_vec, theta_vec], 1)  # [mm, mm, deg]

    # -----------------------------------------
    # Run measurement
    # -----------------------------------------
    F_vec = np.zeros([2, N])

    for i in range(N):
        pos = x_y_theta_vec[i, :]

        print(f"moving robot to pos={pos}")
        m.move_pos_w_mid(pos, Sprvsr)

        print("recording force")
        Snsr.measure(0.5)  # force measured in [N]
        Sprvsr.global_force(Snsr, m)

        F_vec[:, i] = np.array([Sprvsr.Fx, Sprvsr.Fy])  # [N]

    print("finished logging force measurements")

    # convert from [N] to [mN]
    F_vec_mN = F_vec * Sprvsr.convert_F

    return x_y_theta_vec, F_vec_mN


def stress_strain(m: "MecaClass", Snsr: "ForsentekClass", theta_max: float, theta_ss: float, N: int, 
	              y_step: float, x_step: float, connect_hinge: bool = True, move_mod: str = "lin",
	              sweep_mod: str = "pole"):
    """
    Protocol:
      Sweep 1:  theta 0 -> +theta_max -> 0
      Translate: MoveLin along y by y_step (theta held at 0)
      Sweep 2:  theta 0 -> -theta_max -> 0
    
    inputs:
    connect_hinge - bool, True: start from above, let user connect hinge and only then lower tip
    y_step        - float [mm], linear motion along +y after first sweep
	move_mod      - str, "lin" or "pose"

    Returns:
      thetas_all: (M,) angles commanded (deg)
      Fx_all, Fy_all: (M,) measured mean forces for each angle
    """
    m.set_frames(mod="stress_strain")

    if connect_hinge:
        # move above hinge
        m.robot.MoveLin(129.5, -12.5, 41, -180, 0, 0)
        m.robot.WaitIdle()

        input("place hinge below tip and press Enter to connect")

        # lower to connect
        m.robot.MoveLin(129.5, -12.5, 30, -180, 0, 0)
        m.robot.WaitIdle()

    init_pos_6 = tuple(m.robot.GetPose())  # (x,y,z,rx,ry,rz)
    x0, y0, z0, rx0, ry0, _rz0 = init_pos_6

    def theta_sweep(theta_end: float, theta_ss: float, N: int) -> np.ndarray:
        """
        Build: -theta_ss -> theta_end -> +theta_ss, inclusive, without duplicating endpoints.
        N is the total number of points in the sweep.
        """
        if N < 2:
            return np.array([0.0], dtype=float) 

        n_up = N // 2 + 1
        n_down = N - n_up + 1

        up = np.linspace(-theta_ss, theta_end, n_up)                 # includes 0 and theta_end
        down = np.linspace(theta_end, theta_ss, n_down)[1:]         # drop theta_end duplicate
        return np.concatenate([up, down])

    def run_sweep_at_xy(theta_end: float, theta_ss: float, x: float, y: float):
        thetas = theta_sweep(theta_end, theta_ss, N)
        Fx = np.zeros((thetas.size,), dtype=float)
        Fy = np.zeros((thetas.size,), dtype=float)

        for i, th in enumerate(thetas):
            target6 = np.array([x, y, z0, rx0, ry0, th], dtype=float)
            m.move_pos_w_mid(target6, Sprvsr=None, mod=move_mod)

            Snsr.measure()
            Fx[i] = float(np.mean(Snsr.force_data[:, 0]))
            Fy[i] = float(np.mean(Snsr.force_data[:, 1]))

        return thetas, Fx, Fy

    if sweep_mod == "pole":
	    # ===== Linear motion to go behind arm
	    y1 = float(y0 - y_step)
	    x1 = float(x0 - x_step)
	    m.robot.MoveLin(x1, y1, z0, rx0, ry0, 0.0)
	    m.robot.WaitIdle()
	    m.robot.MoveLin(x0, y0, z0, rx0, ry0, 0.0)
	    m.robot.WaitIdle()

    # ===== Sweep 1: 0 -> +theta_max -> 0 at (x0,y0)
    th1, Fx1, Fy1 = run_sweep_at_xy(+theta_max, +theta_ss, x0, y0)

    # Ensure theta=0 before the y translation
    m.move_pos_w_mid(np.array([x0, y0, z0, rx0, ry0, 0.0], dtype=float), Sprvsr=None, mod=move_mod)

    if sweep_mod == "pole":
	    # ===== Linear motion to go behind arm, 2nd time
	    y1 = float(y0 + y_step)
	    x1 = float(x0 - x_step)
	    m.robot.MoveLin(x1, y1, z0, rx0, ry0, 0.0)
	    m.robot.WaitIdle()
	    m.robot.MoveLin(x0, y0, z0, rx0, ry0, 0.0)
	    m.robot.WaitIdle()

    # ===== Sweep 2: 0 -> -theta_max -> 0 at (x0,y1)
    th2, Fx2, Fy2 = run_sweep_at_xy(-theta_max, -theta_ss, x0, y0)

    # de-stress at end
    m.robot.MoveLin(x0, y0, z0, rx0, ry0, 0.0)
    m.robot.WaitIdle()

    # concatenate (optionally keep a separator if you want)
    if sweep_mod == "pole":  # offset thetas due to pole
    	# -x_TRF is half a link, pole_rad is pole radius, constant angle shift is the tangent
    	delta_theta = np.rad2deg(np.arctan(m.pole_rad / (-m.x_TRF)))
    	th1 = th1 + delta_theta
    	th2 = th2 - delta_theta
    thetas_all = np.concatenate([th1, th2])
    Fx_all = np.concatenate([Fx1, Fx2])
    Fy_all = np.concatenate([Fy1, Fy2])

    return thetas_all, Fx_all, Fy_all


def calibrate_forces_all_axes(m: MecaClass, Snsr: ForsentekClass, weights_gr: list) -> None:
	voltages_array = np.zeros([np.size(weights_gr)+1, 3])
	forces_array = np.zeros([np.size(weights_gr)+1, 3])
	stds_array = np.zeros([np.size(weights_gr)+1, 3])
	axes = ['x', 'y', 'z']
	joints = np.array([[0, 60, 30, -90, -90, 90],  # x axis force sensor at ai1
		               [0, 60, 30, -90, -90, 180],   # y axis force sensor at ai2
		               [0, 60, -60, 0, -90, 180]])  # z axis force sensor at ai3

	# positions = np.array([[220.0, 0.0, 100, -90, 0, 180],  # x axis force sensor at ai1
	# 	                  [220.0, 0.0, 100, -90, 0, 90],   # y axis force sensor at ai2
	# 	                  [220.0, 0.0, 360, 0, 0, 90]])  # z axis force sensor at ai3
	for i in range(3):
		m.move_joints(joints[i, :])
		voltages_vec, forces_vec, stds_vec = calibrate_forces_1axis(Snsr, weights_gr, axes[i])
		voltages_array[:, i] = voltages_vec
		forces_array[:, i] = forces_vec
		stds_array[:, i] = stds_vec

	print('voltages_array', voltages_array)
	print('forces_array', forces_array)
	print('stds_array', stds_array)

	# ---- save ONE combined tri-axial file (9 columns) ----
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	filename = f"calibration_triaxial_{timestamp}.csv"
	file_helpers.save_calibration_csv(filename, voltages_array, forces_array, stds_array)

	fit_params_array = helpers.fit_force_vs_voltage(voltages_array, forces_array, stds_array)

	return voltages_array, forces_array, stds_array, fit_params_array


def calibrate_forces_1axis(Snsr: ForsentekClass, weights_gr: list, axis: str):
	"""
	weights in grams
	"""
	if axis == 'x':
		dim2 = 0
	elif axis == 'y':
		dim2 = 1
	elif axis == 'z':
		dim2 = 2
	else:
		print('axis has to be x, y or z string')
	weights_kg = np.asarray(weights_gr)*10**(-3)  # weights in kg
	voltages = np.zeros(np.size(weights_kg))
	stds = np.zeros(np.size(weights_kg))
	g = 9.81

	# --- gravity load ---
	input("Place sensor so only g acts and press Enter...")
	Snsr.measure(1, mode='V')
	data = Snsr.voltage_data[:, dim2]
	voltage_g = np.mean(data)  # voltage of just the gravity load
	stds_g = np.std(data)
	print(f"voltage_g = {voltage_g:.6f} V")
	    
	for i, weight in enumerate(weights_kg):
	    input(f"place weight = {weight:.6f} and press Enter")
	    Snsr.measure(1, mode='V')
	    data = Snsr.voltage_data[:, dim2]
	    voltages[i] = np.mean(data)
	    stds[i] = np.std(data)

	voltages_vec = np.append(voltage_g, voltages)
	stds_vec = np.append(stds_g, stds)
	forces_vec = g * np.append(0, weights_kg)
	print('v', voltages_vec)
	print('f', forces_vec)
	print('std', stds_vec)
	
	plt.errorbar(voltages_vec, forces_vec, yerr=stds_vec, fmt=".", capsize=3, label="data (±1σ)")
	# data with y-error bars
	plt.xlabel(r"$V\,[\mathrm{V}]$")
	plt.ylabel(r"$F\,[\mathrm{N}]$")
	plt.legend()
	plt.tight_layout()
	plt.show()

	return voltages_vec, forces_vec, stds_vec
