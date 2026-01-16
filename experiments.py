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


def sweep_measurement_fixed_origami(m: MecaClass, Snsr: ForsentekClass, Sprvsr: SupervisorClass,
	                                x_range: float, y_range: float, theta_range: float, N: int):
	if N == 1:
		x_vec = np.array([x_range])
		y_vec = np.array([y_range])
		theta_vec = np.array([theta_range])
	else:
		x_vec = np.linspace(-x_range, x_range, N)
		y_vec = np.linspace(-y_range, y_range, N)
		theta_vec = np.linspace(-theta_range, theta_range, N)
	x_y_theta_vec = np.stack([x_vec, y_vec, theta_vec], 1)
	F_vec = np.zeros([2, N])

	for i in range(N):
		pos = x_y_theta_vec[i, :]  # [x, y, theta_z]

		# move arm
		print(f"moving robot to pos={pos}")
		m.move_pos(pos)

		# record force
		print("recording force")
		Snsr.measure(2)   # [Fx, Fy, torque]
		Sprvsr.global_force(Snsr, m, i)
		F_vec[:, i] = np.array([Sprvsr.Fx, Sprvsr.Fy])
	print("finished logging force measurements")

	return x_y_theta_vec, F_vec


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
	_, data_3d = Snsr.measure(1, mode='V')
	data = data_3d[:, dim2]
	voltage_g = np.mean(data)  # voltage of just the gravity load
	stds_g = np.std(data)
	print(f"voltage_g = {voltage_g:.6f} V")
	    
	for i, weight in enumerate(weights_kg):
	    input(f"place weight = {weight:.6f} and press Enter")
	    _, data_3d = Snsr.measure(1, mode='V')
	    data = data_3d[:, dim2]
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
