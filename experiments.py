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


def sweep_measurement_fixed_origami(x_range, y_range, theta_range, N, robot, force_sensor):
	x_vec = np.linspace(-x_range, x_range, N)
	y_vec = np.linspace(-y_range, y_range, N)
	theta_vec = np.linspace(-theta_range, theta_range, N)
	x_y_theta_vec = np.stack([x_vec, y_vec, theta_vec], 1)

	logger = Logger()
	print("Logging to:", logger.path)
	logger.start()

	for i in range(N):
		pos = x_y_theta_vec[i, :]  # [x, y, theta_z]

		# move arm
		robot.move_lin(pos)

		# record force
		F = force_sensor.record(pos[3])   # [Fx, Fy, torque]
		logger.log(pos, F)

	logger.stop()
	print("finished logging force measurements")


def calibrate_forces(Snsr: ForsentekClass, weights_gr: list):
	"""
	weights in grams
	"""
	weights_kg = np.asarray(weights_gr)*10**(-3)  # weights in kg
	voltages = np.zeros(np.size(weights_kg))
	stds = np.zeros(np.size(weights_kg))
	g = 9.81

	# --- zero / flat ---
	input("Place sensor flat (no load) and press Enter...")
	_, data = Snsr.measure(1)
	voltage_flat = np.mean(data)
	stds_flat = np.std(data)
	print(f"voltage_flat = {voltage_flat:.6f} V")

	# --- gravity load ---
	input("Place sensor so only g acts and press Enter...")
	_, data = Snsr.measure(1)
	voltage_g = np.mean(data)  # voltage of just the gravity load
	stds_g = np.std(data)
	print(f"voltage_g = {voltage_g:.6f} V")
	    
	for i, weight in enumerate(weights_kg):
	    input(f"place weight = {weight:.6f} and press Enter")
	    _, data = Snsr.measure(1)
	    voltages[i] = np.mean(data)
	    stds[i] = np.std(data)

	voltages_vec = np.append(voltage_flat, voltages-voltage_g)
	stds_vec = np.append(stds_flat, stds)
	forces_vec = g * np.append(0, weights_kg)

	print('v', voltages_vec)
	print('f', forces_vec)
	print('std', stds_vec)

	fit_params = helpers.fit_force_vs_voltage(voltages_vec, forces_vec, stds_vec)

	plot_func.calibration_forces(voltages_vec, forces_vec, stds_vec, fit_params)

	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	filename = f"calibration_{timestamp}.csv"

	file_helpers.save_calibration_csv(filename, voltages_vec, forces_vec, stds_vec)

	return voltages_vec, forces_vec, stds, fit_params
