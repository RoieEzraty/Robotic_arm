from __future__ import annotations
import configparser
import copy
import csv
import pathlib
from numpy.typing import NDArray
import numpy as np
from typing import TYPE_CHECKING, Callable, Union, Optional
import numpy as np

if TYPE_CHECKING:
	from Logger import Logger


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
