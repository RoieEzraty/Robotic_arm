from __future__ import annotations
import configparser
import copy
import csv
import pathlib
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional
import numpy as np

import file_helpers

if TYPE_CHECKING:
    from Logger import Logger
    from ForsentekClass import ForsentekClass


# ===================================================
# Class - Supervisor Variables - Loss, update values, etc.
# ===================================================


class SupervisorClass:
    """
    Class with variables dictated by supervisor

    Attributes
    ----------

    problem        - str, type of measurement used in problem
                 'tau' = torque is optimized
                 'Fy'  = force in y direction is optimized
    """
    problem: str

    def __init__(self, config_path: str = "supervisor_config.ini") -> None:
        """
        Parameters
        ----------
        
        
        """
        # config
        self.cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        self.cfg.read(pathlib.Path(config_path))

        # other sizes
        self.experiment = str(self.cfg.get("experiment", "experiment"))
        if self.experiment == "training":
            self.T = self.cfg.getint("Training", "T")
            self.rand_key_dataset = self.cfg.getint("Training", "rand_key_dataset")
            self.alpha = self.cfg.getint("Training", "alpha")
        elif self.experiment == "sweep":
            self.T = self.cfg.getint("sweep", "T")
            self.x_range = self.cfg.getint("sweep", "x_range")
            self.y_range = self.cfg.getint("sweep", "y_range")
            self.theta_range = self.cfg.getint("sweep", "theta_range")

        # initiate arrays in training time 
        self.input_update_in_t = np.zeros([self.T,])
        self.loss_in_t = np.zeros([self.T,])
        self.loss_norm_in_t = np.zeros([self.T,])
        self.loss_MSE_in_t = np.zeros([self.T,])        

    def init_dataset(self, dataset_path: str = "dataset.csv") -> None:
        rng = np.random.default_rng(self.rand_key_dataset)
        pos_force_rows = file_helpers.load_pos_force(dataset_path)

        # Convert to arrays (ML-friendly)
        pos = np.array([r["pos"] for r in pos_force_rows], dtype=float)      # (N, 3)
        force = np.array([r["force"] for r in pos_force_rows], dtype=float)  # (N, 2)

        # Shuffle consistently
        idx = rng.permutation(len(pos))
        self.pos_in_t = pos[idx]
        self.desired_F_in_t = force[idx]

    def draw_measurement(self, t) -> None:
        self.pos = self.pos_in_t[t, :]

    def global_force(self, Snsr: "ForsentekClass") -> NDArray[np.float_]:
        theta_z = self.pos[2] + Snsr.theta_sensor
        self.F = Snsr.local_F[0] * np.cos(theta_z) + Snsr.local_F[1] * np.sin(theta_z)
        return self.F
