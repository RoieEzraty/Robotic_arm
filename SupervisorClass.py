from __future__ import annotations
import configparser
import copy
import csv
import pathlib
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional
import numpy as np

if TYPE_CHECKING:
    from Logger import Logger


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

    def __init__(self, config_path: str = "robot_config.ini") -> None:
        """
        Parameters
        ----------
        
        
        """
        # config
        self.cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        self.cfg.read(pathlib.Path(config_path))

        # parameters
        self.T = self.cfg.getint("Training", "T")
        self.rand_key_dataset = self.cfg.getint("Training", "rand_key_dataset")
        self.alpha = self.cfg.getint("Training", "alpha")

        self.input_update_in_t = np.zeros([self.T,])
        self.loss_in_t = np.zeros([self.T,])
        self.loss_norm_in_t = np.zeros([self.T,])
        self.loss_MSE_in_t = np.zeros([self.T,])        
        
        self.tip_pos = [0, 0, 0]  # [x, y, theta_z]
        self.tip_pos_in_t[0] = self.tip_pos

    def init_dataset(self, dataset_path: str = "dataset.csv") -> None:
        rng = np.random.default_rng(self.rand_key_dataset)

        self.theta_in_t = rng.uniform(low=-90, high=90, size=(self.T, Strctr.H))  # (T, hinges)
        if self.problem == 'Fy':
            # x, y coords from thetas
            self.pos_in_t = funcs_geometry.forward_points(Strctr.L, self.theta_in_t)

    def load_pos_force(path: str):
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    "t_unix": float(r["t_unix"]),
                    "pos": (float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])),
                    "force": (float(r["force_x"]), float(r["force_y"])),
                })
        return rows
