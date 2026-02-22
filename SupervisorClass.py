from __future__ import annotations
import configparser
import copy
import csv
import pathlib
import re
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional
import numpy as np
import pandas as pd

import file_helpers, plot_func, helpers, experiments

if TYPE_CHECKING:
    from Logger import Logger
    from ForsentekClass import ForsentekClass
    from MecaClass import MecaClass


# ===================================================
# Class - Supervisor Variables - Loss, update values, etc.
# ===================================================


class SupervisorClass:
    """
    Class with variables dictated by supervisor

    Attributes
    ----------

    problem - str, type of measurement used in problem
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
        self.dataset_type = str(self.cfg.get("experiment", "dataset_type"))
        if self.dataset_type == "from file":
            self.dataset_path = str(self.cfg.get("experiment", "dataset_path"))

        if self.experiment == "training":
            self.T = self.cfg.getint("training", "T")
            self.rand_key_dataset = self.cfg.getint("training", "rand_key_dataset")
            self.alpha = self.cfg.getfloat("training", "alpha")
            self.init_buckle = self.cfg.get("training", "init_buckle")
            self.desired_buckle = self.cfg.get("training", "desired_buckle")
        elif self.experiment == "predetermined training":
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                self.T = sum(1 for _ in f) - 2  # subtract header and t=0
            # find all bracket contents and convert to numpy arrays of buckles
            buckles = re.findall(r"\[([^\]]+)\]", self.dataset_path)
            self.init_buckle = np.fromstring(buckles[0], sep=' ')
            self.desired_buckle = np.fromstring(buckles[1], sep=' ')
        elif self.experiment == "sweep":
            self.T = self.cfg.getint("sweep", "T")
            self.x_range = self.cfg.getint("sweep", "x_range")
            self.y_range = self.cfg.getint("sweep", "y_range")
            self.theta_range = self.cfg.getint("sweep", "theta_range")  # [deg]

        self.L = self.cfg.getfloat("chain", "L")
        self.H = self.cfg.getint("chain", "H")

        # initiate arrays in training time 
        self.F_in_t = np.zeros([self.T, 2])
        self.pos_update_in_t = np.zeros([self.T, 3])  # 2nd dim is [x, y, theta_z]
        self.total_angle_update_in_t = np.zeros([self.T])  # [deg]
        self.loss_in_t = np.zeros([self.T, 2])
        self.loss_norm_in_t = np.zeros([self.T, 2])
        self.loss_MSE_in_t = np.zeros([self.T,])  

        self.origin_rel_to_sim = np.array([108, -14, 0]) 
        self.convert_F = self.cfg.getfloat("files", "convert_F")   

    def init_dataset(self, dataset_path: Optional[str] = "dataset.csv",
                     out_path="dataset.csv", measure_des: Optional[bool] = False, 
                     m: Optional[MecaClass] = None, Snsr: Optional[ForsentekClass] = None) -> None:

        if measure_des:
            print('measuring desired forces in training configuration solely')
            pos_base, force_base = experiments.sweep_measurement_fixed_origami(m, Snsr, self,
                                                                               path=dataset_path)
            file_helpers.write_supervisor_dataset(pos_base, force_base, out_path)
        else:
            if self.experiment == 'training':  # use random key for dataset
                rng = np.random.default_rng(self.rand_key_dataset)
            pos_force_rows = file_helpers.load_pos_force(dataset_path)

            # # Convert to arrays (ML-friendly)
            # pos = np.array([r["pos"] for r in pos_force_rows], dtype=float)      # (N, 3)
            # force = np.array([r["force"] for r in pos_force_rows], dtype=float)  # (N, 2)

            # # Shuffle consistently
            # idx = rng.permutation(len(pos))
            # self.pos_in_t = pos[idx]
            # self.desired_F_in_t = force[idx]
            if self.experiment == 'training':  # upload positions and desired forces
                pos_base = np.array([r["pos"] for r in pos_force_rows], dtype=float)      # (N, 3)
                force_base = np.array([r["force"] for r in pos_force_rows], dtype=float)  # (N, 2)
            elif self.experiment == 'predetermined training':
                pos_base = np.array([r["pos_meas"] for r in pos_force_rows], dtype=float)      # (N, 3)
                pos_update = np.array([r["pos_update"] for r in pos_force_rows], dtype=float)      # (N, 3)
                force_base = np.array([r["force_meas"] for r in pos_force_rows], dtype=float)  # (N, 2)

        N = pos_base.shape[0]
        if N == 0:
            raise ValueError("Dataset is empty")

        # ---- allocate training-time arrays ----
        self.pos_in_t = np.zeros((self.T, 3), dtype=float)
        self.desired_F_in_t = np.zeros((self.T, 2), dtype=float)

        # ---- fill position and desired values by cycling with reshuffle ----
        if self.experiment == 'training':
            if self.dataset_type == 'from file':
                reps = int(np.ceil(self.T / N))
                pos_tiled = np.tile(pos_base, (reps, 1))
                force_tiled = np.tile(force_base, (reps, 1))
                self.pos_in_t = pos_tiled[:self.T]
                self.desired_F_in_t = force_tiled[:self.T]
            elif self.dataset_type == 'shuffle':
                t = 0
                while t < self.T:
                    idx = rng.permutation(N)  # new shuffle each cycle

                    k = min(N, self.T - t)    # how many we still need
                    self.pos_in_t[t:t+k] = pos_base[idx[:k]]
                    self.desired_F_in_t[t:t+k] = force_base[idx[:k]]

                    t += k
        # ---- fill position and desired values from file ----
        elif self.experiment == 'predetermined training':
            self.pos_in_t = pos_base[1:, :]
            self.desired_F_in_t = force_base[1:, :]
            self.pos_update_in_t = pos_update[1:, :]  # 1st instance is zeros
            # append last instance again
            self.pos_update_in_t = np.vstack([self.pos_update_in_t, self.pos_update_in_t[-1]])

    def draw_measurement(self, t) -> None:
        self.pos = self.pos_in_t[t, :]

    def global_force(self, Snsr: ForsentekClass, m: MecaClass, t: Optional[int] = None,
                     plot: bool = False) -> NDArray[np.float_]:
        force_in_t = Snsr.force_data * self.convert_F  # [mN]
        measure_t = Snsr.t
        m._get_current_pos()
        theta = -(m.current_pos[-1] - Snsr.theta_sensor)  # [deg]
        Fx_global_in_t, Fy_global_in_t = helpers.rotate_force_frame(force_in_t, theta)

        if plot:
            plot_func.force_global_during_measurement(measure_t, Fx_global_in_t, Fy_global_in_t)

        self.Fx = np.mean(Fx_global_in_t)
        self.Fy = np.mean(Fy_global_in_t)
        if t is not None:
            self.F_in_t[t, :] = np.array([self.Fx, self.Fy])

    def calc_loss(self, t: int, norm_force: float) -> None:
        self.loss = (self.desired_F_in_t[t, :] - self.F_in_t[t, :]) / self.convert_F  # [mN]
        self.loss_norm = self.loss / norm_force  # from [mN] to dimless
        self.loss_MSE = np.mean(self.loss_norm ** 2)  # scalar

        self.loss_in_t[t, :] = self.loss
        self.loss_norm_in_t[t, :] = self.loss_norm
        self.loss_MSE_in_t[t] = self.loss_MSE

    def calc_tip_update(self, m: MecaClass, t: int, correct_for_total_angle: bool = True) -> None:
        tip_pos = self.pos_update_in_t[t, :2]
        if t == 0:
            # prev_pos_update = m.current_pos
            prev_pos_update = self.pos_in_t[0]
        else:
            prev_pos_update = self.pos_update_in_t[t-1, :]        

        sgn_x = np.sign(prev_pos_update[0]) 
        sgn_y = np.sign(prev_pos_update[1]) 
        delta_x_update = self.alpha * self.loss_norm[0] * sgn_x * m.norm_length
        delta_y_update = - self.alpha * self.loss_norm[0] * sgn_x * m.norm_length
        delta_theta_update = - self.alpha * self.loss_norm[1] * m.norm_angle  # [deg]

        self.pos_update_in_t[t, :] = prev_pos_update + np.array([delta_x_update, delta_y_update,
                                                                 delta_theta_update])
        print('pos_update_in_t[t, :] before correct for tot angle = ', self.pos_update_in_t[t, :])

        if correct_for_total_angle:
            if t == 1:
                prev_total_angle = 0.0  # [deg]
            else:
                prev_total_angle = self.total_angle_update_in_t[t-1]  # [deg]
            self.total_angle = helpers.get_total_angle(self.L, tip_pos, prev_total_angle)
            delta_total_angle = self.total_angle - prev_total_angle
            print('current total_angle', self.total_angle)
            # add delta total angle
            self.pos_update_in_t[t, 2] += delta_total_angle
            # save as variable in t
            self.total_angle_update_in_t[t] = self.total_angle
        
        print('pos_update_in_t[t, :] after correct for tot angle = ', self.pos_update_in_t[t, :])
