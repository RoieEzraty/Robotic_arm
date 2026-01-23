from __future__ import annotations
import configparser
import copy
import csv
import pathlib
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional
import numpy as np

import file_helpers, plot_func, helpers

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
        if self.experiment == "training":
            self.T = self.cfg.getint("training", "T")
            self.rand_key_dataset = self.cfg.getint("training", "rand_key_dataset")
            self.alpha = self.cfg.getfloat("training", "alpha")
        elif self.experiment == "sweep":
            self.T = self.cfg.getint("sweep", "T")
            self.x_range = self.cfg.getint("sweep", "x_range")
            self.y_range = self.cfg.getint("sweep", "y_range")
            self.theta_range = self.cfg.getint("sweep", "theta_range")

        # initiate arrays in training time 
        self.F_in_t = np.zeros([self.T, 2])
        self.pos_update_in_t = np.zeros([self.T, 3])  # 2nd dim is [x, y, theta_z]
        self.total_angle_update_in_t = np.zeros([self.T])
        self.loss_in_t = np.zeros([self.T, 2])
        self.loss_norm_in_t = np.zeros([self.T, 2])
        self.loss_MSE_in_t = np.zeros([self.T,])  

        self.origin_rel_to_sim = np.array([108, -14, 0])      

    def init_dataset(self, dataset_path: str = "dataset.csv") -> None:
        rng = np.random.default_rng(self.rand_key_dataset)
        pos_force_rows = file_helpers.load_pos_force(dataset_path)

        # # Convert to arrays (ML-friendly)
        # pos = np.array([r["pos"] for r in pos_force_rows], dtype=float)      # (N, 3)
        # force = np.array([r["force"] for r in pos_force_rows], dtype=float)  # (N, 2)

        # # Shuffle consistently
        # idx = rng.permutation(len(pos))
        # self.pos_in_t = pos[idx]
        # self.desired_F_in_t = force[idx]
        pos_base = np.array([r["pos"] for r in pos_force_rows], dtype=float)      # (N, 3)
        force_base = np.array([r["force"] for r in pos_force_rows], dtype=float)  # (N, 2)

        N = len(pos_base)
        if N == 0:
            raise ValueError("Dataset is empty")

        # ---- allocate training-time arrays ----
        self.pos_in_t = np.zeros((self.T, 3), dtype=float)
        self.desired_F_in_t = np.zeros((self.T, 2), dtype=float)

        # ---- fill by cycling with reshuffle ----
        t = 0
        while t < self.T:
            idx = rng.permutation(N)  # new shuffle each cycle

            k = min(N, self.T - t)    # how many we still need
            self.pos_in_t[t:t+k] = pos_base[idx[:k]]
            self.desired_F_in_t[t:t+k] = force_base[idx[:k]]

            t += k

    def draw_measurement(self, t) -> None:
        self.pos = self.pos_in_t[t, :]

    def global_force(self, Snsr: ForsentekClass, m: MecaClass, t: Optional[int] = None,
                     plot: bool = False) -> NDArray[np.float_]:
        force_in_t = Snsr.force_data
        measure_t = Snsr.t
        theta_z_deg = m.robot.GetPose()[-1]
        theta = np.deg2rad(theta_z_deg - 90.0)

        Fx_global_in_t = force_in_t[:, 0]*np.cos(theta) + force_in_t[:, 1]*np.sin(theta)
        Fy_global_in_t = -force_in_t[:, 0]*np.sin(theta) + force_in_t[:, 1]*np.cos(theta)

        if plot:
            plot_func.force_global_during_measurement(measure_t, Fx_global_in_t, Fy_global_in_t)

        self.Fx = np.mean(Fx_global_in_t)
        self.Fy = np.mean(Fy_global_in_t)
        if t is not None:
            self.F_in_t[t, :] = np.array([self.Fx, self.Fy])

    def calc_loss(self, t: int, norm_force: float) -> None:
        self.loss = self.F_in_t[t, :] - self.desired_F_in_t[t, :]  # [N]
        self.loss_norm = self.loss / norm_force  # dimless
        self.loss_MSE = np.mean(self.loss_norm ** 2)  # scalar

        self.loss_in_t[t, :] = self.loss
        self.loss_norm_in_t[t, :] = self.loss_norm
        self.loss_MSE_in_t[t] = self.loss_MSE

    def calc_tip_update(self, m: MecaClass, t: int, correct_for_total_angle: bool = True) -> None:
        tip_pos = self.pos_update_in_t[t, :2]
        print('current update tip_pos = ', tip_pos)
        if t == 0:
            prev_pos_update = m.current_pos
        else:
            prev_pos_update = self.pos_update_in_t[t-1, :]
        print('previous tip update value = ', prev_pos_update)        

        sgn_x = np.sign(prev_pos_update[0]) 
        sgn_y = np.sign(prev_pos_update[1]) 
        sgn_theta = -1  # Meca robot measures angles CW when head is inverted
        delta_x_update = self.alpha * self.loss_norm[0] * sgn_x * m.norm_length
        delta_y_update = - self.alpha * self.loss_norm[0] * sgn_x * m.norm_length
        delta_theta_update = sgn_theta * (- self.alpha * self.loss_norm[1] * m.norm_angle)

        self.pos_update_in_t[t, :] = prev_pos_update + np.array([delta_x_update, delta_y_update,
                                                                 delta_theta_update])
        print('pos_update_in_t[t, :] before correct for tot angle = ', self.pos_update_in_t[t, :])

        if correct_for_total_angle:
            if t == 1:
                prev_total_angle = 0.0
            else:
                prev_total_angle = self.total_angle_update_in_t[t-1]
            print('prev_total_angle', prev_total_angle)
            self.total_angle = helpers.get_total_angle(m.pos_origin, tip_pos, prev_total_angle)
            delta_total_angle = self.total_angle - prev_total_angle
            print('current total_angle', self.total_angle)
            print('delta_total_angle', delta_total_angle)
            # add delta total angle
            self.pos_update_in_t[t, 2] += delta_total_angle
            # save as variable in t
            self.total_angle_update_in_t[t] = self.total_angle
        
        print('pos_update_in_t[t, :] after correct for tot angle = ', self.pos_update_in_t[t, :])
