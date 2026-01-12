import configparser
import pathlib
import nidaqmx
import time
import numpy as np

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Tuple, Optional

import file_helpers, helpers, plot_func

from nidaqmx.constants import TerminalConfiguration, AcquisitionType


class ForsentekClass:
    def __init__(self, config_path: str = "forsentek_config.ini"):
        # config
        self.cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        self.cfg.read(pathlib.Path(config_path))

        # ip priority: explicit arg > config file
        self.DEV = self.cfg.get("ni", "DEV", fallback=None)
        CHAN_x = self.cfg.get("ni", "Channel_x", fallback=None)
        CHAN_y = self.cfg.get("ni", "Channel_y", fallback=None)
        CHAN_z = self.cfg.get("ni", "Channel_z", fallback=None)
        self.CHAN_x = f"{self.DEV}/{CHAN_x}"
        self.CHAN_y = f"{self.DEV}/{CHAN_y}"
        self.CHAN_z = f"{self.DEV}/{CHAN_z}"
        self.fs = self.cfg.getfloat("ni", "samp_freq", fallback=None)
        self.T = self.cfg.getfloat("ni", "T", fallback=None)
        self.min_val = self.cfg.getfloat("amp", "min_val", fallback=None)
        self.max_val = self.cfg.getfloat("amp", "max_val", fallback=None)

        self.theta_sensor = self.cfg.getfloat("rotation", "angle", fallback=None)

        # ------ create task ------
        self.task = nidaqmx.Task()
        # self.task.ai_channels.add_ai_voltage_chan(self.CHAN,
        #                                           terminal_config=TerminalConfiguration.RSE,
        #                                           min_val=self.min_val, max_val=self.max_val)
        for ch in (self.CHAN_x, self.CHAN_y, self.CHAN_z):
            self.task.ai_channels.add_ai_voltage_chan(ch, terminal_config=TerminalConfiguration.RSE,
                                                      min_val=self.min_val, max_val=self.max_val)

        # -------
        self.calibration_path = self.cfg.get("calibration", "calibration_path")
        voltages_vec, forces_vec, stds_vec = file_helpers.load_calibration_csv(self.calibration_path)
        force_fit_params = helpers.fit_force_vs_voltage(voltages_vec, forces_vec, stds_vec)
        plot_func.calibration_forces(voltages_vec, forces_vec, stds_vec, force_fit_params)
        self.F_a, self.F_b = force_fit_params[0], force_fit_params[1]

    def force_from_voltage(self, V):
        return self.F_a * V + self.F_b

    def measure(self, T: Optional[float] = None) -> Tuple[NDArray[float], NDArray[float]]:
        if T is not None:
            N = int(T * self.fs)
        else:
            N = int(self.T * self.fs)
        self.task.timing.cfg_samp_clk_timing(rate=self.fs, sample_mode=AcquisitionType.FINITE,
                                             samps_per_chan=N)
        t = np.arange(N) / self.fs
        data = np.asarray(self.task.read(number_of_samples_per_channel=N)).T
        force_data = self.force_from_voltage(data)
        return t, force_data

    def mean_force(self, force_in_t):
        self.local_F = np.mean(force_in_t, axis=0)

    def close(self):
        self.task.close()
