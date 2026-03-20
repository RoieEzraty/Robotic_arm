import nidaqmx
import time
import numpy as np

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Tuple, Optional

import file_helpers, helpers, plot_func

from nidaqmx.constants import TerminalConfiguration, AcquisitionType


class ForsentekClass:
    def __init__(self, CFG=None):
        if CFG is None:
            from arm_config import CFG as DEFAULT_CFG
            CFG = DEFAULT_CFG
        self.CFG = CFG

        self.DEV = str(CFG.Snsr.DEV)
        CHAN_x = str(CFG.Snsr.Channel_x)
        CHAN_y = str(CFG.Snsr.Channel_y)
        CHAN_z = str(CFG.Snsr.Channel_z)
        self.CHAN_x = f"{self.DEV}/{CHAN_x}"
        self.CHAN_y = f"{self.DEV}/{CHAN_y}"
        self.CHAN_z = f"{self.DEV}/{CHAN_z}"
        self.fs = float(CFG.Snsr.samp_freq)
        self.T = float(CFG.Snsr.T)
        self.min_val = float(CFG.Snsr.min_val)
        self.max_val = float(CFG.Snsr.max_val)

        # angle between x of force sensor and x of robot
        self.theta_sensor = float(CFG.Snsr.angle)

        # ------ create task ------
        self.task = nidaqmx.Task()
        # self.task.ai_channels.add_ai_voltage_chan(self.CHAN,
        #                                           terminal_config=TerminalConfiguration.RSE,
        #                                           min_val=self.min_val, max_val=self.max_val)
        for ch in (self.CHAN_x, self.CHAN_y, self.CHAN_z):
            self.task.ai_channels.add_ai_voltage_chan(ch, terminal_config=TerminalConfiguration.RSE,
                                                      min_val=self.min_val, max_val=self.max_val)

        self.norm_force = float(CFG.Snsr.norm_force)

    def force_from_voltage(self, V):
        # return self.F_a * V + self.F_b
        return self.F_a*(V-self.V0)

    def measure(self, T: Optional[float] = None, mode: str = 'F',
                timeout=None) -> Tuple[NDArray[float], NDArray[float]]:
        if T is None:
            T = self.T

        N = int(T * self.fs)

        # timeout should be longer than the acquisition duration
        if timeout is None:
            timeout = T + 1.0  # 1s margin

        # If a previous finite acquisition is still armed/running, stop it before reconfig
        try:
            self.task.stop()
        except nidaqmx.errors.DaqError:
            pass  # task might not be started yet

        # self.task.timing.cfg_samp_clk_timing(rate=self.fs, sample_mode=AcquisitionType.FINITE,
        #                                      samps_per_chan=N)
        # self.voltage_data = np.asarray(self.task.read(number_of_samples_per_channel=N)).T

        self.task.timing.cfg_samp_clk_timing(
            rate=self.fs,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=N,
        )

        self.t = np.arange(N) / self.fs

        # Explicitly start so the timing engine is definitely running
        self.task.start()

        try:
            self.voltage_data = np.asarray(self.task.read(number_of_samples_per_channel=N, 
                                                          timeout=timeout)).T
        finally:
            # For finite tasks, stopping after read keeps the next call clean
            try:
                self.task.stop()
            except nidaqmx.errors.DaqError:
                pass

        self.t = np.arange(N) / self.fs
        
        if mode == 'F':
            self.force_data = self.force_from_voltage(self.voltage_data)

    def calibrate_daily(self, V0_from_file: bool = True) -> None:
        self.calibration_path = str(self.CFG.Snsr.calibration_path)

        voltages_arr, forces_arr, stds_arr = file_helpers.load_calibration_csv(self.calibration_path)
        force_fit_params = helpers.fit_force_vs_voltage(voltages_arr, forces_arr, stds_arr)
        print('force_fit_params=', force_fit_params)
        plot_func.calibration_forces(voltages_arr, forces_arr, stds_arr, force_fit_params)
        # self.F_a, self.F_b = force_fit_params[0], force_fit_params[1]
        self.F_a = force_fit_params[1]
        print('make sure robot is in home position=')
        if V0_from_file:
            self.V0_path = str(self.CFG.Snsr.V0_path)
            self.V0 = file_helpers.load_V0(self.V0_path)
            print(f"Loaded V0 from {self.V0_path}: {self.V0}")
        else:
            self.V0_path = "V0_latest.npz"
            self.measure(2, mode='V')
            self.V0 = np.mean(self.voltage_data, axis=0)
            file_helpers.save_V0(self.V0_path, self.V0)
            print(f"Saved V0 to {self.V0_path}")

    def mean_force(self, force_in_t):
        self.local_F = np.mean(force_in_t, axis=0)

    def close(self):
        self.task.close()
