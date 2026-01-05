import configparser
import pathlib
import nidaqmx
import time
import numpy as np

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Tuple, Optional

from nidaqmx.constants import TerminalConfiguration, AcquisitionType


class ForsentekClass:
    def __init__(self, config_path: str = "forsentek_config.ini"):
        # config
        self.cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        self.cfg.read(pathlib.Path(config_path))

        # ip priority: explicit arg > config file
        self.DEV = self.cfg.get("ni", "DEV", fallback=None)
        CHAN = self.cfg.get("ni", "Channel", fallback=None)
        self.CHAN = f"{self.DEV}/{CHAN}"
        print('DEV', self.DEV)
        print('CHAN', self.CHAN)
        self.fs = self.cfg.getfloat("ni", "samp_freq", fallback=None)
        self.T = self.cfg.getfloat("ni", "T", fallback=None)
        self.min_val = self.cfg.getfloat("amp", "min_val", fallback=None)
        self.max_val = self.cfg.getfloat("amp", "max_val", fallback=None)

        # --- create task ---
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(self.CHAN, terminal_config=TerminalConfiguration.RSE,
                                                  min_val=self.min_val, max_val=self.max_val)

    def measure(self, T: Optional[float] = None) -> Tuple[NDArray[float], NDArray[float]]:
        if T is not None:
            N = int(T * self.fs)
        else:
            N = int(self.T * self.fs)
        self.task.timing.cfg_samp_clk_timing(rate=self.fs, sample_mode=AcquisitionType.FINITE,
                                             samps_per_chan=N)
        t = np.arange(N) / self.fs
        data = np.asarray(self.task.read(number_of_samples_per_channel=N))
        return t, data

    def mean_force(self, force_in_t):
        self.force = np.mean(force_in_t)

    def close(self):
        self.task.close()
