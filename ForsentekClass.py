from __future__ import annotations

import nidaqmx
import threading
import time
import numpy as np

from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Callable

import file_helpers
import helpers
import plot_func


class ForsentekClass:
    """Forsentek force-sensor interface and acquisition utilities.

    Methods
    --------
    force_from_voltage()
        Convert voltage samples into force samples using the daily calibration offset and sensor calibration file 
        (performed using weights).
    measure()
        Measure voltage data in time [V], optionally convert to force [N]. Returns data in time during measurement.
    calibrate_daily()
        Load calibration data, fit voltage-to-force linear relation, and set zero-force offset ``V0`` 
        either from pre-created file or from current measurement.
    mean_force()
        Compute and store mean local-frame force across time.
    close()
        Close NI-DAQmx task.
    """

    DEV: str                             # NI device name.
    CHAN_x: str                          # NI analog-input channel names.
    CHAN_y: str                          # """
    CHAN_z: str                          # """
    fs: float                            # Sampling frequency in Hz.
    T: float                             # Default measurement duration, [s].
    min_val: float                       # DAQ input voltage limits.
    max_val: float                       # Characteristic force scale used for normalization. [N]
    theta_sensor: float                  # angle of force relative to robot z axis angle
    norm_pos: float                      # normalized position [m]
    norm_angle: float                    # normalized angle [deg]
    norm_force: float                    # normalized forces [mN] from single hinge torque file
    task: nidaqmx.Task                   # NI-DAQmx acquisition task for the 3 axes.
    voltage_data: NDArray[np.float64]    # Most recently acquired voltage samples, shape ``(T*fs, 3)``.
    force_data: NDArray[np.float64]      # Most recently computed force samples, shape ``(T*fs, 3)``.
    t: NDArray[np.float64]               # Time vector corresponding to the latest acquisition, shape ``(T*fs,)``.
    local_F: NDArray[np.float64]         # Mean local-frame force over the latest dataset, shape ``(3,)``
    F_a: NDArray[np.float64]             # Calibration slope converting voltage to force ``(3,)``.
    V0: NDArray[np.float64]              # Zero-force reference voltage, shape ``(3,)``
    calibration_path: str                # path of calibration file, created during experiments.calibrate_forces_all_axes()
    V0_path: str                         # offsets file, created during calibrate_daily()
    last_force_event: "ForceLimitEvent"  # ForceLimitEvent class instance w/important force reading during listener
    force_threshold: float               # [N], force above which Snsr sends signal to stop & revert robot motion
    force_chunk_T: float                 # [s], measure forces every chunk time during robot motion
    revert_on_force: bool                # if True, revert robot motion (upon exploding force). Else, stay in place

    def __init__(self, CFG: Optional[object] = None) -> None:
        """
        Parameters
        ----------
        CFG : optional, Python configuration object. If omitted, module-level ``CFG`` imported from ``arm_config``.
        """
        if CFG is None:
            from arm_config import CFG as default_cfg
            CFG = default_cfg
        self.CFG = CFG

        # NI
        self.DEV = str(CFG.Snsr.DEV)
        chan_x = str(CFG.Snsr.Channel_x)
        chan_y = str(CFG.Snsr.Channel_y)
        chan_z = str(CFG.Snsr.Channel_z)
        self.CHAN_x = f"{self.DEV}/{chan_x}"
        self.CHAN_y = f"{self.DEV}/{chan_y}"
        self.CHAN_z = f"{self.DEV}/{chan_z}"
        self.min_val = float(CFG.Snsr.min_val)
        self.max_val = float(CFG.Snsr.max_val)
        self.task = nidaqmx.Task()
        for channel in (self.CHAN_x, self.CHAN_y, self.CHAN_z):
            self.task.ai_channels.add_ai_voltage_chan(channel, terminal_config=TerminalConfiguration.RSE,
                                                      min_val=self.min_val, max_val=self.max_val)

        # sampling and limits
        self.fs = float(CFG.Snsr.samp_freq)
        self.T = float(CFG.Snsr.T)

        # orientation
        self.theta_sensor = float(CFG.Snsr.angle)

        # single hinge stress-strain
        self.tau_of_theta = file_helpers.build_torque_from_file(CFG.Variabs.tau_file, angles_in_degrees=True)  
        tau_plus = float(np.abs(self.tau_of_theta(45)))
        tau_minus = float(np.abs(self.tau_of_theta(-45)))
        
        # normalizations for update values
        self.norm_pos = float(CFG.Sprvsr.L)
        self.norm_angle = float(180)
        self.norm_torque = np.mean([tau_plus, tau_minus])
        self.norm_force = self.norm_torque / self.norm_pos
        self.norm_force = self.norm_force  # [N]

        # dummy initializations
        self.voltage_data = np.empty((0, 3), dtype=float)
        self.force_data = np.empty((0, 3), dtype=float)
        self.t = np.empty(0, dtype=float)
        self.local_F = np.zeros(3, dtype=float)
        self.F_a = np.zeros(3, dtype=float)  # assigned later in calibrate_daily()
        self.V0 = np.zeros(3, dtype=float)
        self.calibration_path = ""  # assigned later during calibrate_daily()
        self.V0_path = ""  # assigned later during calibrate_daily()

        # force listener state
        self._task_lock = threading.RLock()
        self._listener_stop = threading.Event()
        self._listener_triggered = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None
        self._listener_error: Optional[BaseException] = None
        self.last_force_event: "ForceLimitEvent" = None
        self.force_threshold: float = float(CFG.Snsr.force_threshold)
        self.force_chunk_T: float = float(CFG.Snsr.force_chunk_T)
        self.revert_on_force: bool = CFG.Snsr.revert_on_force

    def force_from_voltage(self, voltage: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert voltage samples [V] into force [N] using the current calibration.

        Parameters
        ----------
        voltage : NDArray[np.float64]
            Voltage samples [V], typically shape ``(T*fs, 3)``.

        Returns
        -------
        NDArray[np.float64]
            Force samples [N], same shape as ``voltage``.
        """
        return self.F_a * (np.asarray(voltage, dtype=float) - self.V0)

    def measure(self, T: Optional[float] = None, mode: str = "F",
                timeout: Optional[float] = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Acquire temporal data from force sensor.

        Parameters
        ----------
        T       : float, optional, Acquisition duration [s]. If omitted, ``self.T`` is used.
        mode    : If ``"F"``, convert acquired voltage [V] into force [N]. 
                  If ``"V"``, keep only the raw voltage data, for use in experiments.calibrate_forces_1axis()
        timeout : float, optional, read timeout in seconds. If omitted, uses ``T + 1``.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            Measured data and corresponding time vector. The returned data is
            ``force_data`` [N] for ``mode == "F"`` and ``voltage_data`` [V] otherwise.

        Notes
        -----
        - Bug fix applied here: previous implementation declared tuple return type but returned
          nothing. Now returns the acquired data and time vector.
        - ``measure()`` and ``start_force_listener()`` both use same NI-DAQ task so  cannot run simultaneously.
        """
        if self.force_listener_running():
            raise RuntimeError("Cannot call measure() while the force listener is running.")

        # ------- set duration and timeout ------
        duration = self.T if T is None else float(T)
        n_samples = int(duration * self.fs)
        if n_samples <= 0:
            raise ValueError("Measurement duration must produce at least one sample.")

        read_timeout = duration + 1.0 if timeout is None else float(timeout)

        with self._task_lock:
            try:
                self.task.stop()
            except nidaqmx.errors.DaqError:
                pass

            self.task.timing.cfg_samp_clk_timing(rate=self.fs, sample_mode=AcquisitionType.FINITE,
                                                 samps_per_chan=n_samples)
            self.t = np.arange(n_samples, dtype=float) / self.fs
            self.task.start()

            # ------ measure ------
            try:
                raw_data = self.task.read(number_of_samples_per_channel=n_samples, timeout=read_timeout)
                raw_arr = np.asarray(raw_data, dtype=float)
                if raw_arr.ndim == 1:
                    self.voltage_data = raw_arr.reshape(1, -1)
                else:
                    self.voltage_data = raw_arr.T
            finally:
                try:
                    self.task.stop()
                except nidaqmx.errors.DaqError:
                    pass

        # return force or voltage as dictated by user
        if mode == "F":
            self.force_data = self.force_from_voltage(self.voltage_data)
            return self.force_data.copy(), self.t.copy()
        if mode == "V":
            return self.voltage_data.copy(), self.t.copy()

        raise ValueError("mode must be either 'F' or 'V'.")

    def start_force_listener(self, on_limit: Optional[Callable[[ForceLimitEvent], None]] = None) -> None:
        """Start background force monitoring.

        Parameters
        ----------
        threshold   : float. Force threshold [N].
        mode        : {"norm_xy", "norm_xyz", "abs_axes"}. Rule used to convert each force sample to a scalar.
        chunk_T     : float, optional. Duration [s] of each NI-DAQ read chunk. Smaller values react faster.
        consecutive : int, optional. Number of consecutive over-threshold chunks required before triggering.
        on_limit    : callable, optional. Called once with ``ForceLimitEvent`` when the threshold is exceeded.

        Notes
        -----
        This method is intended for robot motion safety. It uses the same NI-DAQ task as ``measure()``,
        so do not call ``measure()`` while the listener is running.
        """
        if self.force_threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        if self.force_chunk_T <= 0.0:
            raise ValueError("chunk_T must be positive.")
        if self.force_listener_running():
            raise RuntimeError("Force listener is already running.")

        self._listener_stop.clear()
        self._listener_triggered.clear()
        self._listener_error = None
        self.last_force_event = None

        chunk_samples = max(1, int(round(self.force_chunk_T * self.fs)))
        print('chunk_samples', chunk_samples)
        self._listener_thread = threading.Thread(target=self._force_listener_loop, args=(chunk_samples, on_limit),
                                                 daemon=True)
        self._listener_thread.start()
        print('listener started')

    def stop_force_listener(self) -> Optional[ForceLimitEvent]:
        """Stop the background force listener and return the latest threshold event, if any."""
        print('stopping force listener')
        self._listener_stop.set()

        if self._listener_thread is not None and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)

        with self._task_lock:
            try:
                self.task.stop()
            except nidaqmx.errors.DaqError:
                pass

        if self._listener_error is not None:
            raise RuntimeError("Force listener failed.") from self._listener_error

        print('listener stopped')
        return self.last_force_event

    def force_listener_running(self) -> bool:
        """Return whether the background force listener thread is alive."""
        return self._listener_thread is not None and self._listener_thread.is_alive()

    def force_limit_triggered(self) -> bool:
        """Return whether the latest listener run detected an over-threshold force."""
        return self._listener_triggered.is_set()

    def _force_listener_loop(self, chunk_samples: int,
                             on_limit: Optional[Callable[[ForceLimitEvent], None]]) -> None:
        """Read force chunks until stopped or until force exceeds threshold."""
        timeout = max(1.0, 3.0 * chunk_samples / self.fs)
        print('started _force_listener_loop')

        try:
            with self._task_lock:
                try:
                    self.task.stop()
                except nidaqmx.errors.DaqError:
                    pass

                self.task.timing.cfg_samp_clk_timing(rate=self.fs, sample_mode=AcquisitionType.CONTINUOUS,
                                                     samps_per_chan=max(chunk_samples, 2))
                self.task.start()

            print('self._listener_stop', self._listener_stop)
            while not self._listener_stop.is_set():
                print('listener stop is not set')
                with self._task_lock:
                    print('reading data')
                    raw_data = self.task.read(number_of_samples_per_channel=chunk_samples, timeout=timeout)

                raw_arr = np.asarray(raw_data, dtype=float)
                if raw_arr.ndim == 1:
                    voltage_chunk = raw_arr.reshape(1, -1)
                else:
                    voltage_chunk = raw_arr.T

                force_chunk = self.force_from_voltage(voltage_chunk)
                self.voltage_data = voltage_chunk.copy()
                self.force_data = force_chunk.copy()
                self.t = np.arange(force_chunk.shape[0], dtype=float) / self.fs

                force_values = np.linalg.norm(force_chunk[:, :3], axis=1)

                idx = int(np.argmax(force_values))
                peak = float(force_values[idx])
                print('peak=', peak)

                if peak >= self.force_threshold:
                    self.last_force_event = ForceLimitEvent(t_unix=time.time(), force=force_chunk[idx, :].copy(),
                                                            peak_value=peak, threshold=self.force_threshold)
                    self._listener_triggered.set()
                    if on_limit is not None:
                        on_limit(self.last_force_event)
                    break

        except BaseException as exception:
            self._listener_error = exception
        finally:
            with self._task_lock:
                try:
                    self.task.stop()
                except nidaqmx.errors.DaqError:
                    pass

    def calibrate_daily(self, V0_from_file: bool = True) -> None:
        """Measure the offset of linear voltage-to-force relation. 
        The slope is determined in experiments.calibrate_forces_all_axes(). This is just for offset.


        Parameters
        ----------
        V0_from_file : bool, True  = load zero-force voltage offset from file. 
                             False = measure and save offset.

        Notes
        -----
        run when sensor is free, not connected to chain or anything else
        """
        # ------ path to import calibration slope from ------
        self.calibration_path = str(self.CFG.Snsr.calibration_path)

        # ------ import voltage-force relation slope ------
        #
        voltages_arr, forces_arr, stds_arr = file_helpers.load_calibration_csv(self.calibration_path)
        
        # ------ fit measured slope, print parameters and plot ------\
        # F = a*V + b separately for x/y/z, NDArray(2,3)
        force_fit_params = helpers.fit_force_vs_voltage(voltages_arr, forces_arr, stds_arr)
        print("force_fit_params=", force_fit_params)
        plot_func.calibration_forces(voltages_arr, forces_arr, stds_arr, force_fit_params)

        # ------ offset from file ------
        self.F_a = force_fit_params[1]
        print("make sure robot is in home position=")

        # ------ load offset from file, if provided ------
        if V0_from_file:
            self.V0_path = str(self.CFG.Snsr.V0_path)
            self.V0 = np.asarray(file_helpers.load_V0(self.V0_path), dtype=float)
            print(f"Loaded V0 from {self.V0_path}: {self.V0}")
            return

        # ------ measure offset, if offset file not provided ------
        self.V0_path = "V0_latest.npz"  # default name for offset file
        voltage_data, _ = self.measure(2.0, mode="V")  # measure voltage for 2[s]
        self.V0 = np.mean(voltage_data, axis=0)
        file_helpers.save_V0(self.V0_path, self.V0)
        print(f"Saved V0 to {self.V0_path}")

    def mean_force(self, force_in_t: NDArray[np.float64]) -> NDArray[np.float64]:
        """Take mean local-frame force across time.

        Parameters
        ----------
        force_in_t : NDArray[np.float64]. Time series of force samples, shape ``(T*fs, 3)``.

        Returns
        -------
        NDArray[np.float64]. Mean local-frame force, shape ``(3,)``.
        """
        self.local_F = np.mean(np.asarray(force_in_t, dtype=float), axis=0)
        return self.local_F.copy()

    def close(self) -> None:
        """Close the NI-DAQmx task."""
        self.task.close()


@dataclass(frozen=True)
class ForceLimitEvent:
    """Information captured when the force listener crosses the safety threshold.

    Attributes
    ----------
    t_unix     : float, Unix timestamp at which the threshold crossing was detected.
    force      : NDArray[np.float64] Local-frame forces ``[Fx, Fy, Fz]`` [N] that triggered event.
    peak_value : float, Scalar force value [N].
    threshold  : float, Force threshold [N] used for the listener.
    """

    t_unix: float
    force: NDArray[np.float64]
    peak_value: float
    threshold: float
