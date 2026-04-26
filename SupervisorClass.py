from __future__ import annotations

import numpy as np
from numpy import array, zeros
from numpy.typing import NDArray

import copy

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import experiments, file_helpers, helpers, plot_func

if TYPE_CHECKING:
    from arm_config import Config
    from ForsentekClass import ForsentekClass
    from MecaClass import MecaClass


class SupervisorClass:
    """Supervisor-side: parameters, datasets, measurements, and loss.

    Methods:
    --------
    init_dataset(m=None, Snsr=None) -> None
        Initialize supervisor dataset, depending on selected experiment type.
        If if ``measure_des``, measure desired force response using robot (``m``) and sensor (``Snsr``).
    draw_measurement(self, t) -> None
        Load Measurement position at step ``t`` into ``self.pos``.
    global_force(Fx, Fy) -> np.ndarray
        Convert local force measurements into the global frame. Returns array ``[Fx_global, Fy_global]``.
    calc_loss(t, norm_force) -> None
        Compute and store loss, normalized loss and its MSE at time step ``t``.
    calc_tip_update(m: "MecaClass", t, correct_for_total_angle) -> None
        Calculate Update modality tip position at training time step ``t``, based on loss.

    Helpers:
    --------
    _infer_T_from_dataset(dataset_path) -> int
        Number of steps from a predetermined dataset file.
    """

    experiment: str                               # experiment mode from ``CFG.Sprvsr.experiment``.
    dataset_type: str                             # dataset mode from ``CFG.Sprvsr.dataset_type``.
    dataset_path: str                             # path of file where tip positions during measurement is found
    T: int                                        # Number of supervisor steps, int.
    L: float                                      # length of 3d print edge + tape
    H: int                                        # number of hinges in chain
    convert_F: float                              # scalar converting from Forsentek force [N] to supervisor [mN]
    origin_rel_to_sim: NDArray[np.float64]        # the (0, 0) of simulation, tricky but this is it
    pos_in_t: NDArray[np.float64]                 # tip positions for each T for Measurement, shape ``(T, 3)``. [mm, mm, deg]
    F_in_t: NDArray[np.float64]                   # Measured global force at each step, shape ``(T, 2)``. [mN, mN]
    desired_F_in_t: NDArray[np.float64]           # Desired force at each step, shape ``(T, 2)``. [mN, mN]
    pos_update_in_t: NDArray[np.float64]          # Update values, shape ``(T, 3)`` [mm, mm, deg]
    total_angle_update_in_t: NDArray[np.float64]  # wrapped angle between tip and base, CCW from x axis, shape ``(T,)``, [deg]
    loss_in_t: NDArray[np.float64]                # Loss for every T (pos_in_t), shape ``(T, 2)``. [mN, mN]
    loss_norm_in_t: NDArray[np.float64]           # Loss_in_t normalized by typical force, shape ``(T, 2)``
    loss_MSE_in_t: NDArray[np.float64]            # Mean Squared Error of the loss, shape ``(T,)``
    pos: NDArray[np.float64]                      # current tip position, shape ``(3,)``
    loss: NDArray[np.float64]                     # current loss, shape ``(2,)``
    loss_norm: NDArray[np.float64]                # current normalized loss, shape ``(2,)``
    loss_MSE: float                               # Mean Square error of current loss
    Fx: float                                     # current sensed force in global x direction [mN]
    Fy: float                                     # current sensed force in global x direction [mN]
    alpha: float                                  # learning rate for training update rule
    rand_key_dataset: float                       # random seed to initiate dataset, for reproducibility
    normalize_step: bool                          # boolean whether to normalize step size if non-vanishing or not

    def __init__(self, CFG: Optional["Config"] = None, supress_prints: bool = True) -> None:
        """
        Parameters
        ----------
        CFG: Python configuration object. If omitted, module-level ``CFG`` imported from ``arm_config``.
        """
        # load config parameters
        if CFG is None:
            from arm_config import CFG as default_cfg
            CFG = default_cfg

        # general experiment information
        self.experiment = str(CFG.Sprvsr.experiment)
        self.dataset_type = str(CFG.Sprvsr.dataset_type)
        self.dataset_path = str(CFG.Sprvsr.dataset_path)
        self.update_scheme = str(CFG.Sprvsr.update_scheme)

        # orientation
        self.origin_rel_to_sim = np.asarray(CFG.Sprvsr.origin_rel_to_sim, dtype=float)

        # experiment-specific parameters
        if self.experiment == "training":
            self.T = int(CFG.Sprvsr.T)
            self.rand_key_dataset = int(CFG.Sprvsr.rand_key_dataset)
            self.alpha = float(CFG.Sprvsr.alpha)
        elif self.experiment == "predetermined training":
            self.T = self._infer_T_from_dataset(self.dataset_path)
            self.alpha = float(CFG.Sprvsr.alpha)
        elif self.experiment == "sweep":
            self.T = int(CFG.Sprvsr.sweep_T)
            self.x_range = int(CFG.Sprvsr.x_range)
            self.y_range = int(CFG.Sprvsr.y_range)
            self.theta_range = int(CFG.Sprvsr.theta_range)
            self.alpha = float(CFG.Sprvsr.alpha)
        else:
            raise ValueError(f"Unsupported experiment mode: {self.experiment}")

        # tip motion parameters
        self.normalize_step = CFG.Sprvsr.normalize_step
        self.invert_delta_tip = False

        # tip restart bookkeeping for origin-cut handling
        self.origin_cut_restart_count = 0  # consecutive origin-cut restarts
        self.coil_count = 0  # consecutive restarts due to tip coiling
        self.last_restart_reason: Optional[str] = None  # None | "origin_cut" | "coil"
        self.origin_restart_base_frac = 0.6  # base vertical offset in units of L
        self.origin_restart_step_frac = 0.6  # extra offset per repeated cut, in units of L

        # chain and force
        self.L = float(CFG.Sprvsr.L)
        self.H = int(CFG.Sprvsr.H)
        self.convert_F = float(CFG.Sprvsr.convert_F)
        self.norm_pos = self.L
        self.norm_angle = float(np.pi)

        # initialize empty parameters in t
        self.F_in_t = np.zeros((self.T, 2), dtype=float)
        self.desired_F_in_t = np.zeros((self.T, 2), dtype=float)
        self.pos_update_in_t = np.zeros((self.T, 3), dtype=float)
        self.total_angle_update_in_t = np.zeros(self.T, dtype=float)
        self.loss_in_t = np.zeros((self.T, 2), dtype=float)
        self.loss_norm_in_t = np.zeros((self.T, 2), dtype=float)
        self.loss_MSE_in_t = np.zeros(self.T, dtype=float)
        self.pos_in_t = np.zeros((self.T, 3), dtype=float)
        
        # initialize instantaneous parameters
        self.pos = np.zeros(3, dtype=float)
        self.loss = np.zeros(2, dtype=float)
        self.loss_norm = np.zeros(2, dtype=float)
        self.loss_MSE = 0.0
        self.Fx = 0.0
        self.Fy = 0.0

        # prints during run if True
        self.supress_prints = supress_prints

    def init_dataset(self, dataset_path: str = "dataset.csv", out_path: str = "dataset.csv",
                     measure_des: bool = False, m: Optional["MecaClass"] = None,
                     Snsr: Optional["ForsentekClass"] = None) -> None:
        """Initialize tip Measurement values for current experiment.

        Parameters
        ----------
        dataset_path : str, source dataset path. If not provided, measures it's own and saves into file
        out_path     : str, output path. Used when ``measure_des`` == ``True``.
        measure_des  : If ``True``, measure desired forces for training configuration, write to ``out_path``.
        """

        # ------- optionally measure the dataset ------
        if measure_des:
            if m is None or Snsr is None:
                raise ValueError("measure_des=True requires both 'm' and 'Snsr' instances.")
            print("measuring desired forces in training configuration solely")

            # measurement experiment
            pos_base, force_base = experiments.sweep_measurement_fixed_origami(m, Snsr, self, path=dataset_path)

            # write to file, do not save in self
            file_helpers.write_supervisor_dataset(pos_base, force_base, out_path)

        # ------- load dataset regardless of measurement ------
        if self.experiment == "training":
            rng = np.random.default_rng(self.rand_key_dataset)

        if measure_des:  # use forces you just saved to output file
            pos_force_rows = file_helpers.load_pos_force(out_path)
        else:  # import from file
            pos_force_rows = file_helpers.load_pos_force(dataset_path)

        if self.experiment == "training":
            pos_base = array([row["pos"] for row in pos_force_rows], dtype=float)
            force_base = array([row["force"] for row in pos_force_rows], dtype=float)
        elif self.experiment == "predetermined training":
            pos_base = array([row["pos_meas"] for row in pos_force_rows], dtype=float)
            pos_update = array([row["pos_update"] for row in pos_force_rows], dtype=float)
            force_base = array([row["force_meas"] for row in pos_force_rows], dtype=float)
        else:
            raise ValueError("init_dataset currently supports only training modes.")

        num_rows = int(pos_base.shape[0])
        if num_rows == 0:
            raise ValueError("Dataset is empty.")

        # ------ fill in arrays in time ------
        if self.experiment == "training":
            if self.dataset_type == "from file":  # get measured values from file, but tile them repeatedly
                repeats = int(np.ceil(self.T / num_rows))
                pos_tiled = np.tile(pos_base, (repeats, 1))
                force_tiled = np.tile(force_base, (repeats, 1))
                self.pos_in_t = pos_tiled[: self.T]
                self.desired_F_in_t = force_tiled[: self.T]
            elif self.dataset_type == "shuffle":  # shuffle dataset
                t_idx = 0
                while t_idx < self.T:
                    shuffle_idx = rng.permutation(num_rows)
                    batch_size = min(num_rows, self.T - t_idx)
                    selection = shuffle_idx[:batch_size]

                    self.pos_in_t[t_idx:t_idx+batch_size] = pos_base[selection]
                    self.desired_F_in_t[t_idx:t_idx+batch_size] = (force_base[selection])
                    t_idx += batch_size
            else:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        else:  # straight from the file
            self.pos_in_t = pos_base[1:, :]
            self.desired_F_in_t = force_base[1:, :]
            self.pos_update_in_t = pos_update[1:, :]

    def draw_measurement(self, t: int) -> None:
        """Load the measurement pose at step ``t`` into ``self.pos``.

        Parameters
        ----------
        t: Supervisor step index.
        """
        self.pos = self.pos_in_t[t, :].copy()

    def global_force(self, Snsr: "ForsentekClass", m: "MecaClass", t: Optional[int] = None,
                     plot: bool = False) -> NDArray[np.float64]:
        """Compute mean force in the global x-y frame.

        Parameters
        ----------
        t    : Optional, int, supervisor step index at which to store force.
        plot : If ``True``, plot x-y forces during measurement.

        Returns
        -------
        NDArray[np.float64], shape (2, ). Mean global force ``[Fx, Fy]``.
        """
        # setup arrays
        force_in_t = np.asarray(Snsr.force_data, dtype=float) * self.convert_F
        measure_t = np.asarray(Snsr.t, dtype=float)

        # rotate measured forces to global frame
        m._get_current_pos()
        theta = -(float(m.current_pos[-1]) - float(Snsr.theta_sensor))
        Fx_global_in_t, Fy_global_in_t = helpers.rotate_force_frame(force_in_t, theta)

        if plot:
            plot_func.force_global_during_measurement(measure_t, Fx_global_in_t, Fy_global_in_t)

        # store
        self.Fx = float(np.mean(Fx_global_in_t))
        self.Fy = float(np.mean(Fy_global_in_t))
        if t is not None:
            self.F_in_t[t, :] = array([self.Fx, self.Fy], dtype=float)

        return array([self.Fx, self.Fy], dtype=float)

    def calc_loss(self, t: int, norm_force: float) -> None:
        """Compute and store loss for step ``t``. 

        Parameters
        ----------
        t          : Supervisor step index.
        norm_force : Normalization factor used to obtain a dimensionless loss.
        """
        # calculate
        self.loss = (self.desired_F_in_t[t, :] - self.F_in_t[t, :]) / self.convert_F  # (2,) [mN]
        self.loss_norm = self.loss / float(norm_force)  # (2,) [dimless]
        self.loss_MSE = float(np.mean(self.loss_norm ** 2))  # (1,) [dimless]

        # store
        self.loss_in_t[t, :] = self.loss
        self.loss_norm_in_t[t, :] = self.loss_norm
        self.loss_MSE_in_t[t] = self.loss_MSE

    def calc_tip_update(self, m: "MecaClass", t: int, 
                        correct_for_total_angle: Optional[bool] = True,
                        correct_for_coil: Optional[bool] = True,
                        correct_for_cut_origin: Optional[bool] = True,
                        correct_for_update_force: Optional[bool] = True) -> None:
        """Calculate commanded tip position that buckles chain, out of loss value.

        Parameters
        ----------
        t                      : Supervisor step index.
        current_tip_pos        : ndarrat(float) (2,) during measurement, used only in radial_one_to_one update function
        prev_tip_update_pos    : ndarrat(float) (2,) previous update tip pos, for inserting new top into vectors in time
        current_tip_angle      : float, during measurement, used only in radial_one_to_one update function
        prev_tip_update_angle  : float, previous update tip pos, for inserting new top into vectors in time
        correct_for_total_angle, correct_for_coil, correct_for_cut_origin : booleans, whether to correct tip pos due to:
                                                                            addition to simulation total angle,
                                                                            coiled tip (reset tip values from dataset)
                                                                            tip cuts origin (as above)
        """
        # if t == 0:
        #     prev_pos_update = self.pos_in_t[0].copy()
        # else:
        #     prev_pos_update = self.pos_update_in_t[t - 1, :].copy()

        # # calculate update
        # sgn_x = float(np.sign(prev_pos_update[0]))
        # delta_x_update = self.alpha * self.loss_norm[0] * sgn_x * m.norm_length
        # delta_y_update = -self.alpha * self.loss_norm[0] * sgn_x * m.norm_length
        # delta_theta_update = -self.alpha * self.loss_norm[1] * m.norm_angle

        # self.pos_update_in_t[t, :] = prev_pos_update + delta_update
        # print("pos_update_in_t[t, :] before correct for tot angle = ", self.pos_update_in_t[t, :])

        # # correct for total angle
        # if correct_for_total_angle:
        #     if t == 0:
        #         prev_total_angle = 0.0
        #     else:
        #         prev_total_angle = float(self.total_angle_update_in_t[t - 1])

        #     tip_pos = self.pos_update_in_t[t, :2].copy()
        #     self.total_angle = helpers.get_total_angle(self.L, tip_pos, prev_total_angle)
        #     delta_total_angle = self.total_angle - prev_total_angle
        #     self.pos_update_in_t[t, 2] += delta_total_angle
        #     self.total_angle_update_in_t[t] = self.total_angle
        #     print("current total_angle", self.total_angle)
        #     print("pos_update_in_t[t, :] after correct for tot angle = ", self.pos_update_in_t[t, :])

        # ------ delta tip and angle ------
        # through BEASTAL, one_to_one or radial_one_to_one
        dispatch = self._get_delta_dispatch()
        fn = dispatch.get(self.update_scheme, None)  # function to calculate delta tip and angle from update_scheme
        if fn is None:
            raise ValueError(f"Unknown update_scheme='{self.update_scheme}'")
        delta_tip_x, delta_tip_y, delta_angle = fn(t)
        delta_tip = array([delta_tip_x, delta_tip_y])  # assemble into 3d array
        if not self.supress_prints:
            print(f'delta_tip before corr {delta_tip}')
            print(f'delta_angle before corr {delta_angle}')

        # ------ normalize step if update is non-zero ------
        if self.normalize_step and np.linalg.norm(np.append(delta_tip, delta_angle)) > 10**(-12):
            step_size = np.linalg.norm(np.append(delta_tip, delta_angle))
            # tradeoffs since angles and positions are different units
            if self.update_scheme == 'loss_diff':
                tradeoff_pos_angle = 1
            else:
                tradeoff_pos_angle = 2
            delta_tip = copy.copy(delta_tip)/step_size*self.alpha
            delta_angle = copy.copy(delta_angle)/step_size*self.alpha * tradeoff_pos_angle
            if not self.supress_prints:
                print(f'normalized position to {delta_tip}')
                print(f'normalized angle to {float(delta_angle)}')

        # ------ insert into vectors in time ------
        # get previous tip positions
        if t == 1:
            prev_tip_update = self.pos_in_t[t, :]
        else:
            prev_tip_update = self.pos_update_in_t[t-1, :]
        if not self.supress_prints:
            print(f'prev_tip_update{prev_tip_update}')

        # invert delta_tip if required
        if self.invert_delta_tip is True:  # invert tip
            delta_tip = -delta_tip
            delta_angle = -delta_angle

        # add to tip update in time
        self.pos_update_in_t[t, :] = prev_tip_update + np.append(delta_tip, delta_angle)

        # ------ correct for total angle ------
        # add change in tip angle to the total angle from the origin
        if correct_for_total_angle:
            if t == 1:
                prev_total_angle = 0.0
            else:
                prev_total_angle = self.total_angle_update_in_t[t-1]
            total_angle = helpers.get_total_angle(self.L, self.pos_update_in_t[t, :2], prev_total_angle)
            self.total_angle_update_in_t[t] = total_angle
            delta_total_angle = total_angle - prev_total_angle
            self.pos_update_in_t[t, 2] += delta_total_angle
            if not self.supress_prints:
                print(f'total angle {total_angle}')
                print(f'add delta tip angle {delta_total_angle} to correct for total angle ')

        # ------ correct for coil or cut origin ------
        cond_coil = helpers.coil(self.pos_update_in_t[t, 2], revolutions=1.5)
        cond_cut_origin = helpers.swept_last_edge_crosses_first_edge(tip_prev=prev_tip_update[:2],
                                                                     angle_prev=prev_tip_update[2],
                                                                     tip_new=self.pos_update_in_t[t, :2],
                                                                     angle_new=self.pos_update_in_t[t, 2],
                                                                     L=self.L, include_endpoints=False)

        self.restart = False

        if correct_for_cut_origin and cond_cut_origin:
            print('origin is cut')
            self.coil_count = 0
            self.origin_cut_restart_count += 1

            # sign is just from angle direction of tip
            side_sign = np.sign(delta_angle)
            self._restart_flat_with_y_bias(t, side_sign=side_sign)
            print(f'setting update tip pos={self.pos_update_in_t[t, :2]}, angle={self.pos_update_in_t[t, 2]}')
            prev_total_angle = 0.0
            self.last_restart_reason = "origin_cut"

        elif correct_for_coil and cond_coil:
            print('coiled up too much')
            self.pos_update_in_t[t, :] = self.pos_in_t[t, :]
            self.total_angle_update_in_t[t] = 0.0
            print(f'setting update tip pos={self.pos_update_in_t[t, :2]}, angle={self.pos_update_in_t[t, 2]}')
            prev_total_angle = 0.0
            self.restart = True
            self.last_restart_reason = "coil"
            self.origin_cut_restart_count = 0
            self.coil_count += 1

        # invert sign of tip change if training fails
        # STILL TO DO

        if not self.supress_prints:
            delta_tip_after_corr = self.pos_update_in_t[t, :2] - self.pos_update_in_t[t-1, :2]
            delta_angle_after_corr = self.pos_update_in_t[t, 2] - self.pos_update_in_t[t-1, 2]
            print(f'delta_tip after correcting coil and cut origin {delta_tip_after_corr}')
            print(f'delta_angle after correcting coil and cut origin {delta_angle_after_corr}')

        # ------ update total angle -------
        self.total_angle_update_in_t[t] = helpers.get_total_angle(self.L, self.pos_update_in_t[t, :2],
                                                                  prev_total_angle)
        if not self.supress_prints:
            print(f'total angle end of calc_update {self.total_angle_update_in_t[t]}')

    # ---------------------------------------------------------------
    # Helpers for Supervisor Class
    # ---------------------------------------------------------------
    @staticmethod
    def _infer_T_from_dataset(dataset_path: str) -> int:
        """Infer number of steps from predetermined dataset file.
        The function subtracts two rows: one header and one ``t = 0`` state.

        Parameters
        ----------
        dataset_path: Path to the dataset CSV file.

        Returns
        -------
        int, number of training steps in file.
        """
        with Path(dataset_path).open("r", encoding="utf-8") as file_obj:
            return sum(1 for _ in file_obj) - 2

    def _restart_flat_with_y_bias(self, t: int, side_sign: float) -> None:
        """
        Restart from flat, but bias the tip slightly above/below the x-axis.
        Repeated origin cuts increase the vertical bias magnitude.
        """
        mag = (self.origin_restart_base_frac +
               max(0, self.origin_cut_restart_count - 1) * self.origin_restart_step_frac) * self.L

        y_restart = float(side_sign) * mag
        x_restart = float((self.H + 1) * self.L - mag)
        tip_restart = np.array([x_restart, y_restart], dtype=np.float32)
        total_angle_restart = helpers.get_total_angle(self.L, tip_restart, prev_total_angle=0.0)

        self.pos_update_in_t[t, :2] = tip_restart
        self.pos_update_in_t[t, 2] = total_angle_restart
        self.total_angle_update_in_t[t] = total_angle_restart
        self.restart = True

        if not self.supress_prints:
            print(f"cut origin for the {self.origin_cut_restart_count} time")

    def _get_delta_dispatch(self):
        """
        Map update_scheme -> function that computes (delta_tip_x, delta_tip_y, delta_angle).
        Each function must return 3 scalars in *your current convention*.

        Returns:
        --------
        function that calculates tip update values inside self.calc_update
        """
        return {"one_to_one": self._delta_one_to_one,
                "loss_diff": self._delta_loss_diff}

    def _delta_one_to_one(self, t):
        """
        change tip directly from loss, no pseudo inverse, calculations in cartesian coordinates
        dx = +alpha*loss_x*sign(y)
        dy = -alpha*loss_x*sign(x)
        dtheta = -alpha*loss_y

        Parameters:
        -----------
        t : int, current training time step

        Returns:
        --------
        3 floats of change in tip position during update
        """
        sgnx = np.sign(self.pos_update_in_t[t-1, 0])
        sgny = np.sign(self.pos_update_in_t[t-1, 1])
        if sgnx == 0.0:
            sgnx = 1
        if sgny == 0.0:
            sgny = 1
        delta_tip_x = - self.alpha * self.loss[0] * (-sgny) * self.norm_pos  # up to Mar17
        delta_tip_y = - self.alpha * self.loss[0] * (+sgnx) * self.norm_pos  # up to Mar17
        delta_angle = - self.alpha * self.loss[1] * self.norm_angle  # up to Mar17
        return delta_tip_x, delta_tip_y, delta_angle

    def _delta_loss_diff(self, t):
        sgnx = np.sign(self.pos_update_in_t[t-1, 0])
        sgny = np.sign(self.pos_update_in_t[t-1, 1])
        if sgnx == 0.0:
            sgnx = 1
        if sgny == 0.0:
            sgny = 1
        loss_diff = self.loss[0] - self.loss[1]
        loss_add = self.loss[0] + self.loss[1]
        delta_tip_x = - self.alpha * loss_diff * (-sgny) * self.norm_pos  # Mar23
        delta_tip_y = - self.alpha * loss_diff * (+sgnx) * self.norm_pos  # Mar23
        delta_angle = - self.alpha * loss_add * self.norm_angle  # Mar23
        return delta_tip_x, delta_tip_y, delta_angle

    # @staticmethod
    # def _parse_buckles_from_path(dataset_path: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    #     """Extract initial and desired buckle arrays from a dataset path.

    #     Parameters
    #     ----------
    #     dataset_path
    #         Path string containing bracketed buckle specifications.

    #     Returns
    #     -------
    #     tuple[NDArray[np.float64], NDArray[np.float64]]
    #         Parsed initial and desired buckle arrays.

    #     Raises
    #     ------
    #     ValueError
    #         If the path does not contain at least two bracketed arrays.
    #     """
    #     import re

    #     buckles = re.findall(r"\[([^\]]+)\]", dataset_path)
    #     if len(buckles) < 2:
    #         raise ValueError(
    #             "Expected at least two bracketed buckle arrays in dataset_path.",
    #         )

    #     init_buckle = np.fromstring(buckles[0], sep=" ", dtype=float)
    #     desired_buckle = np.fromstring(buckles[1], sep=" ", dtype=float)
    #     return init_buckle, desired_buckle
