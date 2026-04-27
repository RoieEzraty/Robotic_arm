from __future__ import annotations
import logging
import numpy as np
import copy
import pathlib

import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import robot_helpers, file_helpers, helpers

if TYPE_CHECKING:
    from Logger import Logger
    from SupervisorClass import SupervisorClass
    from ForsentekClass import ForsentekClass, ForceLimitEvent

# Use tool to setup default console and file logger
tools.SetDefaultLogger(logging.INFO, f'{pathlib.Path(__file__).stem}.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class MecaClass:
    """Mecademic500 robot class for origami-arm experiments.

    Methods
    --------
    connect()
        Connect to the robot, activate it, home it, apply motion parameters from config, set working/tool frames.
    home()
        Send robot to home and wait until homing completes.
    set_frames(mod=None)
        Set tool and world reference frames (TRF/WRF) according to experiment mode.
        "stress_strain" = pole connected, long tip
        "training"      = clamp tip connected, shift origin to chain origin.
    move_to_home()
        Linearly move tip to configured home position [3[mm], 3[deg]]. Currently not in use.
    move_to_sleep_pos()
        Move the robot to the configured sleep joint configuration [6[deg]].
    move_joints(joints)
        Move the robot in joint space using a 6-DOF joint target.
    move_pos_w_mid(points, Sprvsr=None)
        Move to either 3-DOF target ``(x, y, theta_z)`` or 6-DOF robot position. 
        For 3-DOF targets, sanitize against workspace limits first, using sanitize_target().
    move_lin_or_pose(target, mod)
        Execute linear or position move. Try recovery if robot enters error state. Used in move_pos_w_mid()
    _get_current_pos()
        Read current robot position and store corresponding 3-DOF in ``self.current_pos``.
    pts_3_to_6(points)
        Convert simulation-space target ``(x, y, theta_z)`` into 6-DOF robot position.
    pts_6_to_3(points)
        Convert 6-DOF robot position into simulation-space ``(x, y, theta_z)``.
    sim_to_robot_theta(theta_sim_deg)
        Convert simulation rotation convention into robot rotation convention.
    robot_to_sim_theta(theta_robot_deg)
        Convert robot rotation convention into simulation rotation convention.
    sanitize_target(points3, Sprvsr)
        Clamp 3-DOF target to allowed chain and robot workspace limits.
    clamp_to_circle_xy(x, y, theta_z, Sprvsr, robot_margin=1.0, chain_margin=3.0)
        Clamp planar target coordinates according to robot reach and effective chain radius.
        Used inside sanitize_target()
    disconnect()
        Disconnect from the robot controller.
    _recover_robot()
        Clear motion queue, reset errors, reactivate, and re-home the robot.
    """

    ip: str                           # robot ip is always "192.168.0.100"
    robot: object                     # robot object initialized with mecademicpy.robot_initializer 
    pos_home: tuple[float, float, float, float, float, float]      # 3 positions [mm] and 3 angles [deg] of home
    joints_sleep: tuple[float, float, float, float, float, float]  # 6 joints angles of sleep position [6*deg]
    pos_origin: NDArray[np.float64]                                # (x, y, z) [mm] of chain origin. 
                                                                   # careful for short and long table holders
    norm_length: float                # normalized length (single link) [mm]. For loss calculation etc. 
    norm_angle: float                 # normalized angle [deg]. For loss calculation etc.
    theta_sim_to_robot: float         # orientation of robot and simulation angle. Tip is tilted down so -1.
    R_robot: float                    # Allowed range of motion for robot tip in XY plane. 
                                      # Calculated by me as circle with points fit_circle_xy(pts_robot)
    R_chain: float                    # Allowed range of motion so as to not break chain.
                                      # Calculate at every step, accounting for tip and total angles.
    x_TRF: float                      # shift in relative x direction due to chain tip being farther than robot tip
    y_TRF: float                      # shift in relative y direction, generally zero
    z_TRF: float                      # shift in height. different when pole or chain clamp are mounted.
    x_WRF: float                      # shift in x direction also due chain base not in x=0
    y_WRF: float                      # shift in y direction also due chain base not in y=0
    current_pos: NDArray[np.float64]  # (x, y, theta) [mm, mm, deg] current position of robot tip
    pole_rad: float                   # if pole is mounted, account for its radius shifting position of chain tip

    def __init__(self, Sprvsr: "SupervisorClass", CFG=None, margin: float = 1.0):
        """
        Parameters
        ----------
        margin : float, safety margin [mm] subtracted from the chain-based workspace radius
        """
        if CFG is None:
            from arm_config import CFG as DEFAULT_CFG
            CFG = DEFAULT_CFG
        self.CFG = CFG

        # ip priority: explicit arg > config file
        self.ip = str(CFG.Variabs.ip)
        logger.info(f'Robot IP: {self.ip}')
        
        # robot instance
        self.robot = initializer.RobotWithTools()

        # get origin
        self.pos_home = tuple(CFG.Variabs.pos_home)
        self.joints_sleep = tuple(CFG.Variabs.joints_sleep)
        self.pos_origin = np.asarray(CFG.Variabs.pos_origin, dtype=float)

        self.norm_length = float(CFG.Variabs.norm_length)
        self.norm_angle = float(CFG.Variabs.norm_angle)  # [deg]

        self.theta_sim_to_robot = float(CFG.Variabs.theta_sim_to_robot)

        limits_path = str(CFG.Variabs.limits_path)
        pts_robot = file_helpers.load_perimeter_xy(limits_path, x_col="x", y_col="y")
        # allowed radius of motion, and offset origin
        self.R_robot = helpers.fit_circle_xy(pts_robot)
        print(f"Radius allowed due to robot margins, = {self.R_robot:.2f}")
        self.R_chain = (Sprvsr.L - margin) * (Sprvsr.H + 1)
        print(f"Radius allowed due to chain length, = {self.R_chain:.2f}")

        # These will be assigned later under set_targets
        self.x_TRF = 0.0
        self.y_TRF = 0.0
        self.z_TRF = 0.0
        self.x_WRF = 0.0
        self.y_WRF = 0.0
        self.current_pos = np.zeros(3, dtype=float)
        self.pole_rad = 0.0  # will be set only if mod=="stress_strain"

    def connect(self) -> None:
        """Connect, activate, home, configure, and initialize robot frames."""
        # try to connect
        try:
            self.robot.Connect(address=self.ip, disconnect_on_exception=False)
        except mdr.InterruptException as exception:
            logger.error(f'Robot operation was interrupted: {exception}...')
        except mdr.CommunicationError as exception:
            logger.error(f'Failed to connect to the robot: {exception}...')
        except mdr.DisconnectError as exception:
            logger.error(f'Disconnected from the robot: {exception}...')
        except mdr.MecademicNonFatalException as exception:
            logger.error(f'Robot exception occurred: {exception}...')
        except KeyboardInterrupt:
            logger.warning('Control-C pressed, quitting')
            pass

        # activate
        logger.info("Activating")
        initializer.reset_sim_mode(self.robot)
        self.robot.ActivateRobot()
        self.robot.WaitActivated()
        logger.info("Robot activated")
  
        self.home()  # save as home

        # apply hyperparams from config
        logger.info("Applying config parameters")
        robot_helpers.apply_motion_config(self.robot, self.CFG)
        logger.info("Config parameters done")

        # set offset due to force sensor and gripper
        self.set_frames()
        
        # log (and display) current position
        logger.info(f'Current arm position: {tuple(self.robot.GetPose())}')

    def home(self) -> None:
        """Home the robot and wait for completion. Used in connect()"""
        logger.info("Homing")
        self.robot.Home()
        self.robot.WaitHomed()
        logger.info("Robot at home")

    def set_frames(self, mod: Optional[str] = None) -> None:
        """Set robot tool/world frames (x, y, z offsets), operation mode specific.

        Parameters
        ----------
        mod : "training"      = chain tip clamp mounted on robot tip
              "stress_strain" = pole is connected to robot tip, sinlge hinge connected to table.
              "elevated"      = elevate tip a little more to allow connecting chain
        """
        # initiate
        self.x_TRF, self.y_TRF, self.z_TRF = 0.0, 0.0, 0.0
        self.x_WRF, self.y_WRF = 0.0, 0.0

        # z in all cases should account for holder + load cell
        load_cell_thick = float(self.CFG.Variabs.load_cell_thick)
        cable_holder_len = float(self.CFG.Variabs.cable_holder_len)
        self.z_TRF += load_cell_thick + cable_holder_len

        if mod == 'stress_strain':
            # ------ TRF ------
            # set origin at chain base and tip at chain end
            x_offset_tip = float(self.CFG.Variabs.offset_chain_tip)  # tip, negative sign
            self.x_TRF += - x_offset_tip

            # add offset due to pole tip, up to its middle
            pole_len_mid = float(self.CFG.Variabs.pole_len_mid)
            self.z_TRF += pole_len_mid

            # ------ WRF ------
            self.x_WRF += self.pos_origin[0]
            self.y_WRF += self.pos_origin[1]

            self.pole_rad = float(self.CFG.Variabs.pole_rad)
        elif mod == 'training' or mod == 'elevated':
            # ------ TRF ------
            # set tip at chain end
            x_offset_tip = float(self.CFG.Variabs.offset_chain_tip)  # tip, negative sign
            self.x_TRF += x_offset_tip

            holder_len = float(self.CFG.Variabs.holder_len)
            self.z_TRF += holder_len

            # ------ WRF ------
            self.x_WRF += self.pos_origin[0]
            self.y_WRF += self.pos_origin[1]

            if mod == 'elevated':  # elevate tip a little more to allow connecting chain
                self.z_TRF += 1/2*holder_len
        else:  # cable holder always positioned
            holder_len = float(self.CFG.Variabs.holder_len)
            self.z_TRF += holder_len

        # set in robot
        self.robot.SetTrf(self.x_TRF, self.y_TRF, self.z_TRF, 0.0, 0.0, 0.0)
        self.robot.SetWRF(self.x_WRF, self.y_WRF, 0.0, 0.0, 0.0, 0.0)

    def move_to_home(self) -> None:
        """Move linearly to configured home pose."""
        logger.info('Moving the robot to home')
        self.robot.WaitIdle()
        self.robot.MoveLin(*self.pos_home)
        self.robot.WaitIdle()

    def move_to_sleep_pos(self) -> None:
        """Move to the configured sleep joint pose."""
        logger.info('Moving the robot to sleep position')
        self.move_joints(self.joints_sleep)

    def move_joints(self, joints: NDArray[np.float64] | tuple[float, ...] | list[float]) -> None:
        """Move robot to ``joints`` in joint space.

        Parameters
        ----------
        joints : array-like, shape (6,), joint-space target.
        """
        if np.size(joints) != 6:
            logger.info("position given is not x, y, theta_z or 6 DOFs")
            return

        target = copy.copy(joints)
        logger.info("Moving the robot - joints")
        robot_helpers.assert_ready(self.robot)
        self.robot.MoveJoints(*target)
        self.robot.WaitIdle()
        logger.info("Robot done moving")

    def move_pos_w_mid(self, points: NDArray[np.float64], Sprvsr: Optional["SupervisorClass"] = None,
                       Snsr: Optional["ForsentekClass"] = None) -> bool:
        """Move to either 3-DOF simulation target or 6-DOF robot target.

        Parameters
        ----------
        points          : NDArray[np.float64] Either ``(x, y, theta_z)`` [mm, mm, deg] in simulation coordinates
                          or ``(x, y, z, rx, ry, rz)`` [3*mm, 3*deg] in robot position coordinates.
        Sprvsr          : SupervisorClass, optional. Required when ``points`` is 3-DOF.
        Snsr            : ForsentekClass, optional. If provided with ``force_threshold``, guard motion by force.
        force_threshold : float, optional. Local-frame force threshold [N].
        force_mode      : {"norm_xy", "norm_xyz", "abs_axes"}. Rule for threshold comparison.
        force_chunk_T   : float, optional. Duration [s] of each sensor read chunk during motion.
        force_consecutive : int, optional. Consecutive over-threshold chunks required before stopping.
        revert_on_force : bool, optional. If True, return to pose before guarded move after force event.

        Returns
        -------
        bool
            True if the motion finished normally. False if force guard stopped/reverted the motion.

        Notes
        -----
        The original behavior is unchanged if ``Snsr`` or ``force_threshold`` is omitted.
        """
        target: tuple[float, float, float, float, float, float]

        if np.size(points) == 3:
            if Sprvsr is None:
                raise ValueError("Sprvsr must be provided when moving with a 3-DOF target.")
            point_sanit = self.sanitize_target(points, Sprvsr)
            target = self.pts_3_to_6(point_sanit)
        elif np.size(points) == 6:
            target_arr = np.asarray(points, dtype=float).copy()
            target_arr[-1] = self.sim_to_robot_theta(target_arr[-1])
            target = tuple(target_arr)
        else:
            logger.info("position given is not x, y, theta_z or 6 DOFs")
            return False

        # here previously was "correct for too big a twist"

        # move one final time
        # self.move_lin_split(target)
        completed = self.move_lin_or_pose(target, Snsr, mod='lin')
        if completed:
            self.current_pos = self.pts_6_to_3(target)
        else:
            self._get_current_pos()
        return completed

    def move_lin_or_pose(self, target: tuple[float, float, float, float, float, float],
                         Snsr: Optional["ForsentekClass"] = None, mod: str = 'lin',
                         revert_on_force: bool = True) -> bool:
        """Execute either ``MoveLin`` or ``MovePose`` with optional force guarding.

        Parameters
        ----------
        target          : tuple. Robot pose ``(x, y, z, rx, ry, rz)``.
        mod             : {"lin", "pose"}. Motion primitive.
        Snsr            : ForsentekClass, optional. If provided with ``force_threshold``, guard motion by force.
        force_threshold : float, optional. Local-frame force threshold [N].
        force_mode      : {"norm_xy", "norm_xyz", "abs_axes"}. Rule for threshold comparison.

        Returns
        -------
        bool
            True if the motion finished normally. False if force guard stopped/reverted the motion.
        """
        robot_helpers.assert_ready(self.robot)

        if mod not in ("lin", "pose"):
            raise ValueError("mod must be either 'lin' or 'pose'.")

        def motion() -> None:
            if mod == 'lin':
                self.robot.MoveLin(*target)
            else:  # mod == 'pose'
                self.robot.MovePose(*target)
            self.robot.WaitIdle()

        try:
            if Snsr is not None and Snsr.force_threshold is not None:
                return self.run_with_force_guard(motion, Snsr, revert_on_force=revert_on_force)

            motion()
            return True

        except (mdr.MecademicNonFatalException, mdr.MecademicFatalException, mdr.InterruptException,
                Exception):
            # Robot likely entered error on invalid move. Preserve previous recovery behavior.
            self._recover_robot()
            motion()
            return True

    def run_with_force_guard(self, motion: Callable[[], None], Snsr: "ForsentekClass",
                             revert_on_force: bool = True) -> bool:
        """Run a blocking robot-motion callable while monitoring force in the background.

        Parameters
        ----------
        motion            : callable. Function that starts robot motion and blocks until it ends.
        Snsr              : ForsentekClass. Sensor instance used for the background force listener.
        force_threshold   : float. Local-frame force threshold [N].
        force_mode        : {"norm_xy", "norm_xyz", "abs_axes"}. Rule for threshold comparison.
        force_chunk_T     : float, optional. Duration [s] of each sensor read chunk.
        revert_on_force   : bool, optional. If True, return to the robot pose before the motion.

        Returns
        -------
        bool
            True if ``motion`` completed normally. False if force exceeded the threshold.

        Notes
        -----
        Use this for additional movements that are not routed through ``move_pos_w_mid()``.
        """
        start_pose = tuple(np.asarray(self.robot.GetPose(), dtype=float))
        force_event: Optional["ForceLimitEvent"] = None

        def on_force_limit(event: "ForceLimitEvent") -> None:
            logger.warning("Force threshold exceeded: %.6g N >= %.6g N, force=%s",
                           event.peak_value, event.threshold, event.force)
            self.stop_motion()

        Snsr.start_force_listener(on_limit=on_force_limit)
        try:
            try:
                print('motion on in run_with_force_guard')
                motion()
            except (mdr.MecademicNonFatalException, mdr.MecademicFatalException, mdr.InterruptException,
                    Exception):
                # If the force listener stopped the robot, do not recover and retry the unsafe motion.
                if not Snsr.force_limit_triggered():
                    raise
            finally:
                force_event = Snsr.stop_force_listener()
                print('force_event', force_event)

            if force_event is None:
                return True

            if revert_on_force:
                self._revert_to_pose(start_pose)

            return False

        except Exception:
            if Snsr.force_listener_running():
                Snsr.stop_force_listener()
            raise

    def stop_motion(self) -> None:
        """Stop current robot motion and clear queued motion when supported by the Mecademic API."""
        for method_name in ("PauseMotion", "ClearMotion"):
            fn = getattr(self.robot, method_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception as exception:
                    logger.warning("%s failed while stopping motion: %s", method_name, exception)

    def _revert_to_pose(self, pose: tuple[float, float, float, float, float, float]) -> None:
        """Return to a previously saved robot pose after a force-limited stop."""
        resume = getattr(self.robot, "ResumeMotion", None)
        if callable(resume):
            try:
                resume()
            except Exception as exception:
                logger.warning("ResumeMotion failed before revert: %s", exception)

        robot_helpers.assert_ready(self.robot)
        self.robot.MoveLin(*pose)
        self.robot.WaitIdle()
        self._get_current_pos()

    def _get_current_pos(self) -> None:
        """Read current robot position and store corresponding 3-DOF simulation state."""
        current_pos_6 = np.asarray(self.robot.GetPose(), dtype=float)  # 6 DOFs from robot
        theta_z = self.robot_to_sim_theta(current_pos_6[-1])  # robot and simulation angles don't agree
        self.current_pos = np.array([current_pos_6[0], current_pos_6[1], theta_z], dtype=float)  # 3 DOFs

    def pts_3_to_6(self, points: NDArray[np.float64] | tuple[float, ...] | list[float]
                   ) -> tuple[float, float, float, float, float, float]:
        """Convert ``(x, y, theta_z)`` [mm, mm, deg] into robot position [3*mm, 3*deg] using configured home"""
        pts = np.asarray(points, dtype=float)
        return (pts[0], pts[1], self.pos_home[2], self.pos_home[3], self.pos_home[4], 
                self.sim_to_robot_theta(pts[2]))
        
    def pts_6_to_3(self, points: NDArray[np.float64] | tuple[float, ...] | list[float]
                   ) -> NDArray[np.float64]:
        """Convert robot position [3*mm, 3*deg] into simulation position ``(x, y, theta_z)`` [mm, mm, deg]."""
        pts = np.asarray(points, dtype=float)
        return np.array([pts[0], pts[1], self.robot_to_sim_theta(pts[-1])], dtype=float)

    def sim_to_robot_theta(self, theta_sim_deg: float) -> float:
        """Convert simulation rotation convention into robot rotation convention."""
        return self.theta_sim_to_robot * float(theta_sim_deg)  # invert CCW->CW

    def robot_to_sim_theta(self, theta_robot_deg: float) -> float:
        """Convert robot rotation convention into simulation rotation convention."""
        return self.theta_sim_to_robot * float(theta_robot_deg)  # invert CW->CCW

    def sanitize_target(self, points3: NDArray[np.float64] | tuple[float, ...] | list[float],
                        Sprvsr: "SupervisorClass") -> NDArray[np.float64]:
        """Clamp a 3-DOF target to allowed workspace limits."""
        x, y, theta_z = map(float, points3)
        x, y = self.clamp_to_circle_xy(x, y, theta_z, Sprvsr)
        return np.array([x, y, theta_z])

    def clamp_to_circle_xy(self, x: float, y: float, theta_z: float, Sprvsr: "SupervisorClass",
                           robot_margin: float = 1.0, chain_margin: float = 3.0) -> tuple[float, float]:
        """Clamp planar coordinates to robot and chain workspace constraints.
        If (x,y) is outside the circle of radius (R-margin), project it to the nearest point on the circle.

        Parameters
        ----------
        x, y, theta_z : floats. Requested planar coordinates in simulation frame [mm], [mm], [deg].
        robot_margin : float, Safety margin from robot reach boundary [mm].
        chain_margin : float, Safety margin from effective chain radius [mm].

        Returns
        -------
        tuple[float, float]. Clamped planar coordinates.
        """
        # account for previous total angle to calculate current total angle, in [deg]
        prev_total_angle = float(getattr(Sprvsr, "total_angle", 0.0))

        # calculate current total angle
        Sprvsr.total_angle = helpers.get_total_angle(Sprvsr.L, np.array([x, y]), prev_total_angle)  # [deg]

        # effective radius of chain
        R_eff = helpers.effective_radius(self.R_chain, Sprvsr.L, Sprvsr.total_angle, theta_z)
        print(f'effective Radius inside clamp_to_circle_xy = {R_eff}')

        x_tip, y_tip = helpers.TRF_to_robot_tip(x, y, theta_z, self.x_TRF)
        r_robot = np.hypot(x_tip+self.pos_origin[0], y_tip+self.pos_origin[1])
        r_chain = np.hypot(x, y)

        x2, x3, y2, y3 = None, None, None, None

        # ------ chain constraint ------
        if r_chain >= (R_eff - chain_margin):
            scale = (R_eff - chain_margin) / r_chain
            x2 = x * scale
            y2 = y * scale
            print(f'clamped from x={x},y={y} to x={x2},y={y2} due to chain revolusions')
            print(f'since r_chain={r_chain}')
            print(f'but maximal R_eff of chain={R_eff}')

        # ------ robot radius constraints ------
        if r_robot >= (self.R_robot - robot_margin):
            scale = (self.R_robot - robot_margin) / r_robot
            print('scale =', scale)
            x3 = -self.pos_origin[0] + (x+self.pos_origin[0]) * scale
            y3 = -self.pos_origin[1] + (y+self.pos_origin[1]) * scale
            # x3 = x * scale
            # y3 = y * scale
            print(f'clamped from x={x},y={y} to x={x3},y={y3} due to robot limits')
            print(f'since r_robot={r_robot}')
            print(f'but maximal robot margins={self.R_robot}')

        # clamp to minimal radius detected
        x_clamp = np.nanmin(np.array([x, x2, x3], dtype=float))
        y_clamp = np.nanmin(np.array([y, y2, y3], dtype=float))
        return float(x_clamp), float(y_clamp)

    def disconnect(self) -> None:
        """Disconnect from the robot controller."""
        self.robot.Disconnect()

    def _recover_robot(self) -> None:
        """Recover robot from a controller fault and return it to a homed state."""
        logger.warning("Recovering robot from fault...")

        # Clear queue just in case, then reset and resume
        try:
            self.robot.ClearMotion()
        except Exception:
            pass

        self.robot.ResetError()
        self.robot.ResumeMotion()
        self.robot.WaitIdle()

        self.robot.DeactivateRobot()
        self.robot.WaitDeactivated()

        self.robot.ActivateRobot()
        self.robot.WaitActivated()

        self.robot.Home()
        self.robot.WaitHomed()

        logger.info("Robot recovered and ready")


# ----------------------------
# NOT IN USE
# ----------------------------

# def move_lin_split(self, target: tuple[float, float, float, float, float, float], mod: str = "lin") -> None:
#     """Execute a split move to avoid >180 deg rotation protection.

#     Parameters
#     ----------
#     target : tuple[float, float, float, float, float, float]
#         Target robot pose ``(x, y, z, rx, ry, rz)``.
#     mod : {"lin", "pose"}, default "lin"
#         Motion primitive used for each segment.
#     """
#     logger.info("Moving the robot - linear (split spatial + angular)")
#     robot_helpers.assert_ready(self.robot)

#     cur = tuple(self.robot.GetPose())
#     tx, ty, tz, trx, try_, trz = target

#     spatial_pose = (tx, ty, tz, cur[3], cur[4], cur[5])
#     logger.info("MoveLin spatial: xyz=(%.2f,%.2f,%.2f) ori_hold=(%.2f,%.2f,%.2f)", 
#                 tx, ty, tz, cur[3], cur[4], cur[5])
#     self.move_lin_or_pose(spatial_pose, mod)

#     cur2 = tuple(self.robot.GetPose())
#     start_rz = cur2[5]
#     delta = robot_helpers.shortest_delta_deg(start_rz, trz)

#     if abs(delta) < 1e-6:
#         final_pose = (tx, ty, tz, trx, try_, trz)
#         logger.info("No rotation needed; finishing with final orientation.")
#         self.move_lin_or_pose(final_pose, mod)
#         logger.info("Robot finished moving")
#         return

#     steps = robot_helpers.split_rotation(delta, max_step=160.0)
#     logger.info("Rotate in place: start_rz=%.2f, target_rz=%.2f, delta(shortest)=%.2f, steps=%d",
#                 start_rz, trz, delta, len(steps))

#     rz_running = start_rz
#     for idx, dstep in enumerate(steps, start=1):
#         rz_running += dstep
#         rot_pose = (tx, ty, tz, trx, try_, rz_running)
#         logger.info("Rotate step %d/%d: rz=%.2f (d=%.2f)", idx, len(steps), rz_running, dstep)
#         self.move_lin_or_pose(rot_pose, mod)

#     final_pose = (tx, ty, tz, trx, try_, trz)
#     logger.info("Finalize pose: rz=%.2f", trz)
#     self.move_lin_or_pose(final_pose, mod)
#     logger.info("Robot finished moving")


# def correct_too_big_rot(self, target: tuple[float, float, float, 
#                                             float, float, float]) -> Optional[NDArray[np.float64]]:
#     """Build intermediate position when the ``rz`` jump exceeds 180 deg."""
#     if hasattr(self, "current_pos"):
#         starting_pos = np.asarray(self.pts_3_to_6(self.current_pos), dtype=float)
#     else:
#         starting_pos = np.asarray(self.robot.GetPose(), dtype=float)

#     target_arr = np.asarray(target, dtype=float)
#     delta = target_arr - starting_pos
#     print(f"target={target}")
#     print(f"starting_pos={tuple(starting_pos)}")

#     rot_idx = np.array([5], dtype=int)
#     over = np.abs(delta[rot_idx]) > 180.0
#     if not np.any(over):
#         logger.info("no correction for too big rot")
#         return None

#     mid = target_arr.copy()
#     for idx, is_over in zip(rot_idx, over):
#         if is_over:
#             mid[idx] = starting_pos[idx] + 0.5 * delta[idx]
#             logger.info("Large rotation detected. Splitting MoveLin into two steps. start=%s mid=%s target=%s",
#                         tuple(starting_pos), tuple(mid), target)
#     return mid
