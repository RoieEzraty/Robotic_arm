from __future__ import annotations
import configparser
import logging
import pathlib
import numpy as np
import copy
import ast

import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

import robot_helpers, file_helpers, helpers

if TYPE_CHECKING:
    from Logger import Logger
    from SupervisorClass import SupervisorClass

# Use tool to setup default console and file logger
tools.SetDefaultLogger(logging.INFO, f'{pathlib.Path(__file__).stem}.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class MecaClass:
    def __init__(self, Sprvsr: "SupervisorClass", margin: float = 5.0,
                 config_path: str = "robot_config.ini"):
        # config
        self.cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        self.cfg.read(pathlib.Path(config_path))

        # ip priority: explicit arg > config file
        self.ip = self.cfg.get("robot", "ip", fallback=None)
        logger.info(f'Robot IP: {self.ip}')
        
        # robot instance
        self.robot = initializer.RobotWithTools()

        # get origin
        self.pos_home = ast.literal_eval(self.cfg.get("position", "pos_home"))
        self.joints_home = ast.literal_eval(self.cfg.get("position", "joints_home"))
        self.pos_sleep = ast.literal_eval(self.cfg.get("position", "pos_sleep"))
        self.joints_sleep = ast.literal_eval(self.cfg.get("position", "joints_sleep"))
        self.pos_origin = ast.literal_eval(self.cfg.get("position", "pos_origin"))

        self.norm_length = ast.literal_eval(self.cfg.get("position", "norm_length"))
        self.norm_angle = ast.literal_eval(self.cfg.get("position", "norm_angle"))

        self.theta_sim_to_robot = ast.literal_eval(self.cfg.get("position", "theta_sim_to_robot"))

        limits_path = self.cfg.get("limits", "path")
        pts_robot = file_helpers.load_perimeter_xy(limits_path, x_col="x", y_col="y")
        # allowed radius of motion, and offset origin
        self.R_robot = helpers.fit_circle_xy(pts_robot)
        print(f"Radius allowed due to robot margins, = {self.R_robot:.2f}")
        self.R_chain = (Sprvsr.L + margin) * Sprvsr.H
        print(f"Radius allowed due to chain length, = {self.R_chain:.2f}")

    def connect(self):
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
  
        self.home()

        # apply hyperparams from config
        logger.info("Applying config parameters")
        robot_helpers.apply_motion_config(self.robot, self.cfg)
        logger.info("Config parameters done")

        # set offset due to force sensor and gripper
        self.set_TRF_wrt_holder()
        
        # log (and display) current position
        logger.info(f'Current arm position: {tuple(self.robot.GetPose())}')

    def home(self):
        logger.info("Homing")
        # robot_helpers.assert_ready(self.robot)
        self.robot.Home()
        self.robot.WaitHomed()
        logger.info("Robot at home")

    def set_TRF_wrt_holder(self):
        load_cell_thick = self.cfg.getfloat("position", "load_cell_thick", fallback=None)
        holder_len = self.cfg.getfloat("position", "holder_len", fallback=None)
        self.tip_length = load_cell_thick + holder_len
        self.robot.SetTrf(0.0, 0.0, self.tip_length, 0.0, 0.0, 0.0)

    def move_to_origin(self):
        # current_pos = self.robot.GetPose()
        # tolerance = 2  # below this no motion is accounted in z axis
        # home_joints = (10, 56.5, 0, 0, 30, 180)
        logger.info('Moving the robot to origin')
        # self.move_joints(self.joints_home)
        self.robot.WaitIdle()
        # if current_pos[2] > self.z_origin + tolerance:
        #     self.robot.MoveLin(self.x_origin/2, self.y_origin, self.z_sleep,
        #                        self.theta_x_origin, self.theta_y_origin, self.theta_z_origin)
        #     self.robot.WaitIdle()
        # else:
        #     pass
        self.robot.MoveLin(*self.pos_home)
        self.robot.WaitIdle()

    def move_to_sleep_pos(self) -> None:
        logger.info('Moving the robot to sleep position')
        self.move_joints(self.joints_sleep)

    def move_joints(self, joints) -> None:
        # if np.size(joints) == 3:
        #     current_joints = self.robot.GetJoints()
        #     target = (joints[0], joints[1], current_joints[2], current_joints[3],
        #               current_joints[4], joints[2])
        #     logger.info('poisition given is not x, y, theta_z or 6 DOFs')
        if np.size(joints) == 6:
            target = copy.copy(joints)
            logger.info('Moving the robot - joints')
            robot_helpers.assert_ready(self.robot)
            self.robot.MoveJoints(*target)
            self.robot.WaitIdle()
            logger.info('Robot done moving')
        else:
            logger.info('poisition given is not x, y, theta_z or 6 DOFs')

    def move_pos(self, points: NDArray, Sprvsr: "SupervisorClass") -> None:
        if np.size(points) == 3:
            point_sanit = self.sanitize_target(points, Sprvsr)
            target = (point_sanit[0], point_sanit[1], self.pos_home[2], self.pos_home[3],
                      self.pos_home[4], self.sim_to_robot_theta(point_sanit[2]))
        elif np.size(points) == 6:
            target = np.array(points, dtype=float).copy()
            target[-1] = self.sim_to_robot_theta(target[-1])
            target = tuple(target)
            # theta_z = self.sim_to_robot_theta(points[-1])
            # target = copy.copy(np.array([points[:-1], theta_z]))
        else:
            logger.info('poisition given is not x, y, theta_z or 6 DOFs')

        # correct for too big a twist
        mid = self.correct_too_big_rot(target)
        while mid is not None:
            # self.move_lin_split(mid)
            self.move_lin(mid)
            mid = self.correct_too_big_rot(target)
        # self.move_lin_split(target)
        self.move_lin(target)

        # save current position as self.current_pos after every movement
        self._get_current_pos()

    def move_lin(self, target):
        # logger.info('Moving the robot - linear')
        robot_helpers.assert_ready(self.robot)
        try:
            self.robot.MoveLin(*target)
            self.robot.WaitIdle()
            # logger.info('Robot finished moving')
        except (mdr.MecademicNonFatalException, mdr.MecademicFatalException, mdr.InterruptException,
                Exception):
            # Robot likely entered error on invalid move
            self._recover_robot()
            self.robot.MoveLin(*target)
            self.robot.WaitIdle()

    def move_lin_split(self, target):
        """
        target = (x, y, z, rx, ry, rz)  # in your units (mm, deg)
        Strategy:
          1) spatial MoveLin with *current* orientation (avoids 180° prot)
          2) rotate-in-place using multiple MoveLin steps < 180°
        """
        logger.info("Moving the robot - linear (split spatial + angular)")
        robot_helpers.assert_ready(self.robot)

        # Current pose and target pose
        cur = tuple(self.robot.GetPose())
        tx, ty, tz, trx, try_, trz = target

        # 1) Spatial move: go to xyz but keep current orientation
        spatial_pose = (tx, ty, tz, cur[3], cur[4], cur[5])

        logger.info(
            f"MoveLin spatial: xyz=({tx:.2f},{ty:.2f},{tz:.2f}) "
            f"ori_hold=({cur[3]:.2f},{cur[4]:.2f},{cur[5]:.2f})"
        )
        self.move_lin(spatial_pose)

        # 2) Rotation-only move(s) at fixed xyz, fixed rx/ry, stepping rz
        # Decide rotation delta relative to current orientation (after spatial move)
        # Read pose again in case controller normalized it.
        cur2 = tuple(self.robot.GetPose())
        start_rz = cur2[5]

        # Use shortest-path delta to requested rz.
        # (If you want "continuous >360", you must feed an unwrapped rz here,
        #  OR maintain your own desired_unwrapped and convert it into a sequence.)
        delta = robot_helpers.shortest_delta_deg(start_rz, trz)

        # If the shortest delta is small, just do final pose once
        if abs(delta) < 1e-6:
            final_pose = (tx, ty, tz, trx, try_, trz)
            logger.info("No rotation needed; finishing with final orientation.")
            self.move_lin(final_pose)
            logger.info("Robot finished moving")
            return

        # Split into safe steps (<180) to avoid MX_ST_BLOCKED_BY_180_DEG_PROT
        steps = robot_helpers.split_rotation(delta, max_step=160.0)

        logger.info(
            f"Rotate in place: start_rz={start_rz:.2f}, target_rz={trz:.2f}, "
            f"delta(shortest)={delta:.2f}, steps={len(steps)}"
        )

        rz_running = start_rz
        for i, dstep in enumerate(steps, start=1):
            rz_running = rz_running + dstep
            rot_pose = (tx, ty, tz, trx, try_, rz_running)
            logger.info(f"Rotate step {i}/{len(steps)}: rz={rz_running:.2f} (d={dstep:.2f})")
            self.move_lin(rot_pose)

        # Ensure we land exactly on requested target pose
        final_pose = (tx, ty, tz, trx, try_, trz)
        logger.info(f"Finalize pose: rz={trz:.2f}")
        self.move_lin(final_pose)

        logger.info("Robot finished moving")

    def _get_current_pos(self) -> None:
        current_pos_6 = self.robot.GetPose()
        theta_z = self.robot_to_sim_theta(current_pos_6[-1])
        self.current_pos = np.array([current_pos_6[0], current_pos_6[1], theta_z])

    def sim_to_robot_theta(self, theta_sim_deg: float) -> float:
        return self.theta_sim_to_robot * float(theta_sim_deg)  # invert CCW->CW

    def robot_to_sim_theta(self, theta_robot_deg: float) -> float:
        return self.theta_sim_to_robot * float(theta_robot_deg)  # invert CW->CCW

    def sanitize_target(self, points3, Sprvsr: "SupervisorClass"):
        x, y, theta_z = map(float, points3)
        x, y = self.clamp_to_circle_xy(x, y, theta_z, Sprvsr)
        return np.array([x, y, theta_z])

    def correct_too_big_rot(self, target):
        # correct for too big a twist
        starting_pos = tuple(self.robot.GetPose())
        delta = np.asarray(starting_pos) - np.asarray(target)
        rot_idx = np.array([3, 4, 5], dtype=int)  # Rotational indices in Mecademic pose: rx, ry, rz
        over = np.abs(delta[rot_idx]) > 180.0
        if np.any(over):
            # Intermediate pose:
            # - x,y,z go directly to target
            # - rotations that are "over" go halfway; others go directly to target
            mid = np.asarray(target)
            for i, is_over in zip(rot_idx, over):
                if is_over:
                    mid[i] = starting_pos[i] + 0.5 * delta[i]
                logger.info('Large rotation detected. Splitting MoveLin into two steps. '
                            f'start={starting_pos} mid={mid} target={target}')
            return mid
        else:
            return None

    def clamp_to_circle_xy(self, x, y, theta_z, Sprvsr: "SupervisorClass", margin=0.0):
        """
        If (x,y) is outside the circle of radius (R-margin), project it to the nearest point on the circle.
        """
        # account for previous total angle to calculate current total angle
        if hasattr(Sprvsr, "total_angle"):
            prev = Sprvsr.total_angle
        else:
            prev = 0.0

        # calculate current total angle
        Sprvsr.total_angle = np.rad2deg(helpers.get_total_angle(self.pos_origin, np.array([x, y]), prev))
        print(f'total_angle inside clamp_to_circle_xy = {Sprvsr.total_angle}')

        # effective radius of chain
        R_eff = helpers.effective_radius(self.R_chain, Sprvsr.L, Sprvsr.total_angle, theta_z)
        print(f'effective Radius inside clamp_to_circle_xy = {R_eff}')

        # R_eff = max(0.0, self.R - margin)
        r_robot = np.hypot(x, y)
        r_chain = np.hypot(x-self.pos_origin[0], y-self.pos_origin[1])

        x2, x3, y2, y3 = None, None, None, None

        if r_chain >= R_eff:
            scale = R_eff / r_robot
            x2 = self.pos_origin[0] + (x-self.pos_origin[0]) * scale
            y2 = self.pos_origin[1] + (y-self.pos_origin[1]) * scale
            print(f'clamped from x={x},y={y} to x={x2},y={y2} due to robot limits')

        elif r_robot >= self.R_robot:
            scale = R_eff / r_robot
            x3 = x * scale
            y3 = y * scale
            print(f'clamped from x={x},y={y} to x={x3},y={y3} due to chain revolusions')

        x_clamp = np.nanmin(np.array([x, x2, x3], dtype=float))
        y_clamp = np.nanmin(np.array([y, y2, y3], dtype=float))

        return float(x_clamp), float(y_clamp)

    def disconnect(self) -> None:
        self.robot.Disconnect()

    def _recover_robot(self) -> None:
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
