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

import robot_helpers, file_helpers, helpers

# Use tool to setup default console and file logger
tools.SetDefaultLogger(logging.INFO, f'{pathlib.Path(__file__).stem}.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class MecaClass:
    def __init__(self, config_path: str = "robot_config.ini"):
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
        limits_path = self.cfg.get("limits", "path")
        pts = file_helpers.load_perimeter_xy(limits_path, x_col="x", y_col="y")
        # allowed radius of motion, and offset origin
        self.cx, self.cy, self.R = robot_helpers.fit_circle_xy(pts)
        print("Circle:", self.cx, self.cy, self.R)

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

    # def move_to_floor(self):
    #     logger.info('Moving the robot to floor')

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

    def move_pos(self, points: NDArray) -> None:
        if np.size(points) == 3:
            point_sanit = self.sanitize_target(points)
            target = (point_sanit[0], point_sanit[1], self.pos_home[2], self.pos_home[3],
                      self.pos_home[4], point_sanit[2])
        elif np.size(points) == 6:
            target = copy.copy(points)
        else:
            logger.info('poisition given is not x, y, theta_z or 6 DOFs')

        # correct for too big a twist
        mid = self.correct_too_big_rot(target)
        while mid is not None:
            self.move_lin(mid)
            mid = self.correct_too_big_rot(target)
        self.move_lin(target)

        # save current position as self.current_pos after every movement
        self._get_current_pos()

    def move_lin(self, target):
        logger.info('Moving the robot - linear')
        robot_helpers.assert_ready(self.robot)
        try:
            self.robot.MoveLin(*target)
            self.robot.WaitIdle()
        except (mdr.MecademicNonFatalException, mdr.MecademicFatalException, mdr.InterruptException,
                Exception):
            # Robot likely entered error on invalid move
            self._recover_robot()
        self.robot.MoveLin(*target)
        self.robot.WaitIdle()
        logger.info('Robot finished moving')

    def _get_current_pos(self) -> None:
        current_pos_6 = self.robot.GetPose()
        self.current_pos = np.array([current_pos_6[0], current_pos_6[1], current_pos_6[-1]])

    def sanitize_target(self, points3):
        # x, y, z, rx, ry, rz = map(float, target6)

        # # TODO: set these to YOUR safe numbers
        # XMIN, XMAX = 40.0, 220.0
        # YMIN, YMAX = -120.0, 120.0
        # ZMIN, ZMAX = 10.0, 220.0

        # x = helpers.clamp(x, XMIN, XMAX)
        # y = helpers.clamp(y, YMIN, YMAX)
        # z = helpers.clamp(z, ZMIN, ZMAX)

        # # keep angles in a reasonable wrapped range
        # rx = helpers.wrap_deg(rx)
        # ry = helpers.wrap_deg(ry)
        # rz = helpers.wrap_deg(rz)
        x, y, rz = map(float, points3)
        x, y = self.clamp_to_circle_xy(x, y)
        return np.array([x, y, rz])

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

    def clamp_to_circle_xy(self, x, y, margin=0.0):
        """
        If (x,y) is outside the circle of radius (R-margin), project it to the nearest point on the circle.
        """
        R_eff = max(0.0, self.R - margin)
        dx = x - self.cx
        dy = y - self.cy
        r = np.hypot(dx, dy)

        if r <= R_eff or r == 0.0:
            return float(x), float(y)  # unchanged

        scale = R_eff / r
        x2 = self.cx + dx * scale
        y2 = self.cy + dy * scale
        print(f'clamped from {x},{y} to {x2}, {y2}')
        return float(x2), float(y2)

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
