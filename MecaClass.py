from __future__ import annotations
import configparser
import logging
import pathlib
import numpy as np
import copy

import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools

from numpy.typing import NDArray

import robot_helpers

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
        logger.info("Robot IP: %s", self.ip)
        
        # robot instance
        self.robot = initializer.RobotWithTools()

        # get origin
        self.x_origin = self.cfg.getfloat("position.origin", "x_origin")
        self.y_origin = self.cfg.getfloat("position.origin", "y_origin")
        self.z_origin = self.cfg.getfloat("position.origin", "z_origin")
        self.z_sleep = self.cfg.getfloat("position.origin", "z_sleep")
        self.theta_x_origin = self.cfg.getfloat("position.origin", "theta_x_origin")
        self.theta_y_origin = self.cfg.getfloat("position.origin", "theta_y_origin")
        self.theta_z_origin = self.cfg.getfloat("position.origin", "theta_z_origin")

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

        # home
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
        load_cell_thick = self.cfg.getfloat("position.origin", "load_cell_thick", fallback=None)
        holder_len = self.cfg.getfloat("position.origin", "holder_len", fallback=None)
        self.tip_length = load_cell_thick + holder_len
        self.robot.SetTrf(0.0, 0.0, self.tip_length, 0.0, 0.0, 0.0)

    # def move_to_floor(self):
    #     logger.info('Moving the robot to floor')

    def move_to_origin(self):
        logger.info('Moving the robot to origin')
        current_pos = self.robot.GetPose()
        if current_pos[2] > self.z_origin:
            self.robot.MoveLin(self.x_origin/2, self.y_origin, self.z_sleep,
                               self.theta_x_origin, self.theta_y_origin, self.theta_z_origin)
            self.robot.WaitIdle()
        else:
            pass
        self.robot.MoveLin(self.x_origin, self.y_origin, self.z_origin,
                           self.theta_x_origin, self.theta_y_origin, self.theta_z_origin)

    def move_to_sleep_pos(self) -> None:
        logger.info('Moving the robot to sleep position')
        sleep_pos = (40, 0, 230, 180, 0, 0)
        self.robot.MoveLin(*sleep_pos)

    def move_joints(self, joints) -> None:
        logger.info('Moving the robot - joints')
        robot_helpers.assert_ready(self.robot)
        self.robot.MoveJoints(*joints)
        # self.robot.WaitIdle()
        # logger.info('Robot done moving')

    def move_lin(self, points: NDArray) -> None:
        logger.info('Moving the robot - linear')
        starting_pos = tuple(self.robot.GetPose())
        robot_helpers.assert_ready(self.robot)
        if np.size(points) == 3:
            target = (points[0], points[1], self.z_origin, self.theta_x_origin,
                      self.theta_y_origin, points[2])
        elif np.size(points) == 6:
            target = copy.copy(points)
        else:
            logger.info('poisition given is not x, y, theta_z or 6 DOFs')
        delta = np.asarray(starting_pos) - np.asarray(target)
        # Rotational indices in Mecademic pose: rx, ry, rz
        rot_idx = np.array([3, 4, 5], dtype=int)
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
            self.robot.MoveLin(*mid)
            self.robot.WaitIdle()

        self.robot.MoveLin(*target)
        self.robot.WaitIdle()
        logger.info('Robot finished moving')

    def move_lin_allDOFs(self, target: tuple) -> None:
        logger.info('Moving the robot - linear')
        starting_pos = tuple(self.robot.GetPose())
        robot_helpers.assert_ready(self.robot)
        delta = np.asarray(starting_pos) - np.asarray(target)
        # Rotational indices in Mecademic pose: rx, ry, rz
        rot_idx = np.array([3, 4, 5], dtype=int)
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
            self.robot.MoveLin(*mid)
            self.robot.WaitIdle()

        self.robot.MoveLin(*target)
        self.robot.WaitIdle()
        logger.info('Robot finished moving')

    def disconnect(self) -> None:
        self.robot.Disconnect()

    def recover_robot(self) -> None:
        logger.warning("Recovering robot from fault...")

        self.robot.ResetError()
        self.robot.WaitIdle()

        self.robot.DeactivateRobot()
        self.robot.WaitDeactivated()

        self.robot.ActivateRobot()
        self.robot.WaitActivated()

        self.robot.Home()
        self.robot.WaitHomed()

        logger.info("Robot recovered and ready")
