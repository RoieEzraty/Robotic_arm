from __future__ import annotations
import configparser
import logging
import pathlib
import numpy as np

import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools

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

    def home(self):
        logger.info("Homing")
        # robot_helpers.assert_ready(self.robot)
        self.robot.Home()
        self.robot.WaitHomed()
        logger.info("Robot at home")

    def move_to_origin(self):
        logger.info('Moving the robot to origin')
        self.move_lin(self, (self.x_origin, self.y_origin, self.z_origin,
                             self.theta_x_origin, self.theta_y_origin, self.theta_z_origin))

    def move_joints(self, joints):
        logger.info('Moving the robot - joints')
        robot_helpers.assert_ready(self.robot)
        self.robot.MoveJoints(*joints)
        # self.robot.WaitIdle()
        # logger.info('Robot done moving')

    def move_lin(self, x_y_theta):
        logger.info('Moving the robot - linear')
        starting_pos = tuple(self.robot.GetPose())
        # robot_helpers.assert_ready(self.robot)
        # self.robot.MoveJoints(*starting_pos)  # Move joints using the unrolled array as arguments
        robot_helpers.assert_ready(self.robot)
        # points = starting_pos
        points = (x_y_theta[0], x_y_theta[1], self.z_origin, self.theta_x_origin,  self.theta_y_origin,
                 x_y_theta[2])
        print(points)
        self.robot.MoveLin(*points)
        # self.robot.WaitIdle()
        # logger.info('Robot finished moving')

    def disconnect(self):
        self.robot.Disconnect()

    def recover_robot(self):
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

    # def _apply_motion_config(self):
    #     # Helpers
    #     def getfloat(section, key, default=None):
    #         return self.cfg.getfloat(section, key, fallback=default)

    #     def getint(section, key, default=None):
    #         return self.cfg.getint(section, key, fallback=default)

    #     def call_if_exists(method_name, *args):
    #         fn = getattr(self.robot, method_name, None)
    #         if callable(fn):
    #             fn(*args)
    #             logger.info(f"Applied {method_name}{args}")
    #             return True
    #         return False

    #     # Cartesian
    #     lin_vel = getfloat("motion.cartesian", "lin_vel", None)
    #     if lin_vel is not None:
    #         call_if_exists("SetCartLinVel", lin_vel)

    #     ang_vel = getfloat("motion.cartesian", "ang_vel", None)
    #     if ang_vel is not None:
    #         call_if_exists("SetCartAngVel", ang_vel)

    #     lin_acc = getfloat("motion.cartesian", "lin_acc", None)
    #     if lin_acc is not None:
    #         call_if_exists("SetCartLinAcc", lin_acc)

    #         ang_acc = getfloat("motion.cartesian", "ang_acc", None)
    #     if ang_acc is not None:
    #         call_if_exists("SetCartAngAcc", ang_acc)

    #     # Joint (API names vary by mecademicpy version, so try a few)
    #     jvel = getfloat("motion.joint", "vel", None)
    #     if jvel is not None:
    #         (call_if_exists("SetJointVel", jvel)
    #          or call_if_exists("SetJointsVel", jvel)
    #          or call_if_exists("SetJointVelPct", jvel)
    #          or call_if_exists("SetJointsVelPct", jvel))

    #     jacc = getfloat("motion.joint", "acc", None)
    #     if jacc is not None:
    #         (call_if_exists("SetJointAcc", jacc)
    #          or call_if_exists("SetJointsAcc", jacc)
    #          or call_if_exists("SetJointAccPct", jacc)
    #          or call_if_exists("SetJointsAccPct", jacc))

    #     # Path / blending
    #     blending = getint("motion.path", "blending", None)
    #     if blending is not None:
    #         call_if_exists("SetBlending", blending)

    #     logger.info("Motion config applied.")
