# helpers.py
import logging
import numpy as np

logger = logging.getLogger(__name__)


def apply_motion_config(robot, cfg):
    """
    Apply motion-related hyperparameters to a Mecademic robot instance.
    Safe across mecademicpy versions.
    """

    def getfloat(section, key, default=None):
        return cfg.getfloat(section, key, fallback=default)

    def getint(section, key, default=None):
        return cfg.getint(section, key, fallback=default)

    def call_if_exists(method_name, *args):
        fn = getattr(robot, method_name, None)
        if callable(fn):
            fn(*args)
            logger.info(f"Applied {method_name}{args}")
            return True
        return False

    # Cartesian
    lin_vel = getfloat("motion", "lin_vel")
    if lin_vel is not None:
        call_if_exists("SetCartLinVel", lin_vel)

    lin_acc = getfloat("motion", "lin_acc")
    if lin_acc is not None:
        call_if_exists("SetCartLinAcc", lin_acc)

    # Joint
    jvel = getfloat("motion", "vel")
    if jvel is not None:
        (call_if_exists("SetJointVel", jvel)
         or call_if_exists("SetJointsVel", jvel)
         or call_if_exists("SetJointVelPct", jvel)
         or call_if_exists("SetJointsVelPct", jvel))

    jacc = getfloat("motion", "acc")
    if jacc is not None:
        (call_if_exists("SetJointAcc", jacc)
         or call_if_exists("SetJointsAcc", jacc)
         or call_if_exists("SetJointAccPct", jacc)
         or call_if_exists("SetJointsAccPct", jacc))

    # Path
    blending = getint("motion", "blending")
    if blending is not None:
        call_if_exists("SetBlending", blending)


def assert_ready(robot):
    state = robot.GetStatusRobot()
    if state.error_status:
        raise RuntimeError("Robot is in ERROR state")
    if not state.activation_state:
        raise RuntimeError("Robot not activated")
    if not state.homing_state:
        raise RuntimeError("Robot not homed")


def on_robot_error(error):
    logger.error(
        f"ROBOT ERROR | code={error.error_code} | "
        f"severity={error.severity} | "
        f"message='{error.error_message}'"
    )


def on_robot_state(state):
    if state.in_error:
        logger.error(
            f"ROBOT ENTERED ERROR STATE | "
            f"error_id={state.error_id} | "
            f"description={state.error_description}"
        )


def fit_circle_xy(points_xy: np.ndarray):
    """
    Least-squares circle fit (Kåsa-style).
    points_xy: (N,2) array with columns [x,y]
    returns: (cx, cy, R)
    """
    x = points_xy[:, 0].astype(float)
    y = points_xy[:, 1].astype(float)

    # Solve: x^2 + y^2 + A x + B y + C = 0
    A = np.c_[x, y, np.ones_like(x)]
    b = -(x**2 + y**2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = sol

    cx = -a / 2.0
    cy = -b_ / 2.0
    R = np.sqrt(max(0.0, cx**2 + cy**2 - c))
    return float(cx), float(cy), float(R)
