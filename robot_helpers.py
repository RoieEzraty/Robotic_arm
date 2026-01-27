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


def wrap_to_180(deg: float) -> float:
    """Wrap angle to (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def shortest_delta_deg(a_from: float, a_to: float) -> float:
    """Signed shortest delta from a_from to a_to in degrees."""
    return wrap_to_180(a_to - a_from)


def split_rotation(delta_deg: float, max_step: float = 160.0):
    """Yield signed steps whose absolute value <= max_step."""
    if max_step <= 0:
        raise ValueError("max_step must be > 0")
    n = int(np.ceil(abs(delta_deg) / max_step)) if abs(delta_deg) > 0 else 0
    if n == 0:
        return []
    step = delta_deg / n
    return [step] * n


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
