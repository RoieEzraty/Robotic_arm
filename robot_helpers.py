from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply_motion_config(robot: Any, CFG: Any) -> None:
    """Apply motion-related configuration values to a Mecademic robot instance.
    Used in MecaClass.connect()
    """

    def getfloat(key: str, default: float | None = None) -> float | None:
        value = getattr(CFG.Variabs, key, default)
        return None if value is None else float(value)

    def getint(key: str, default: int | None = None) -> int | None:
        value = getattr(CFG.Variabs, key, default)
        return None if value is None else int(value)

    def call_if_exists(method_name: str, *args: object) -> bool:
        fn = getattr(robot, method_name, None)
        if callable(fn):
            fn(*args)
            logger.info("Applied %s%s", method_name, args)
            return True
        return False

    # Cartesian coords
    lin_vel = getfloat("lin_vel")  # linear velocity
    if lin_vel is not None:
        call_if_exists("SetCartLinVel", lin_vel)

    lin_acc = getfloat("lin_acc")  # linear acceleration
    if lin_acc is not None:
        call_if_exists("SetCartLinAcc", lin_acc)

    # Joints
    jvel = getfloat("vel")  # joint velocity
    if jvel is not None:
        (call_if_exists("SetJointVel", jvel)
         or call_if_exists("SetJointsVel", jvel)
         or call_if_exists("SetJointVelPct", jvel)
         or call_if_exists("SetJointsVelPct", jvel))

    jacc = getfloat("acc")  # joint acceleration
    if jacc is not None:
        (call_if_exists("SetJointAcc", jacc)
         or call_if_exists("SetJointsAcc", jacc)
         or call_if_exists("SetJointAccPct", jacc)
         or call_if_exists("SetJointsAccPct", jacc))

    # Mecademic blending between multiple consecutive motions
    blending = getint("blending")
    if blending is not None:
        call_if_exists("SetBlending", blending)


def assert_ready(robot: Any) -> None:
    """Raise if the robot is not activated, not homed, or currently in error."""
    state = robot.GetStatusRobot()

    if state.error_status:
        raise RuntimeError("Robot is in ERROR state")
    if not state.activation_state:
        raise RuntimeError("Robot not activated")
    if not state.homing_state:
        raise RuntimeError("Robot not homed")


# -----------------------------------
# NOT IN USE
# -----------------------------------
# def wrap_to_180(deg: float) -> float:
#     """Wrap angle in degrees to the interval ``[-180, 180)``.
#     used in shortest_delta_deg()"""
#     return float((deg + 180.0) % 360.0 - 180.0)


# def shortest_delta_deg(a_from: float, a_to: float) -> float:
#     """Return the signed shortest angular change from ``a_from`` to ``a_to``.
#     Used in 

#     Parameters
#     ----------
#     a_from : float
#         Initial angle [deg].
#     a_to : float
#         Target angle [deg].

#     Returns
#     -------
#     float
#         Signed shortest delta angle [deg], wrapped to ``[-180, 180)``.
#     """
#     return wrap_to_180(a_to - a_from)


# def split_rotation(delta_deg: float, max_step: float = 160.0) -> list[float]:
#     """Split a rotation into signed steps whose magnitude does not exceed ``max_step``.

#     Parameters
#     ----------
#     delta_deg : float
#         Total signed rotation to execute [deg].
#     max_step : float, optional
#         Maximal allowed absolute step size [deg].

#     Returns
#     -------
#     list[float]
#         List of equal signed angular increments whose sum is ``delta_deg``.

#     Raises
#     ------
#     ValueError
#         If ``max_step <= 0``.
#     """
#     if max_step <= 0:
#         raise ValueError("max_step must be > 0")

#     n = int(np.ceil(abs(delta_deg) / max_step)) if abs(delta_deg) > 0 else 0
#     if n == 0:
#         return []

#     step = delta_deg / n
#     return [float(step)] * n


# def on_robot_error(error: Any) -> None:
#     """Log a robot error callback payload.

#     Parameters
#     ----------
#     error : object
#         Error-like callback payload exposing ``error_code``, ``severity``,
#         and ``error_message``.
#     """
#     logger.error(
#         "ROBOT ERROR | code=%s | severity=%s | message='%s'",
#         error.error_code,
#         error.severity,
#         error.error_message,
#     )


# def on_robot_state(state: Any) -> None:
#     """Log entry into a robot error state.

#     Parameters
#     ----------
#     state : object
#         State-like callback payload exposing ``in_error``, ``error_id``, and
#         ``error_description``.
#     """
#     if state.in_error:
#         logger.error(
#             "ROBOT ENTERED ERROR STATE | error_id=%s | description=%s",
#             state.error_id,
#             state.error_description,
#         )
