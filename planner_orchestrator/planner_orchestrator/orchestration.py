"""Pure orchestration helpers (Phase 4). ROS-free so the atomic-action -> skill
dispatch mapping and the relative-motion geometry are unit-testable.

Each atomic action is executed by an EXISTING FLAT skill on the Pi executive:
  TURN / DRIVE_FORWARD -> GoToPose at a pose computed RELATIVE to the robot's real
                          pose (the relative motion is the action's arg, not a
                          VLM-emitted coordinate)
  DRIVE_TO_VISIBLE     -> ApproachDetection
  GO_TO_FRONTIER       -> ExploreFrontier(id)
  GET_OBSERVATION      -> GetObservation
  STOP                 -> Stop
  DONE                 -> terminal (no skill)
The executive owns motion + safety; the VLM/orchestrator never touches cmd_vel.
"""
from __future__ import annotations

import math
from typing import Tuple

from planner_orchestrator.planner_logic import (
    Action, TURN, DRIVE_FORWARD, DRIVE_TO_VISIBLE, GO_TO_FRONTIER,
    GET_OBSERVATION, DONE, STOP,
)

SKILL_GO_TO_POSE = 'go_to_pose'
SKILL_EXPLORE = 'explore_frontier'
SKILL_APPROACH = 'approach_detection'
SKILL_GET_OBS = 'get_observation'
SKILL_STOP = 'stop'
SKILL_NONE = ''

_SKILL_MAP = {
    TURN: SKILL_GO_TO_POSE,
    DRIVE_FORWARD: SKILL_GO_TO_POSE,
    DRIVE_TO_VISIBLE: SKILL_APPROACH,
    GO_TO_FRONTIER: SKILL_EXPLORE,
    GET_OBSERVATION: SKILL_GET_OBS,
    STOP: SKILL_STOP,
    DONE: SKILL_NONE,
}


def skill_for_action(kind: int) -> str:
    """Executive skill action name that executes this atomic action ('' = none)."""
    return _SKILL_MAP.get(kind, SKILL_NONE)


def is_terminal(kind: int) -> bool:
    """DONE/STOP end the VLM mission."""
    return kind in (DONE, STOP)


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def relative_goal(x: float, y: float, yaw: float, action: Action) -> Tuple[float, float, float]:
    """Absolute (gx, gy, gyaw) for a relative TURN/DRIVE_FORWARD from the robot's
    real pose. TURN rotates in place; DRIVE_FORWARD advances along the heading.
    Any other action returns the current pose (caller won't use it)."""
    if action.kind == TURN:
        return (x, y, wrap_angle(yaw + action.turn_yaw_rad))
    if action.kind == DRIVE_FORWARD:
        d = action.forward_dist_m
        return (x + d * math.cos(yaw), y + d * math.sin(yaw), yaw)
    return (x, y, yaw)
