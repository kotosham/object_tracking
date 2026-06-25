"""Pure orchestration helpers (Phase 4). ROS-free so the atomic-action -> skill
dispatch mapping and the relative-motion geometry are unit-testable.

Each atomic action maps to an executive skill (or is handled by the orchestrator):
  TURN / DRIVE_FORWARD -> GoToPose at a pose computed RELATIVE to the robot's real
                          pose (the relative motion is the action's arg, not a
                          VLM-emitted coordinate)
  DRIVE_TO_VISIBLE     -> ApproachDetection
  DETECT_ALL           -> orchestrator-local detector call (no Pi skill)
  DONE                 -> terminal (no skill)
The executive owns motion + safety; the VLM/orchestrator never touches cmd_vel.
A safe Stop is kept as an executive-owned fallback (see SKILL_STOP) -- it is NOT a
VLM action, so the model can never command a stop, only DONE.
"""
from __future__ import annotations

import math
from typing import Tuple

from planner_orchestrator.planner_logic import (
    Action, TURN, DRIVE_FORWARD, DRIVE_TO_VISIBLE, DETECT_ALL, DONE,
)

SKILL_GO_TO_POSE = 'go_to_pose'
SKILL_APPROACH = 'approach_detection'
SKILL_STOP = 'stop'           # executive-owned safety fallback, not a VLM action
SKILL_NONE = ''

_SKILL_MAP = {
    TURN: SKILL_GO_TO_POSE,
    DRIVE_FORWARD: SKILL_GO_TO_POSE,
    DRIVE_TO_VISIBLE: SKILL_APPROACH,
    DETECT_ALL: SKILL_NONE,       # handled directly by the orchestrator
    DONE: SKILL_NONE,
}


def skill_for_action(kind: int) -> str:
    """Executive skill action name that executes this atomic action ('' = none)."""
    return _SKILL_MAP.get(kind, SKILL_NONE)


def is_terminal(kind: int) -> bool:
    """Only DONE ends the VLM mission (STOP is no longer a VLM action)."""
    return kind == DONE


def should_launch_lead_replan(action_index: int, batch_len: int,
                              async_enabled: bool, already_pending: bool) -> bool:
    """Phase 4.6 anytime policy: launch the NEXT replan while the LAST action of the
    current batch is executing (lead-time), exactly once per batch, and only when
    async replanning is on and no replan is already in flight. Adopting the result
    happens later at the commit-point (batch boundary) so an in-flight action is
    never interrupted (consensus-horizon)."""
    return (async_enabled and not already_pending
            and batch_len > 0 and action_index == batch_len - 1)


def describe_occupancy_grid(width: int, height: int, resolution: float,
                            robot_xy: Tuple[float, float],
                            n_free: int, n_occupied: int, n_unknown: int) -> str:
    """Human description of the SLAM occupancy map attached as the 2nd image, so the
    VLM can read it: what kind of map it is, scale, legend, and where the robot is.
    Pure text (the pixels are rendered ROS-side). explored% = mapped / total cells."""
    total = max(1, width * height)
    explored = 100.0 * (n_free + n_occupied) / total
    span_x = width * resolution
    span_y = height * resolution
    rx, ry = robot_xy
    return (
        'Top-down SLAM occupancy map (built incrementally as the robot drives), '
        'attached as the 2nd image. %dx%d cells at %.3f m/cell (~%.1fm x %.1fm). '
        'Legend: white=free/driveable, black=obstacle, gray=unknown/unexplored. '
        'Red dot=robot, red line=its heading. Map is north-up, metric. '
        'Explored %.0f%% (gray is still unknown). Robot at map (%.2f, %.2f) m.'
        % (width, height, resolution, span_x, span_y, explored, rx, ry))


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
