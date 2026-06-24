"""Unit tests for the pure orchestration helpers (Phase 4)."""
import math

from planner_orchestrator.planner_logic import (
    Action, TURN, DRIVE_FORWARD, DRIVE_TO_VISIBLE, GO_TO_FRONTIER, GET_OBSERVATION,
    DONE, STOP,
)
from planner_orchestrator.orchestration import (
    skill_for_action, is_terminal, relative_goal, wrap_angle, should_launch_lead_replan,
    SKILL_GO_TO_POSE, SKILL_APPROACH, SKILL_EXPLORE, SKILL_GET_OBS, SKILL_STOP, SKILL_NONE,
)


def test_skill_mapping():
    assert skill_for_action(TURN) == SKILL_GO_TO_POSE
    assert skill_for_action(DRIVE_FORWARD) == SKILL_GO_TO_POSE
    assert skill_for_action(DRIVE_TO_VISIBLE) == SKILL_APPROACH
    assert skill_for_action(GO_TO_FRONTIER) == SKILL_EXPLORE
    assert skill_for_action(GET_OBSERVATION) == SKILL_GET_OBS
    assert skill_for_action(STOP) == SKILL_STOP
    assert skill_for_action(DONE) == SKILL_NONE


def test_is_terminal():
    assert is_terminal(DONE) and is_terminal(STOP)
    assert not is_terminal(TURN) and not is_terminal(GO_TO_FRONTIER)


def test_turn_rotates_in_place():
    gx, gy, gyaw = relative_goal(1.0, 2.0, 0.0, Action(TURN, turn_yaw_rad=math.pi / 2))
    assert math.isclose(gx, 1.0) and math.isclose(gy, 2.0)
    assert math.isclose(gyaw, math.pi / 2, abs_tol=1e-6)


def test_turn_wraps_angle():
    _, _, gyaw = relative_goal(0, 0, math.pi * 0.9, Action(TURN, turn_yaw_rad=math.pi * 0.5))
    assert -math.pi <= gyaw <= math.pi   # wrapped


def test_drive_forward_along_heading():
    gx, gy, gyaw = relative_goal(0.0, 0.0, 0.0, Action(DRIVE_FORWARD, forward_dist_m=1.0))
    assert math.isclose(gx, 1.0, abs_tol=1e-6) and math.isclose(gy, 0.0, abs_tol=1e-6)
    gx2, gy2, _ = relative_goal(0.0, 0.0, math.pi / 2, Action(DRIVE_FORWARD, forward_dist_m=2.0))
    assert math.isclose(gx2, 0.0, abs_tol=1e-6) and math.isclose(gy2, 2.0, abs_tol=1e-6)


def test_wrap_angle():
    assert math.isclose(wrap_angle(3 * math.pi), math.pi, abs_tol=1e-6) or \
           math.isclose(wrap_angle(3 * math.pi), -math.pi, abs_tol=1e-6)


def test_lead_replan_fires_on_last_action_of_batch():
    # batch of 3: launch only at the last action (index 2), once
    assert not should_launch_lead_replan(0, 3, True, False)
    assert not should_launch_lead_replan(1, 3, True, False)
    assert should_launch_lead_replan(2, 3, True, False)


def test_lead_replan_single_action_batch_fires_immediately():
    # real-VLM batch is 1 action -> overlaps every step
    assert should_launch_lead_replan(0, 1, True, False)


def test_lead_replan_suppressed_when_pending_or_disabled():
    assert not should_launch_lead_replan(2, 3, True, True)    # already pending
    assert not should_launch_lead_replan(2, 3, False, False)  # async off
    assert not should_launch_lead_replan(0, 0, True, False)   # empty batch
