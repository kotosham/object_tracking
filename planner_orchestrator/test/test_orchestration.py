"""Unit tests for the pure orchestration helpers (Phase 4)."""
import math

from planner_orchestrator.planner_logic import (
    Action, TURN, DRIVE_FORWARD, DRIVE_TO_VISIBLE, DETECT_ALL, DONE,
)
from planner_orchestrator.orchestration import (
    skill_for_action, is_terminal, relative_goal, wrap_angle, should_launch_lead_replan,
    describe_occupancy_grid, SKILL_GO_TO_POSE, SKILL_APPROACH, SKILL_NONE,
)


def test_skill_mapping():
    assert skill_for_action(TURN) == SKILL_GO_TO_POSE
    assert skill_for_action(DRIVE_FORWARD) == SKILL_GO_TO_POSE
    assert skill_for_action(DRIVE_TO_VISIBLE) == SKILL_APPROACH
    assert skill_for_action(DETECT_ALL) == SKILL_NONE   # orchestrator-local
    assert skill_for_action(DONE) == SKILL_NONE


def test_is_terminal():
    assert is_terminal(DONE)
    assert not is_terminal(TURN) and not is_terminal(DRIVE_TO_VISIBLE)
    assert not is_terminal(DETECT_ALL)


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


def test_describe_occupancy_grid():
    desc = describe_occupancy_grid(40, 40, 0.05, (0.5, -0.5),
                                   n_free=600, n_occupied=200, n_unknown=800)
    assert 'occupancy map' in desc.lower()
    assert '0.50' in desc and '-0.50' in desc        # robot pose, %.2f
    assert '50%' in desc                              # (600+200)/1600 explored
    assert 'white=free' in desc and 'gray=unknown' in desc   # legend for the VLM
