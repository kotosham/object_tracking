"""Unit tests for the pure orchestration helpers (Phase 4)."""
import math

from planner_orchestrator.planner_logic import (
    Action, TURN, DRIVE_FORWARD, DRIVE_TO_VISIBLE, DETECT_ALL, DONE,
)
from planner_orchestrator.orchestration import (
    skill_for_action, is_terminal, relative_goal, wrap_angle, should_launch_lead_replan,
    describe_occupancy_grid, forward_clearance,
    SKILL_GO_TO_POSE, SKILL_APPROACH, SKILL_NONE,
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


def _scan(ranges, fov=math.radians(62.4)):
    """(ranges, angle_min, angle_increment) for a symmetric camera-like scan."""
    n = len(ranges)
    inc = fov / max(1, n - 1)
    return ranges, -fov / 2.0, inc


def test_forward_clearance_wall_dead_ahead():
    # 21 rays, wall at 1.0 m: central rays block at ~1.0, oblique ones farther
    ranges, a0, inc = _scan([1.0 / math.cos(-math.radians(31.2) + i * math.radians(62.4) / 20)
                             for i in range(21)])
    c = forward_clearance(ranges, a0, inc, 0.25)
    assert c is not None and math.isclose(c, 1.0, abs_tol=0.02)


def test_forward_clearance_doorframe_outside_corridor_ignored():
    # hit 0.8 m off-axis at 2 m depth: outside the 0.25 m body corridor -> free
    theta = math.atan2(0.8, 2.0)
    r = math.hypot(0.8, 2.0)
    c = forward_clearance([r], theta, 1.0, 0.25)
    assert c is None
    # the SAME hit with a wider corridor (0.9) becomes a real block at depth 2.0
    c2 = forward_clearance([r], theta, 1.0, 0.9)
    assert c2 is not None and math.isclose(c2, 2.0, abs_tol=1e-6)


def test_forward_clearance_near_edge_inside_corridor_blocks():
    # a wall edge 0.2 m off-axis at 0.4 m depth is INSIDE the body sweep: a
    # fixed 12-degree cone would miss it (bearing ~27 deg), the corridor must not
    theta = math.atan2(0.2, 0.4)
    r = math.hypot(0.2, 0.4)
    c = forward_clearance([r], theta, 1.0, 0.25)
    assert c is not None and math.isclose(c, 0.4, abs_tol=1e-6)


def test_forward_clearance_rear_and_invalid_rays_never_block():
    nan = float('nan')
    # rear hit (cos<0), NaN, inf, zero -> no information about the forward corridor
    ranges, a0, inc = _scan([nan, float('inf'), 0.0])
    assert forward_clearance(ranges, a0, inc, 0.25) is None
    assert forward_clearance([1.0], math.pi, 1.0, 0.25) is None      # behind
    assert forward_clearance([], 0.0, 0.0, 0.25) is None             # empty scan


def test_describe_occupancy_grid():
    desc = describe_occupancy_grid(40, 40, 0.05, (0.5, -0.5),
                                   n_free=600, n_occupied=200, n_unknown=800)
    assert 'occupancy map' in desc.lower()
    assert '0.50' in desc and '-0.50' in desc        # robot pose, %.2f
    assert '50%' in desc                              # (600+200)/1600 explored
    assert 'white=free' in desc and 'gray=unknown' in desc   # legend for the VLM
