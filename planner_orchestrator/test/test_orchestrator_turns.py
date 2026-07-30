"""Focused tests for orchestrator-side motion guards."""

from planner_orchestrator.orchestrator_node import PlannerOrchestrator
from geometry_msgs.msg import PointStamped, PoseStamped

from planner_orchestrator.planner_logic import (
    Action, Candidate, ContextMark, NotesBuffer, Observation, DRIVE_FORWARD,
    DRIVE_TO_VISIBLE, TURN,
)


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


def _bare_orchestrator():
    node = object.__new__(PlannerOrchestrator)
    node.semantic_turn_antioscillation = True
    node.semantic_turn_max_streak = 2
    node.semantic_probe_forward_m = 0.45
    node.turn_step = 0.6
    node.min_effective_turn_rad = 0.6
    node.turn_settle_s = 2.0
    node.async_replan = True
    node.initial_scan_when_target_absent = True
    node.initial_scan_left_rad = 3.14
    node.initial_scan_right_rad = 1.57
    node._semantic_turn_side = 'left'
    node._semantic_turn_streak = 2
    node.locked_target_approach_max_attempts = 8
    node._target_nav_lock = None
    node._corridor_scan = {}
    node.notes = NotesBuffer()
    node.get_logger = lambda: _Logger()
    node._activity = lambda *args, **kwargs: None
    return node


def _nav_lock(target='office chair'):
    pt = PointStamped()
    pt.header.frame_id = 'map'
    pt.point.x = 3.0
    pt.point.y = 1.0
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x = 2.5
    pose.pose.position.y = 0.8
    return {
        'target': target,
        'label': target,
        'target_point': pt,
        'final_goal_pose': pose,
        'attempts': 0,
    }


def test_antioscillation_probes_forward_even_with_close_dino_context():
    node = _bare_orchestrator()
    obs = Observation(
        target='office chair',
        context_marks=[
            ContextMark(1, 'desk table', 0.41, distance_m=0.34,
                        side='center', center_x_norm=0.5,
                        relevance='office_context'),
        ],
    )

    action = Action(TURN, turn_yaw_rad=0.6, rationale='inspect left context')
    repaired = node._semantic_explore_antioscillation(
        action, obs, role='semantic_explore')

    assert repaired.kind == DRIVE_FORWARD
    assert repaired.forward_dist_m == node.semantic_probe_forward_m
    assert 'anti_oscillation' in repaired.rationale
    assert 'navigation costmaps decide' in repaired.rationale


def test_antioscillation_advances_after_one_context_turn_when_clear():
    node = _bare_orchestrator()
    node.semantic_turn_max_streak = 1
    obs = Observation(
        target='office chair',
        context_marks=[
            ContextMark(1, 'drawer cabinet', 0.41, distance_m=1.4,
                        side='left', center_x_norm=0.25,
                        relevance='office_context'),
        ],
    )

    action = Action(TURN, turn_yaw_rad=0.6, rationale='inspect left context')
    repaired = node._semantic_explore_antioscillation(
        action, obs, role='semantic_explore')

    assert repaired.kind == DRIVE_FORWARD
    assert repaired.forward_dist_m == node.semantic_probe_forward_m
    assert 'probe forward after inspecting context' in repaired.rationale


def test_initial_scan_sweeps_right_then_left_when_strict_target_absent():
    node = _bare_orchestrator()
    obs = Observation(target='office chair')

    right_actions = node._initial_scan_actions(obs, 0)
    left_actions = node._initial_scan_actions(obs, 1)

    assert len(right_actions) == 1
    assert len(left_actions) == 2
    right = right_actions[0]
    left_return, left_inspect = left_actions
    assert right.kind == TURN and right.turn_yaw_rad == -1.57
    assert left_return.kind == TURN and left_return.turn_yaw_rad == 1.57
    assert left_inspect.kind == TURN and left_inspect.turn_yaw_rad == 1.57
    assert node._initial_scan_actions(obs, 2) == []
    assert node._action_role(left_return, obs) == 'initial_scan'
    assert node._action_role(left_inspect, obs) == 'initial_scan'
    assert node._action_role(right, obs) == 'initial_scan'


def test_initial_scan_ignores_context_promoted_but_stops_for_strict_target():
    node = _bare_orchestrator()
    promoted = Observation(
        target='office chair',
        candidates=[Candidate(3, 'office chair', 0.44, distance_m=2.0,
                              source='context_promoted')],
    )
    strict = Observation(
        target='office chair',
        candidates=[Candidate(1, 'office chair', 0.7, distance_m=2.0,
                              source='target')],
    )

    assert node._initial_scan_actions(promoted, 0)
    assert node._initial_scan_actions(strict, 0) == []


def test_corridor_scan_records_initial_views_as_context_cues():
    node = _bare_orchestrator()
    node._record_corridor_scan(
        'office chair', 1, [],
        [ContextMark(3, 'desk', 0.52, distance_m=2.4,
                     side='right', relevance='office_context')])

    scan = node._corridor_scan_options()
    assert len(scan) == 1
    assert scan[0]['view'] == 'right'
    assert scan[0]['objects'][0]['label'] == 'desk'
    assert 'CORRIDOR_SCAN[right]' in node.notes.summary()
    assert 'not approach' in scan[0]['summary']


def test_turn_settle_suppresses_async_replan_during_turn():
    node = _bare_orchestrator()

    assert not node._should_launch_lead_replan(
        Action(TURN, turn_yaw_rad=1.57), action_index=0,
        batch_len=1, already_pending=False)
    assert node._should_launch_lead_replan(
        Action(DRIVE_FORWARD, forward_dist_m=0.5), action_index=0,
        batch_len=1, already_pending=False)


def test_nav_lock_continues_saved_target_when_target_drops_from_frame():
    node = _bare_orchestrator()
    node._target_nav_lock = _nav_lock()
    obs = Observation(target='office chair')

    action = node._locked_target_action(obs, 'office chair', step_index=3)

    assert action.kind == DRIVE_TO_VISIBLE
    assert action.mark_id == 0
    assert action.arg_label == '__locked_target__'
    assert 'saved point' in action.rationale


def test_nav_lock_updates_from_strict_visible_target_without_vlm():
    node = _bare_orchestrator()
    node._target_nav_lock = _nav_lock()
    obs = Observation(
        target='office chair',
        candidates=[Candidate(7, 'office chair', 0.7, distance_m=1.6,
                              source='target')],
    )

    action = node._locked_target_action(obs, 'office chair', step_index=3)

    assert action.kind == DRIVE_TO_VISIBLE
    assert action.mark_id == 7
    assert 'update the locked map point' in action.rationale
