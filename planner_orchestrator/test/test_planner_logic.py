"""Unit tests for the VLM planner pure logic (Phase 4)."""
from planner_orchestrator.planner_logic import (
    Action, Candidate, CircuitBreaker, ContextMark, DegradationLatch, MockPlanner,
    NotesBuffer, Observation, ReplanScheduler, build_vlm_options, parse_vlm_action,
    context_mark_promotable_to_target, validate_action, DRIVE_TO_VISIBLE,
    DETECT_ALL, DRIVE_FORWARD, TURN, DONE,
)


# ---- MockPlanner policy ----------------------------------------------------

def test_mock_approaches_visible_target():
    obs = Observation(target='bus',
                      candidates=[Candidate(2, 'bus', 0.9, distance_m=2.0),
                                  Candidate(5, 'person', 0.8, distance_m=1.0)])
    a = MockPlanner().plan(obs)
    assert a.kind == DRIVE_TO_VISIBLE and a.mark_id == 2


def test_mock_picks_best_matching_candidate():
    obs = Observation(target='chair',
                      candidates=[Candidate(1, 'chair', 0.4, distance_m=2.5),
                                  Candidate(3, 'office chair', 0.9, distance_m=3.0)])
    a = MockPlanner().plan(obs)
    assert a.kind == DRIVE_TO_VISIBLE and a.mark_id == 3   # best score among matches


def test_target_like_context_can_be_promoted_to_target_candidate():
    mark = ContextMark(2, 'office chair', 0.41, distance_m=1.9,
                       side='right', relevance='target_like')
    assert context_mark_promotable_to_target('chair', mark, min_score=0.35)


def test_weak_or_non_target_context_is_not_promoted():
    weak = ContextMark(9, 'chair', 0.26, distance_m=1.7,
                       side='center', relevance='target_like')
    desk = ContextMark(3, 'desk', 0.50, distance_m=1.5,
                       side='left', relevance='office_context')
    assert not context_mark_promotable_to_target('chair', weak, min_score=0.35)
    assert not context_mark_promotable_to_target('chair', desk, min_score=0.35)


def test_mock_detects_all_then_scans_then_done():
    obs = Observation(target='bus')  # nothing visible
    mp = MockPlanner(scan_turn_limit=2, turn_step_rad=0.5)
    assert mp.plan(obs).kind == DETECT_ALL   # one wide look first
    assert mp.plan(obs).kind == TURN
    assert mp.plan(obs).kind == TURN
    assert mp.plan(obs).kind == DONE         # exhausted


def test_mock_semantic_explore_turns_toward_context_side():
    obs = Observation(
        target='office chair',
        context_marks=[ContextMark(4, 'desk', 0.8, distance_m=2.0,
                                   side='left', center_x_norm=0.2,
                                   relevance='office_context')])
    a = MockPlanner(turn_step_rad=0.5).plan(obs)
    assert a.kind == TURN and a.turn_yaw_rad == 0.5
    assert 'semantic_explore' in a.rationale


def test_mock_semantic_explore_drives_to_center_context():
    obs = Observation(
        target='office chair',
        context_marks=[ContextMark(4, 'desk', 0.8, distance_m=2.0,
                                   side='center', center_x_norm=0.5,
                                   relevance='office_context')])
    a = MockPlanner(semantic_forward_m=0.3).plan(obs)
    assert a.kind == DRIVE_FORWARD and a.forward_dist_m == 0.3


def test_mock_semantic_explore_turns_when_center_context_is_blocked():
    obs = Observation(
        target='office chair',
        context_marks=[
            ContextMark(1, 'desk table', 0.41, distance_m=0.31,
                        side='center', center_x_norm=0.5,
                        relevance='office_context'),
            ContextMark(9, 'chair', 0.26, distance_m=1.69,
                        side='center', center_x_norm=0.5,
                        relevance='target_like'),
        ])
    a = MockPlanner(turn_step_rad=0.6).plan(obs)
    assert a.kind == TURN and a.turn_yaw_rad == 0.3
    assert 'forward probe blocked' in a.rationale


def test_mock_done_when_target_close():
    # a known distance within the reached threshold -> arrived, no need to drive.
    mp = MockPlanner(reached_dist_m=0.8)
    close = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=0.5)])
    assert mp.plan(close).kind == DONE


def test_mock_approaches_until_close():
    mp = MockPlanner(reached_dist_m=0.8)
    far = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=1.5)])
    assert mp.plan(far).kind == DRIVE_TO_VISIBLE      # still far -> keep driving
    near = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=0.5)])
    assert mp.plan(near).kind == DONE                 # now close -> arrived


def test_mock_done_when_lost_after_approach():
    mp = MockPlanner()
    seen = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=1.5)])
    assert mp.plan(seen).kind == DRIVE_TO_VISIBLE     # drove toward it
    # at point-blank the bus overflows the frame and YOLOE drops it -> treated as arrived.
    assert mp.plan(Observation(target='bus')).kind == DONE


def test_mock_does_not_done_when_target_lost_after_far_bounded_approach():
    mp = MockPlanner()
    far = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=5.3)])
    assert mp.plan(far).kind == DRIVE_TO_VISIBLE
    # A far target disappearing after a bounded step is not arrival; observe again.
    assert mp.plan(Observation(target='bus')).kind == DETECT_ALL


def test_mock_does_not_approach_unknown_depth_target_before_drive():
    mp = MockPlanner()
    unknown = Observation(target='bus',
                          candidates=[Candidate(2, 'bus', 0.9, side='center')])
    a = mp.plan(unknown)
    assert a.kind == DRIVE_FORWARD
    assert a.forward_dist_m == 0.6
    assert 'depth unknown' in a.rationale


def test_mock_turns_toward_unknown_depth_target_side():
    mp = MockPlanner()
    unknown = Observation(target='bus',
                          candidates=[Candidate(2, 'bus', 0.9, side='right')])
    a = mp.plan(unknown)
    assert a.kind == TURN and a.turn_yaw_rad < 0.0


def test_mock_approach_is_bounded():
    mp = MockPlanner(reached_dist_m=0.8, max_approaches=3)
    far = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=2.0)])
    kinds = [mp.plan(far).kind for _ in range(4)]
    assert kinds == [DRIVE_TO_VISIBLE, DRIVE_TO_VISIBLE, DRIVE_TO_VISIBLE, DONE]  # bounded


# ---- enum-tool-call validation (anti-hallucination) ------------------------

def test_validate_rejects_phantom_mark():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', distance_m=2.0)])
    ok, _ = validate_action(Action(DRIVE_TO_VISIBLE, mark_id=99), obs)
    assert not ok


def test_validate_accepts_real_mark():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', distance_m=2.0)])
    ok, _ = validate_action(Action(DRIVE_TO_VISIBLE, mark_id=2), obs)
    assert ok


def test_validate_rejects_unknown_depth_mark():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus')])
    ok, reason = validate_action(Action(DRIVE_TO_VISIBLE, mark_id=2), obs)
    assert not ok and 'unknown distance' in reason


def test_parse_repairs_unknown_depth_target_to_probe():
    obs = Observation(target='bus',
                      candidates=[Candidate(2, 'bus', distance_m=0.0,
                                            side='left', center_x_norm=0.2)])
    a, reason = parse_vlm_action({'action': 'DRIVE_TO_VISIBLE', 'mark_id': 2}, obs)
    assert reason == 'OK'
    assert a.kind == TURN and a.turn_yaw_rad > 0.0
    assert 'target_probe' in a.rationale


def test_parse_accepts_target_probe_pseudo_action():
    obs = Observation(target='office chair',
                      candidates=[Candidate(1, 'office chair', distance_m=0.0,
                                            side='center', center_x_norm=0.5)])
    a, reason = parse_vlm_action({'action': 'TARGET_PROBE', 'mark_id': 1}, obs)
    assert reason == 'OK'
    assert a.kind == DRIVE_FORWARD
    assert a.forward_dist_m == 0.6
    assert 'target_probe' in a.rationale


def test_parse_repairs_context_forward_to_directional_turn():
    obs = Observation(
        target='office chair',
        context_marks=[
            ContextMark(1, 'drawer cabinet', 0.40, distance_m=2.2,
                        side='center', center_x_norm=0.5,
                        relevance='office_context'),
            ContextMark(8, 'office chair', 0.30, distance_m=3.3,
                        side='right', center_x_norm=0.8,
                        relevance='target_like'),
        ])
    a, reason = parse_vlm_action(
        {'action': 'DRIVE_FORWARD', 'forward_dist_m': 0.5,
         'rationale': 'probe forward'},
        obs)
    assert reason == 'OK'
    assert a.kind == TURN and a.turn_yaw_rad < 0.0
    assert 'replacing forward probe' in a.rationale


def test_parse_normalizes_tiny_context_turn_to_directional_turn():
    obs = Observation(
        target='chair',
        context_marks=[
            ContextMark(1, 'drawer cabinet', 0.38, distance_m=2.4,
                        side='center', center_x_norm=0.5,
                        relevance='office_context'),
            ContextMark(6, 'desk', 0.27, distance_m=2.0,
                        side='right', center_x_norm=0.8,
                        relevance='office_context'),
        ])
    a, reason = parse_vlm_action(
        {'action': 'TURN', 'turn_yaw_rad': 0.17,
         'rationale': 'turn slightly right toward office context'},
        obs)
    assert reason == 'OK'
    assert a.kind == TURN and a.turn_yaw_rad == -0.6
    assert 'normalizing turn toward context mark' in a.rationale


def test_validate_accepts_argless_actions():
    obs = Observation(target='bus')
    assert validate_action(Action(DETECT_ALL), obs)[0]
    assert validate_action(Action(TURN, turn_yaw_rad=0.5), obs)[0]
    assert validate_action(Action(DONE), obs)[0]


def test_parse_repairs_premature_done_without_strict_target():
    obs = Observation(
        target='office chair',
        context_marks=[ContextMark(1, 'desk', 0.43, distance_m=0.19,
                                   side='center', center_x_norm=0.5,
                                   relevance='office_context')])
    a, reason = parse_vlm_action({'action': 'DONE', 'rationale': 'cannot see target'}, obs)
    assert reason == 'OK'
    assert a.kind == TURN
    assert 'done_guard' in a.rationale


def test_parse_allows_done_with_close_strict_target():
    obs = Observation(
        target='office chair',
        candidates=[Candidate(1, 'office chair', 0.55, distance_m=0.5,
                              source='target')])
    a, reason = parse_vlm_action({'action': 'DONE', 'rationale': 'target reached'}, obs)
    assert reason == 'OK'
    assert a.kind == DONE


# ---- VLM tool-call build / parse -------------------------------------------

def test_build_options_lists_real_marks():
    obs = Observation(target='bus',
                      candidates=[Candidate(2, 'bus', 0.9, distance_m=3.25)],
                      map_text='occupancy map 40x40')
    opt = build_vlm_options(obs)
    mark = opt['visible_marks'][0]
    assert mark['mark_id'] == 2 and mark['distance_m'] == 3.25   # realsense range
    assert mark['source'] == 'target'
    assert 'DRIVE_TO_VISIBLE' in opt['actions'] and 'DETECT_ALL' in opt['actions']
    assert opt['map'] == 'occupancy map 40x40'
    assert 'frontiers' not in opt          # frontier options removed from the vocab


def test_build_options_lists_context_marks():
    obs = Observation(
        target='office chair',
        context_marks=[ContextMark(7, 'drawer cabinet', 0.66, distance_m=1.9,
                                   side='right', center_x_norm=0.74,
                                   relevance='office_context')])
    mark = build_vlm_options(obs)['context_marks'][0]
    assert mark == {
        'mark_id': 7,
        'label': 'drawer cabinet',
        'score': 0.66,
        'distance_m': 1.9,
        'side': 'right',
        'center_x_norm': 0.74,
        'relevance': 'office_context',
    }


def test_build_options_serializes_unknown_distance_as_null():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9)])
    mark = build_vlm_options(obs)['visible_marks'][0]
    assert mark['distance_m'] is None


def test_build_options_omits_map_when_absent():
    opt = build_vlm_options(Observation(target='bus'))
    assert 'map' not in opt


def test_parse_valid_tool_call():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', distance_m=2.0)])
    act, reason = parse_vlm_action({'action': 'DRIVE_TO_VISIBLE', 'mark_id': 2,
                                    'rationale': 'see bus'}, obs)
    assert act is not None and act.kind == DRIVE_TO_VISIBLE and reason == 'OK'


def test_parse_rejects_hallucinated_mark():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', distance_m=2.0)])
    act, reason = parse_vlm_action({'action': 'DRIVE_TO_VISIBLE', 'mark_id': 7}, obs)
    assert act is None and 'not in candidates' in reason


def test_parse_remaps_context_mark_drive_to_semantic_turn():
    obs = Observation(
        target='office chair',
        context_marks=[ContextMark(5, 'office chair chair', 0.35, distance_m=4.8,
                                   side='left', center_x_norm=0.2,
                                   relevance='target_like')])
    act, reason = parse_vlm_action(
        {'action': 'DRIVE_TO_VISIBLE', 'mark_id': 5,
         'rationale': 'inspect the partly visible chair'}, obs)
    assert reason == 'OK'
    assert act.kind == TURN and act.turn_yaw_rad > 0.0
    assert 'semantic_explore' in act.rationale


def test_parse_remaps_center_context_mark_drive_to_short_forward():
    obs = Observation(
        target='office chair',
        context_marks=[ContextMark(2, 'drawer cabinet', 0.40, distance_m=2.7,
                                   side='center', center_x_norm=0.5,
                                   relevance='office_context')])
    act, reason = parse_vlm_action({'action': 'DRIVE_TO_VISIBLE', 'mark_id': 2}, obs)
    assert reason == 'OK'
    assert act.kind == DRIVE_FORWARD and act.forward_dist_m == 0.4


def test_parse_remaps_center_context_mark_drive_to_turn_when_blocked():
    obs = Observation(
        target='office chair',
        context_marks=[
            ContextMark(1, 'desk table', 0.41, distance_m=0.31,
                        side='center', center_x_norm=0.5,
                        relevance='office_context'),
            ContextMark(9, 'chair', 0.26, distance_m=1.69,
                        side='center', center_x_norm=0.5,
                        relevance='target_like'),
        ])
    act, reason = parse_vlm_action({'action': 'DRIVE_TO_VISIBLE', 'mark_id': 9}, obs)
    assert reason == 'OK'
    assert act.kind == TURN and act.turn_yaw_rad > 0.0
    assert 'forward probe blocked' in act.rationale


def test_parse_still_rejects_low_relevance_context_mark_drive():
    obs = Observation(
        target='office chair',
        context_marks=[ContextMark(8, 'floor', 0.80, distance_m=2.0,
                                   side='center', center_x_norm=0.5,
                                   relevance='low')])
    act, reason = parse_vlm_action({'action': 'DRIVE_TO_VISIBLE', 'mark_id': 8}, obs)
    assert act is None and 'not in candidates' in reason


def test_parse_rejects_unknown_action():
    act, reason = parse_vlm_action({'action': 'TELEPORT'}, Observation(target='x'))
    assert act is None


def test_parse_turn_carries_angle():
    act, _ = parse_vlm_action({'action': 'TURN', 'turn_yaw_rad': 0.8}, Observation(target='x'))
    assert act.kind == TURN and abs(act.turn_yaw_rad - 0.8) < 1e-9


def test_parse_detect_all():
    act, reason = parse_vlm_action({'action': 'DETECT_ALL', 'rationale': 'look around'},
                                   Observation(target='x'))
    assert act is not None and act.kind == DETECT_ALL and reason == 'OK'


# ---- replan scheduler ------------------------------------------------------

def test_replan_every_n():
    s = ReplanScheduler(3)
    s.step_done(); assert not s.should_replan()
    s.step_done(); assert not s.should_replan()
    s.step_done(); assert s.should_replan()
    s.replanned(); assert not s.should_replan() and s.steps_since_replan == 0


def test_replan_min_one():
    s = ReplanScheduler(0)   # clamps to 1
    s.step_done(); assert s.should_replan()


# ---- circuit breaker -------------------------------------------------------

def test_circuit_breaker_opens_then_closes():
    cb = CircuitBreaker(max_consecutive_failures=2)
    cb.record_failure(); assert not cb.is_open
    cb.record_failure(); assert cb.is_open
    cb.record_success(); assert not cb.is_open and cb.consecutive_failures == 0


# ---- FMEA 5.1: seamless VLM->FLAT degradation latch ------------------------

PRIMARY, FALLBACK = 'VLM', 'FLAT'


def test_degradation_selects_primary_until_breaker_opens():
    d = DegradationLatch()
    assert d.select(PRIMARY, FALLBACK, cb_open=False) == PRIMARY
    assert not d.degraded


def test_degradation_latches_to_fallback_on_open():
    d = DegradationLatch()
    assert d.select(PRIMARY, FALLBACK, cb_open=True) == FALLBACK
    assert d.degraded


def test_degradation_does_not_flap_back_after_recovery():
    d = DegradationLatch()
    d.select(PRIMARY, FALLBACK, cb_open=True)          # degrade
    # breaker "recovers" -> must STAY on FLAT for the rest of the mission
    assert d.select(PRIMARY, FALLBACK, cb_open=False) == FALLBACK
    assert d.degraded


def test_degradation_announces_once():
    d = DegradationLatch()
    assert not d.just_degraded()                       # not degraded yet
    d.select(PRIMARY, FALLBACK, cb_open=True)
    assert d.just_degraded()                           # fires once at transition
    assert not d.just_degraded()                       # and only once


# ---- notes buffer ----------------------------------------------------------

def test_notes_dedup_and_summary():
    nb = NotesBuffer()
    nb.add_fact('visited room A')
    nb.add_fact('visited room A')   # dup ignored
    nb.add_fact('bus not in room A')
    assert nb.facts == ['visited room A', 'bus not in room A']
    assert 'room A' in nb.summary()
    assert nb.token_estimate() > 0


def test_notes_caps_size():
    nb = NotesBuffer(max_facts=3)
    for i in range(5):
        nb.add_fact('fact %d' % i)
    assert len(nb.facts) == 3 and nb.facts[0] == 'fact 2'   # oldest dropped
