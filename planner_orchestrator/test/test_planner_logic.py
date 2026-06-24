"""Unit tests for the VLM planner pure logic (Phase 4)."""
from planner_orchestrator.planner_logic import (
    Action, Candidate, CircuitBreaker, DegradationLatch, FrontierOpt, MockPlanner,
    NotesBuffer, Observation, ReplanScheduler, build_vlm_options, parse_vlm_action,
    validate_action, DRIVE_TO_VISIBLE, GO_TO_FRONTIER, TURN, DONE,
)


# ---- MockPlanner policy ----------------------------------------------------

def test_mock_approaches_visible_target():
    obs = Observation(target='bus',
                      candidates=[Candidate(2, 'bus', 0.9), Candidate(5, 'person', 0.8)],
                      frontiers=[FrontierOpt(7, 1.0, 50.0)])
    a = MockPlanner().plan(obs)
    assert a.kind == DRIVE_TO_VISIBLE and a.mark_id == 2


def test_mock_picks_best_matching_candidate():
    obs = Observation(target='chair',
                      candidates=[Candidate(1, 'chair', 0.4), Candidate(3, 'office chair', 0.9)])
    a = MockPlanner().plan(obs)
    assert a.kind == DRIVE_TO_VISIBLE and a.mark_id == 3   # best score among matches


def test_mock_explores_when_no_target():
    obs = Observation(target='bus', candidates=[Candidate(1, 'door', 0.7)],
                      frontiers=[FrontierOpt(4, 2.0, 10.0), FrontierOpt(9, 1.0, 80.0)])
    a = MockPlanner().plan(obs)
    assert a.kind == GO_TO_FRONTIER and a.frontier_id == 9   # best frontier score


def test_mock_scans_then_done():
    obs = Observation(target='bus')  # nothing visible, no frontiers
    mp = MockPlanner(scan_turn_limit=2, turn_step_rad=0.5)
    assert mp.plan(obs).kind == TURN
    assert mp.plan(obs).kind == TURN
    assert mp.plan(obs).kind == DONE   # exhausted


def test_mock_scan_resets_after_progress():
    mp = MockPlanner(scan_turn_limit=2)
    empty = Observation(target='bus')
    assert mp.plan(empty).kind == TURN
    # a frontier appears -> explore, scan counter resets
    assert mp.plan(Observation(target='bus', frontiers=[FrontierOpt(1, 1.0, 1.0)])).kind == GO_TO_FRONTIER
    assert mp.plan(empty).kind == TURN   # counter was reset, scans again


# ---- enum-tool-call validation (anti-hallucination) ------------------------

def test_validate_rejects_phantom_mark():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus')])
    ok, _ = validate_action(Action(DRIVE_TO_VISIBLE, mark_id=99), obs)
    assert not ok


def test_validate_accepts_real_mark():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus')])
    ok, _ = validate_action(Action(DRIVE_TO_VISIBLE, mark_id=2), obs)
    assert ok


def test_validate_rejects_phantom_frontier():
    obs = Observation(target='x', frontiers=[FrontierOpt(4, 1.0, 1.0)])
    ok, _ = validate_action(Action(GO_TO_FRONTIER, frontier_id=8), obs)
    assert not ok


def test_validate_rejects_frontier_when_none_exist():
    ok, _ = validate_action(Action(GO_TO_FRONTIER, frontier_id=-1), Observation(target='x'))
    assert not ok


# ---- VLM tool-call build / parse -------------------------------------------

def test_build_options_lists_real_ids():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9)],
                      frontiers=[FrontierOpt(7, 1.5, 30.0)])
    opt = build_vlm_options(obs)
    assert opt['visible_marks'][0]['mark_id'] == 2
    assert opt['frontiers'][0]['id'] == 7
    assert 'DRIVE_TO_VISIBLE' in opt['actions']


def test_parse_valid_tool_call():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus')])
    act, reason = parse_vlm_action({'action': 'DRIVE_TO_VISIBLE', 'mark_id': 2,
                                    'rationale': 'see bus'}, obs)
    assert act is not None and act.kind == DRIVE_TO_VISIBLE and reason == 'OK'


def test_parse_rejects_hallucinated_mark():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus')])
    act, reason = parse_vlm_action({'action': 'DRIVE_TO_VISIBLE', 'mark_id': 7}, obs)
    assert act is None and 'not in candidates' in reason


def test_parse_rejects_unknown_action():
    act, reason = parse_vlm_action({'action': 'TELEPORT'}, Observation(target='x'))
    assert act is None


def test_parse_turn_carries_angle():
    act, _ = parse_vlm_action({'action': 'TURN', 'turn_yaw_rad': 0.8}, Observation(target='x'))
    assert act.kind == TURN and abs(act.turn_yaw_rad - 0.8) < 1e-9


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
