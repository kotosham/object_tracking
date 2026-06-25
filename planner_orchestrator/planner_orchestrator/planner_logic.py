"""Pure planning logic for the VLM Planner Orchestrator (Phase 4).

ROS-free by design so the atomic-action policy, replan-every-N scheduler,
structured VLM tool-call build/parse, circuit-breaker and notes buffer are
unit-testable without a running graph or network.

The orchestrator plans over a small set of HONEST primitives so a VLM-vs-FLAT
comparison is fair: the VLM does its own navigation reasoning from raw motion +
perception, instead of delegating it to a high-level frontier/approach planner.
Vocabulary: TURN, DRIVE_FORWARD, DRIVE_TO_VISIBLE, DETECT_ALL, DONE. It replans
every N atomic steps. Whether the next action comes from the real VLM or the
deterministic mock, it is ALWAYS validated against the real options in the
current Observation (enum tool-call): the planner may only DRIVE_TO_VISIBLE a
mark_id that actually exists, and never emits map coordinates. The VLM is never
on the reactive control path; the Pi executive owns motion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Atomic action kinds (mirror object_tracking_msgs/msg/AtomicAction.msg).
#   TURN           rotate in place by turn_yaw_rad (+ = CCW)
#   DRIVE_FORWARD  drive forward_dist_m along the heading (negative = backward)
#   DRIVE_TO_VISIBLE  approach a detected object by mark_id, via Nav (ApproachDetection)
#   DETECT_ALL     run the detector over a broad vocabulary -> all objects + classes
#   DONE           mission complete / target reached
TURN = 0
DRIVE_FORWARD = 1
DRIVE_TO_VISIBLE = 2
DETECT_ALL = 3
DONE = 4

ACTION_NAMES = {
    TURN: 'TURN', DRIVE_FORWARD: 'DRIVE_FORWARD', DRIVE_TO_VISIBLE: 'DRIVE_TO_VISIBLE',
    DETECT_ALL: 'DETECT_ALL', DONE: 'DONE',
}
ACTION_KINDS = {v: k for k, v in ACTION_NAMES.items()}


@dataclass(frozen=True)
class Candidate:
    """A Set-of-Mark detection the VLM may target by mark_id. distance_m is the
    metric range to the object from the RealSense aligned depth (0.0 if unknown)."""
    mark_id: int
    label: str
    score: float = 0.0
    distance_m: float = 0.0


@dataclass(frozen=True)
class Observation:
    """Everything the planner sees at a replan point. ROS-free. The camera frame and
    the top-down SLAM map are passed as images alongside this (see VlmClient.plan);
    map_text describes that map so the model can read it."""
    target: str                          # mission instruction / object description
    candidates: List[Candidate] = field(default_factory=list)
    notes_facts: List[str] = field(default_factory=list)
    step_index: int = 0                  # atomic steps executed so far this mission
    map_text: str = ''                   # human description of the attached SLAM map


@dataclass
class Action:
    """One atomic action. Mirrors AtomicAction.msg; rationale feeds the notes/logs."""
    kind: int
    turn_yaw_rad: float = 0.0
    forward_dist_m: float = 0.0
    mark_id: int = 0
    arg_label: str = ''
    rationale: str = ''

    @property
    def name(self) -> str:
        return ACTION_NAMES.get(self.kind, 'UNKNOWN')


def _label_matches(target: str, label: str) -> bool:
    """Loose open-vocab match between the mission target and a detection label."""
    t = (target or '').strip().lower()
    l = (label or '').strip().lower()
    if not t or not l:
        return False
    return t in l or l in t or bool(set(t.split()) & set(l.split()))


def validate_action(action: Action, obs: Observation) -> Tuple[bool, str]:
    """Enum-tool-call guard: the action must reference options that REALLY exist.

    Rejects a DRIVE_TO_VISIBLE whose mark_id is not in the candidate list -- this
    is what stops a hallucinating VLM from steering the robot at a phantom target.
    TURN / DRIVE_FORWARD / DETECT_ALL / DONE carry no id to validate.
    """
    if action.kind not in ACTION_NAMES:
        return False, 'unknown action kind %r' % action.kind
    if action.kind == DRIVE_TO_VISIBLE:
        if action.mark_id not in {c.mark_id for c in obs.candidates}:
            return False, 'mark_id %d not in candidates' % action.mark_id
    return True, 'OK'


class ReplanScheduler:
    """Replan every N executed atomic steps (Phase 4, user spec: N=2..3 default 3)."""

    def __init__(self, n: int = 3):
        self.n = max(1, int(n))
        self._since = 0

    def step_done(self) -> None:
        self._since += 1

    def should_replan(self) -> bool:
        return self._since >= self.n

    def replanned(self) -> None:
        self._since = 0

    @property
    def steps_since_replan(self) -> int:
        return self._since


class CircuitBreaker:
    """Open after K consecutive VLM failures/timeouts -> degrade VLM->FLAT.

    p99-driven: the caller feeds observed latencies; a call slower than the
    timeout (derived from measured p99) counts as a failure, as does an error.
    """

    def __init__(self, max_consecutive_failures: int = 3):
        self.max = max(1, int(max_consecutive_failures))
        self._fails = 0
        self._open = False

    def record_success(self) -> None:
        self._fails = 0
        self._open = False

    def record_failure(self) -> None:
        self._fails += 1
        if self._fails >= self.max:
            self._open = True

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def consecutive_failures(self) -> int:
        return self._fails


class DegradationLatch:
    """Phase 5.1 seamless VLM->FLAT degradation. When the circuit-breaker opens
    (VLM lost / unreachable / edge link gone), latch to a FLAT fallback policy for
    the REST of the mission so it CONTINUES (DEGRADED) instead of stopping -- and
    never flaps back to the VLM mid-mission even if the breaker later recovers.

    `select(primary, fallback, cb_open)` returns the planner to use this cycle and
    sets the latch the first time it sees an open breaker. `just_degraded()` fires
    once at the transition so the caller can log / note it exactly once."""

    def __init__(self):
        self._degraded = False
        self._announced = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def select(self, primary, fallback, cb_open: bool):
        if cb_open:
            self._degraded = True
        return fallback if self._degraded else primary

    def just_degraded(self) -> bool:
        """True exactly once, on the cycle degradation first latches."""
        if self._degraded and not self._announced:
            self._announced = True
            return True
        return False


class NotesBuffer:
    """Compact, deduped fact list + summary (context kept instead of frames)."""

    def __init__(self, max_facts: int = 24):
        self.max_facts = max_facts
        self._facts: List[str] = []

    def add_fact(self, fact: str) -> None:
        fact = (fact or '').strip()
        if fact and fact not in self._facts:
            self._facts.append(fact)
            if len(self._facts) > self.max_facts:
                self._facts.pop(0)   # drop oldest; summary keeps the gist

    @property
    def facts(self) -> List[str]:
        return list(self._facts)

    def summary(self) -> str:
        return '; '.join(self._facts)

    def token_estimate(self) -> int:
        # ~4 chars/token rough budget proxy (Phase 4.5 token control).
        return (sum(len(f) for f in self._facts) + len(self._facts)) // 4


class MockPlanner:
    """Deterministic stand-in for the VLM so the whole loop runs/tests with no API.

    Policy (target = mission object), using only the honest primitives: approach a
    matching visible detection; else take ONE wide look (DETECT_ALL) to list what is
    around; else rotate to scan; after a bounded number of fruitless scans declare
    DONE. Same Observation contract as the real VLM client, so swapping in the API
    changes nothing else. Also serves as the FLAT degradation fallback, so it must
    always drive the loop to a terminal action.
    """

    def __init__(self, scan_turn_limit: int = 6, turn_step_rad: float = 0.6,
                 reached_dist_m: float = 0.8, max_approaches: int = 8):
        self.scan_turn_limit = scan_turn_limit
        self.turn_step_rad = turn_step_rad
        self.reached_dist_m = reached_dist_m
        self.max_approaches = max_approaches
        self._scans = 0
        self._looked = False
        self._approaches = 0             # consecutive DRIVE_TO_VISIBLE toward this target

    def plan(self, obs: Observation) -> Action:
        # 1) target visible.
        matches = [c for c in obs.candidates if _label_matches(obs.target, c.label)]
        if matches:
            best = max(matches, key=lambda c: c.score)
            self._scans = 0
            self._looked = False
            # arrived: within the RealSense reached range -> done.
            if 0.0 < best.distance_m <= self.reached_dist_m:
                self._approaches = 0
                return Action(DONE, rationale='target "%s" reached (%.2fm)'
                              % (obs.target, best.distance_m))
            # safety bound so a non-converging approach can't loop forever.
            if self._approaches >= self.max_approaches:
                self._approaches = 0
                return Action(DONE, rationale='target "%s" approached %dx without closing in '
                              '(%.2fm) -> stopping' % (obs.target, self.max_approaches,
                                                       best.distance_m))
            # otherwise keep driving up to it.
            self._approaches += 1
            return Action(DRIVE_TO_VISIBLE, mark_id=best.mark_id, arg_label=best.label,
                          rationale='target "%s" visible as mark %d (%.2fm)'
                          % (obs.target, best.mark_id, best.distance_m))
        # 2) not in view but we WERE just driving up to it -> at point-blank it overflows
        #    the frame and YOLOE drops it: treat that as arrived.
        if self._approaches > 0:
            self._approaches = 0
            return Action(DONE, rationale='target reached (dropped out of frame at close range)')
        # 3) nothing matching in view -> one broad look before scanning.
        if not self._looked:
            self._looked = True
            return Action(DETECT_ALL, rationale='no target in view; detect all objects')
        # 4) rotate to bring new things into view.
        if self._scans < self.scan_turn_limit:
            self._scans += 1
            return Action(TURN, turn_yaw_rad=self.turn_step_rad,
                          rationale='scan-rotate %d/%d (no target)'
                          % (self._scans, self.scan_turn_limit))
        # 5) exhausted -> finish.
        return Action(DONE, rationale='no target after scanning')


def build_vlm_options(obs: Observation) -> dict:
    """Structured options handed to the VLM: the atomic vocabulary + the REAL
    selectable marks (each with its class label and metric distance), plus the SLAM
    map description. The model returns an enum tool-call referencing only these."""
    opts = {
        'target': obs.target,
        'actions': list(ACTION_NAMES.values()),
        'visible_marks': [{'mark_id': c.mark_id, 'label': c.label,
                           'score': round(c.score, 3),
                           'distance_m': round(c.distance_m, 2)}
                          for c in obs.candidates],
        'notes': obs.notes_facts,
        'step_index': obs.step_index,
    }
    if obs.map_text:
        opts['map'] = obs.map_text
    return opts


def parse_vlm_action(resp: dict, obs: Observation) -> Tuple[Optional[Action], str]:
    """Parse a VLM tool-call response into a validated Action.

    `resp` is the decoded tool-call arguments, e.g.
    {"action": "DRIVE_TO_VISIBLE", "mark_id": 2, "rationale": "..."}.
    Returns (Action, 'OK') or (None, reason). The action is validated against the
    real options so a hallucinated id is rejected (caller then falls back/degrades).
    """
    if not isinstance(resp, dict):
        return None, 'response is not an object'
    name = str(resp.get('action', '')).strip().upper()
    if name not in ACTION_KINDS:
        return None, 'unknown action %r' % name
    act = Action(
        kind=ACTION_KINDS[name],
        turn_yaw_rad=float(resp.get('turn_yaw_rad', 0.0) or 0.0),
        forward_dist_m=float(resp.get('forward_dist_m', 0.0) or 0.0),
        mark_id=int(resp.get('mark_id', 0) or 0),
        arg_label=str(resp.get('arg_label', '') or ''),
        rationale=str(resp.get('rationale', '') or ''),
    )
    ok, reason = validate_action(act, obs)
    if not ok:
        return None, reason
    return act, 'OK'
