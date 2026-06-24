"""Pure planning logic for the VLM Planner Orchestrator (Phase 4).

ROS-free by design so the atomic-action policy, replan-every-N scheduler,
structured VLM tool-call build/parse, circuit-breaker and notes buffer are
unit-testable without a running graph or network.

The orchestrator plans over ATOMIC actions (more granular than the FLAT skills):
TURN, DRIVE_FORWARD, DRIVE_TO_VISIBLE, GO_TO_FRONTIER, GET_OBSERVATION, DONE,
STOP. It replans every N atomic steps. Whether the next action comes from the
real VLM or the deterministic mock, it is ALWAYS validated against the real
options in the current Observation (enum tool-call): the planner may only select
a mark_id / frontier_id that actually exists, and never emits map coordinates.
The VLM is never on the reactive control path; the Pi executive owns motion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Atomic action kinds (mirror object_tracking_msgs/msg/AtomicAction.msg).
TURN = 0
DRIVE_FORWARD = 1
DRIVE_TO_VISIBLE = 2
GO_TO_FRONTIER = 3
GET_OBSERVATION = 4
DONE = 5
STOP = 6

ACTION_NAMES = {
    TURN: 'TURN', DRIVE_FORWARD: 'DRIVE_FORWARD', DRIVE_TO_VISIBLE: 'DRIVE_TO_VISIBLE',
    GO_TO_FRONTIER: 'GO_TO_FRONTIER', GET_OBSERVATION: 'GET_OBSERVATION',
    DONE: 'DONE', STOP: 'STOP',
}
ACTION_KINDS = {v: k for k, v in ACTION_NAMES.items()}


@dataclass(frozen=True)
class Candidate:
    """A Set-of-Mark detection the VLM may target by mark_id."""
    mark_id: int
    label: str
    score: float = 0.0


@dataclass(frozen=True)
class FrontierOpt:
    """A frontier option the VLM may pick by id."""
    id: int
    distance_m: float = 0.0
    score: float = 0.0


@dataclass(frozen=True)
class Observation:
    """Everything the planner sees at a replan point. ROS-free."""
    target: str                          # mission instruction / object description
    candidates: List[Candidate] = field(default_factory=list)
    frontiers: List[FrontierOpt] = field(default_factory=list)
    notes_facts: List[str] = field(default_factory=list)
    step_index: int = 0                  # atomic steps executed so far this mission


@dataclass
class Action:
    """One atomic action. Mirrors AtomicAction.msg; rationale feeds the notes/logs."""
    kind: int
    turn_yaw_rad: float = 0.0
    forward_dist_m: float = 0.0
    mark_id: int = 0
    frontier_id: int = -1
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

    Rejects a DRIVE_TO_VISIBLE whose mark_id is not in the candidate list and a
    GO_TO_FRONTIER whose frontier_id is not in the frontier list (>=0). This is
    what stops a hallucinating VLM from steering the robot at a phantom target.
    """
    if action.kind not in ACTION_NAMES:
        return False, 'unknown action kind %r' % action.kind
    if action.kind == DRIVE_TO_VISIBLE:
        if action.mark_id not in {c.mark_id for c in obs.candidates}:
            return False, 'mark_id %d not in candidates' % action.mark_id
    if action.kind == GO_TO_FRONTIER:
        if action.frontier_id >= 0 and action.frontier_id not in {f.id for f in obs.frontiers}:
            return False, 'frontier_id %d not in frontiers' % action.frontier_id
        if action.frontier_id < 0 and not obs.frontiers:
            return False, 'GO_TO_FRONTIER but no frontiers exist'
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

    Policy (target = mission object): approach a matching visible detection;
    else step toward the best frontier; else rotate to scan; after a bounded
    number of fruitless scans declare DONE. Same Observation contract as the real
    VLM client, so swapping in the API changes nothing else.
    """

    def __init__(self, scan_turn_limit: int = 6, turn_step_rad: float = 0.6):
        self.scan_turn_limit = scan_turn_limit
        self.turn_step_rad = turn_step_rad
        self._scans = 0

    def plan(self, obs: Observation) -> Action:
        # 1) target visible -> approach it (best score among matching).
        matches = [c for c in obs.candidates if _label_matches(obs.target, c.label)]
        if matches:
            best = max(matches, key=lambda c: c.score)
            self._scans = 0
            return Action(DRIVE_TO_VISIBLE, mark_id=best.mark_id, arg_label=best.label,
                          rationale='target "%s" visible as mark %d' % (obs.target, best.mark_id))
        # 2) unexplored space -> step toward the best frontier.
        if obs.frontiers:
            self._scans = 0
            best = max(obs.frontiers, key=lambda f: f.score)
            return Action(GO_TO_FRONTIER, frontier_id=best.id,
                          rationale='no target in view; explore frontier %d' % best.id)
        # 3) nothing actionable -> rotate to scan the surroundings.
        if self._scans < self.scan_turn_limit:
            self._scans += 1
            return Action(TURN, turn_yaw_rad=self.turn_step_rad,
                          rationale='scan-rotate %d/%d (no target, no frontier)'
                          % (self._scans, self.scan_turn_limit))
        # 4) exhausted -> finish.
        return Action(DONE, rationale='no target and exploration exhausted')


def build_vlm_options(obs: Observation) -> dict:
    """Structured options handed to the VLM: the atomic vocabulary + the REAL
    selectable ids. The model returns an enum tool-call referencing only these."""
    return {
        'target': obs.target,
        'actions': list(ACTION_NAMES.values()),
        'visible_marks': [{'mark_id': c.mark_id, 'label': c.label, 'score': round(c.score, 3)}
                          for c in obs.candidates],
        'frontiers': [{'id': f.id, 'distance_m': round(f.distance_m, 2), 'score': round(f.score, 2)}
                      for f in obs.frontiers],
        'notes': obs.notes_facts,
        'step_index': obs.step_index,
    }


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
        frontier_id=int(resp.get('frontier_id', -1) if resp.get('frontier_id') is not None else -1),
        arg_label=str(resp.get('arg_label', '') or ''),
        rationale=str(resp.get('rationale', '') or ''),
    )
    ok, reason = validate_action(act, obs)
    if not ok:
        return None, reason
    return act, 'OK'
