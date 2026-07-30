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
import math
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

CONTEXT_EXPLORE_TURN_RAD = 0.6
CONTEXT_EXPLORE_FORWARD_M = 0.4
CONTEXT_EXPLORE_MIN_CLEARANCE_M = 0.8
TARGET_PROBE_TURN_RAD = 0.45
TARGET_PROBE_FORWARD_M = 0.6
TARGET_CONTEXT_PROMOTE_MIN_SCORE = 0.35
STRICT_TARGET_DONE_DIST_M = 0.8
TARGET_EDGE_REACQUIRE_MARGIN_NORM = 0.08


@dataclass(frozen=True)
class Candidate:
    """A Set-of-Mark detection the VLM may target by mark_id. distance_m is the
    metric range to the object from the RealSense aligned depth. NaN/<=0 means
    depth is unknown and cannot be used for ApproachDetection geometry."""
    mark_id: int
    label: str
    score: float = 0.0
    distance_m: float = 0.0
    side: str = 'center'                 # left / center / right in the camera image
    center_x_norm: float = 0.5
    source: str = 'target'               # target / context_promoted / fallback
    pixel_x_norm: float = 0.5            # nav/depth point, not bbox center
    pixel_y_norm: float = 0.5


@dataclass(frozen=True)
class ContextMark:
    """A non-target object visible in the scene. Context marks are for reasoning
    only: they can suggest where to explore, but DRIVE_TO_VISIBLE may still target
    only real target Candidates from visible_marks."""
    mark_id: int
    label: str
    score: float = 0.0
    distance_m: float = 0.0
    side: str = 'center'                 # left / center / right in the camera image
    center_x_norm: float = 0.5           # 0.0 = left edge, 1.0 = right edge
    relevance: str = 'low'               # target_like / office_context / ambiguous / low


OFFICE_CONTEXT_TERMS = (
    'desk', 'table', 'drawer', 'drawer cabinet', 'file cabinet', 'bookcase',
    'shelf', 'bookshelf', 'monitor', 'keyboard', 'laptop', 'computer', 'printer',
    'office chair', 'chair', 'sofa', 'couch',
)


def image_side(center_x_norm: float) -> str:
    try:
        x = float(center_x_norm)
    except (TypeError, ValueError):
        return 'center'
    if x < 0.4:
        return 'left'
    if x > 0.6:
        return 'right'
    return 'center'


def context_relevance_for(target: str, label: str) -> str:
    """Small, deterministic hint for the VLM: which visible non-target objects are
    semantically useful search cues. The image is still the authority; this just
    prevents the fallback policy from treating every random object as equally useful."""
    t = (target or '').strip().lower()
    l = (label or '').strip().lower()
    if not l:
        return 'low'
    if _label_matches(t, l):
        return 'target_like'
    wants_office = any(w in t for w in ('chair', 'office', 'desk', 'table', 'cabinet'))
    if wants_office and any(term in l for term in OFFICE_CONTEXT_TERMS):
        return 'office_context'
    if 'cabinet' in l or 'shelf' in l or 'box' in l:
        return 'ambiguous'
    return 'low'


def context_mark_promotable_to_target(target: str, mark: ContextMark,
                                      min_score: float = TARGET_CONTEXT_PROMOTE_MIN_SCORE
                                      ) -> bool:
    """Whether a context detection should be treated as a real target candidate.

    A context pass can find the target under a broader query, e.g. target="chair"
    while DINO office-context returns label="office chair". If the label really
    matches the mission target and confidence is not just a tiny hint, promote it
    so DRIVE_TO_VISIBLE may use its pixel/depth instead of only turning toward it.
    """
    if mark is None or int(mark.mark_id) <= 0:
        return False
    try:
        score = float(mark.score)
    except (TypeError, ValueError):
        return False
    if score < float(min_score):
        return False
    return (mark.relevance or '').lower() == 'target_like' and _label_matches(
        target, mark.label)


def distance_is_known(distance_m: float) -> bool:
    try:
        d = float(distance_m)
    except (TypeError, ValueError):
        return False
    return math.isfinite(d) and d > 0.0


def distance_for_options(distance_m: float):
    return round(float(distance_m), 2) if distance_is_known(distance_m) else None


def format_distance(distance_m: float) -> str:
    return 'unknown' if not distance_is_known(distance_m) else '%.2fm' % float(distance_m)


def centered_forward_blocker(obs: Observation,
                             min_clearance_m: float = CONTEXT_EXPLORE_MIN_CLEARANCE_M
                             ) -> Optional[ContextMark]:
    """Closest centered context mark that makes a blind/context forward probe unsafe."""
    blockers = [
        m for m in obs.context_marks
        if m.side == 'center'
        and distance_is_known(m.distance_m)
        and float(m.distance_m) < float(min_clearance_m)
    ]
    if not blockers:
        return None
    return min(blockers, key=lambda m: float(m.distance_m))


def useful_context_marks(obs: Observation) -> List['ContextMark']:
    return [
        m for m in obs.context_marks
        if (m.relevance or '').lower() in ('target_like', 'office_context', 'ambiguous')
    ]


def best_context_mark(obs: Observation) -> Optional['ContextMark']:
    useful = useful_context_marks(obs)
    if not useful:
        return None
    return max(useful, key=lambda m: (
        2 if m.relevance == 'target_like' else 1 if m.relevance == 'office_context' else 0,
        float(m.score),
        -float(m.distance_m) if distance_is_known(m.distance_m) else -999.0,
    ))


def best_directional_context_mark(obs: Observation) -> Optional['ContextMark']:
    directional = [m for m in useful_context_marks(obs) if m.side in ('left', 'right')]
    if not directional:
        return None
    return max(directional, key=lambda m: (
        2 if m.relevance == 'target_like' else 1 if m.relevance == 'office_context' else 0,
        float(m.score),
        -float(m.distance_m) if distance_is_known(m.distance_m) else -999.0,
    ))


@dataclass(frozen=True)
class Observation:
    """Everything the planner sees at a replan point. ROS-free. The camera frame and
    the top-down SLAM map are passed as images alongside this (see VlmClient.plan);
    map_text describes that map so the model can read it."""
    target: str                          # mission instruction / object description
    candidates: List[Candidate] = field(default_factory=list)
    context_marks: List[ContextMark] = field(default_factory=list)
    notes_facts: List[str] = field(default_factory=list)
    step_index: int = 0                  # atomic steps executed so far this mission
    map_text: str = ''                   # human description of the attached SLAM map
    corridor_scan: List[dict] = field(default_factory=list)


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


def _strict_known_target_candidates(obs: Observation) -> List[Candidate]:
    return [
        c for c in obs.candidates
        if (c.source or '') == 'target'
        and _label_matches(obs.target, c.label)
        and distance_is_known(c.distance_m)
    ]


def _candidate_edge_side(cand: Candidate,
                         margin_norm: float = TARGET_EDGE_REACQUIRE_MARGIN_NORM
                         ) -> str:
    try:
        x = float(cand.pixel_x_norm)
    except (TypeError, ValueError):
        x = float(cand.center_x_norm)
    margin = max(0.0, min(0.49, float(margin_norm)))
    if x < margin:
        return 'left'
    if x > 1.0 - margin:
        return 'right'
    return ''


def validate_action(action: Action, obs: Observation) -> Tuple[bool, str]:
    """Enum-tool-call guard: the action must reference options that REALLY exist.

    Rejects a DRIVE_TO_VISIBLE whose mark_id is not in the candidate list -- this
    is what stops a hallucinating VLM from steering the robot at a phantom target.
    TURN / DRIVE_FORWARD / DETECT_ALL / DONE carry no id to validate.
    """
    if action.kind not in ACTION_NAMES:
        return False, 'unknown action kind %r' % action.kind
    if action.kind == DRIVE_TO_VISIBLE:
        by_id = {c.mark_id: c for c in obs.candidates}
        if action.mark_id not in by_id:
            return False, 'mark_id %d not in candidates' % action.mark_id
        if not distance_is_known(by_id[action.mark_id].distance_m):
            return False, 'mark_id %d has unknown distance' % action.mark_id
    return True, 'OK'


def context_mark_to_semantic_explore(action: Action, obs: Observation) -> Optional[Action]:
    """Repair a common VLM mistake: it may choose DRIVE_TO_VISIBLE for a
    context_mark (for example a partly visible chair leg) instead of emitting a
    semantic_explore motion. Context marks are not approach targets; they are
    only cues for choosing which corridor/free region to inspect next."""
    if action.kind != DRIVE_TO_VISIBLE:
        return None
    by_target_id = {c.mark_id for c in obs.candidates}
    if action.mark_id in by_target_id:
        return None
    by_context_id = {m.mark_id: m for m in obs.context_marks}
    mark = by_context_id.get(action.mark_id)
    if mark is None:
        return None
    relevance = (mark.relevance or '').lower()
    if relevance not in ('target_like', 'office_context', 'ambiguous'):
        return None

    base = ('semantic_explore: VLM selected context mark %d "%s" (%s on %s), '
            'but context objects are corridor cues, not approach targets'
            % (mark.mark_id, mark.label, mark.relevance, mark.side))
    if action.rationale:
        base += '; original rationale: ' + action.rationale
    if mark.side == 'left':
        return Action(TURN, turn_yaw_rad=CONTEXT_EXPLORE_TURN_RAD,
                      arg_label=mark.label, rationale=base)
    if mark.side == 'right':
        return Action(TURN, turn_yaw_rad=-CONTEXT_EXPLORE_TURN_RAD,
                      arg_label=mark.label, rationale=base)
    return Action(DETECT_ALL, arg_label=mark.label,
                  rationale=(base + '; centered context does not define a safe '
                             'approach target, so refresh detections/map context '
                             'before choosing a corridor'))


def unknown_depth_target_to_probe(action: Action, obs: Observation) -> Optional[Action]:
    """Repair a common VLM mistake: the target is visible but too far/invalid for
    RealSense depth, so DRIVE_TO_VISIBLE cannot compute a 3D Nav2 goal yet. Keep
    pursuing the target by using its image side as a cautious probe motion."""
    if action.kind != DRIVE_TO_VISIBLE:
        return None
    by_id = {c.mark_id: c for c in obs.candidates}
    cand = by_id.get(action.mark_id)
    if cand is None or distance_is_known(cand.distance_m):
        return None

    base = ('target_probe: target mark %d "%s" is visible on %s but depth is unknown; '
            'move to bring it into RealSense range before ApproachDetection'
            % (cand.mark_id, cand.label, cand.side))
    if action.rationale:
        base += '; original rationale: ' + action.rationale
    if cand.side == 'left':
        return Action(TURN, turn_yaw_rad=TARGET_PROBE_TURN_RAD,
                      arg_label=cand.label, rationale=base)
    if cand.side == 'right':
        return Action(TURN, turn_yaw_rad=-TARGET_PROBE_TURN_RAD,
                      arg_label=cand.label, rationale=base)
    return Action(DRIVE_FORWARD, forward_dist_m=TARGET_PROBE_FORWARD_M,
                  arg_label=cand.label, rationale=base)


def edge_target_to_recenter(action: Action, obs: Observation) -> Optional[Action]:
    """A depth-backed target on the image edge is not safe enough to approach.

    Edge detections are often partial/occluded; the chosen nearest depth point can
    land on a chair leg, table leg, or background. First rotate to put the object
    closer to the center, then re-detect before computing a Nav2 approach goal.
    """
    if action.kind != DRIVE_TO_VISIBLE:
        return None
    by_id = {c.mark_id: c for c in obs.candidates}
    cand = by_id.get(action.mark_id)
    if cand is None or not distance_is_known(cand.distance_m):
        return None
    if not _label_matches(obs.target, cand.label):
        return None
    side = _candidate_edge_side(cand)
    if not side:
        return None
    turn = TARGET_PROBE_TURN_RAD if side == 'left' else -TARGET_PROBE_TURN_RAD
    base = ('edge_target_guard: target mark %d "%s" is depth-backed but its '
            'navigation pixel is on the %s image edge; recenter and re-detect '
            'before ApproachDetection'
            % (cand.mark_id, cand.label, side))
    if action.rationale:
        base += '; original rationale: ' + action.rationale
    return Action(TURN, turn_yaw_rad=turn, arg_label=cand.label, rationale=base)


def target_probe_action(obs: Observation, mark_id: int = 0,
                        rationale: str = '') -> Optional[Action]:
    """Convert a pseudo TARGET_PROBE intent into a real atomic action.

    The public action vocabulary intentionally stays small; target_probe is a
    reasoning mode implemented as TURN/DRIVE_FORWARD.
    """
    matches = [c for c in obs.candidates
               if _label_matches(obs.target, c.label)
               and not distance_is_known(c.distance_m)]
    if mark_id:
        matches = [c for c in matches if int(c.mark_id) == int(mark_id)]
    if not matches:
        return None
    best = max(matches, key=lambda c: float(c.score))
    return unknown_depth_target_to_probe(
        Action(DRIVE_TO_VISIBLE, mark_id=best.mark_id,
               arg_label=best.label, rationale=rationale),
        obs)


def context_forward_to_directional_explore(action: Action, obs: Observation) -> Optional[Action]:
    """Keep corridor exploration as the default when the target is absent.

    Context marks are semantic hints, not destinations. Older logic rewrote a
    VLM DRIVE_FORWARD into a TURN toward the strongest desk/cabinet, which made
    the robot orbit furniture instead of exploring the free corridor the map
    showed. Preserve DRIVE_FORWARD unless a close centered context object makes
    moving straight unsafe; in that case, turn to look for an adjacent corridor.
    """
    if action.kind != DRIVE_FORWARD:
        return None
    if obs.candidates:
        return None
    blocker = centered_forward_blocker(obs)
    if blocker is None:
        return None

    best = best_directional_context_mark(obs)
    base = ('semantic_explore: target "%s" not visible; requested corridor '
            'probe is blocked by close centered context mark %d "%s" at %s'
            % (obs.target, blocker.mark_id, blocker.label,
               format_distance(blocker.distance_m)))
    if action.rationale:
        base += '; original rationale: ' + action.rationale
    if best is None:
        return Action(DETECT_ALL,
                      rationale=(base + '; refresh detections before moving, '
                                 'do not drive into the context object'))
    if best.side == 'left':
        return Action(TURN, turn_yaw_rad=CONTEXT_EXPLORE_TURN_RAD,
                      arg_label=best.label,
                      rationale=(base + '; turn left to search for free space '
                                 'beside the obstacle, not to approach the context object'))
    if best.side == 'right':
        return Action(TURN, turn_yaw_rad=-CONTEXT_EXPLORE_TURN_RAD,
                      arg_label=best.label,
                      rationale=(base + '; turn right to search for free space '
                                 'beside the obstacle, not to approach the context object'))

    return None


def context_turn_to_directional_explore(action: Action, obs: Observation) -> Optional[Action]:
    """Normalize weak or wrong-way VLM turns when semantic context already says
    which side is worth inspecting. This prevents many tiny 10-degree "nudges"
    that look like the robot is doing nothing.
    """
    if action.kind != TURN:
        return None
    if obs.candidates:
        return None
    best = best_directional_context_mark(obs)
    if best is None:
        return None
    desired = (CONTEXT_EXPLORE_TURN_RAD if best.side == 'left'
               else -CONTEXT_EXPLORE_TURN_RAD)
    same_direction = float(action.turn_yaw_rad) * desired > 0.0
    if same_direction and abs(float(action.turn_yaw_rad)) >= CONTEXT_EXPLORE_TURN_RAD * 0.75:
        return None
    base = ('semantic_explore: target "%s" not visible; normalizing turn toward '
            'context mark %d "%s" (%s on %s)'
            % (obs.target, best.mark_id, best.label, best.relevance, best.side))
    if action.rationale:
        base += '; original rationale: ' + action.rationale
    return Action(TURN, turn_yaw_rad=desired, arg_label=best.label, rationale=base)


def visible_target_to_approach(action: Action, obs: Observation) -> Optional[Action]:
    """If a strict target is visible but not close enough, keep approaching it.

    This repairs VLM indecision near the goal. If the target is a confirmed
    visible mark, DETECT_ALL/TURN/DRIVE_FORWARD/DONE are worse than simply
    continuing DRIVE_TO_VISIBLE until the close-range DONE threshold is met.
    """
    if action.kind == DRIVE_TO_VISIBLE:
        return None
    strict = _strict_known_target_candidates(obs)
    if not strict:
        return None
    stable = [c for c in strict if not _candidate_edge_side(c)]
    if not stable:
        closest_edge = min(strict, key=lambda c: float(c.distance_m))
        return edge_target_to_recenter(
            Action(DRIVE_TO_VISIBLE, mark_id=closest_edge.mark_id,
                   arg_label=closest_edge.label, rationale=action.rationale),
            obs)
    closest = min(stable, key=lambda c: float(c.distance_m))
    if float(closest.distance_m) <= STRICT_TARGET_DONE_DIST_M:
        return None
    base = ('target_guard: strict target "%s" is visible as mark %d at %.2fm; '
            'approaching it instead of %s'
            % (obs.target, closest.mark_id, float(closest.distance_m), action.name))
    if action.rationale:
        base += '; original rationale: ' + action.rationale
    return Action(DRIVE_TO_VISIBLE, mark_id=closest.mark_id,
                  arg_label=closest.label, rationale=base)


def lost_target_lock_recovery_action(obs: Observation, label: str,
                                     distance_m: float, side: str,
                                     age_steps: int,
                                     turn_step_rad: float = CONTEXT_EXPLORE_TURN_RAD,
                                     forward_dist_m: float = CONTEXT_EXPLORE_FORWARD_M
                                     ) -> Optional[Action]:
    """Recover a recently confirmed target that temporarily dropped out of view.

    The lock deliberately does not reuse a stale pixel as DRIVE_TO_VISIBLE. After
    the robot moves, old image coordinates are no longer a valid 3D observation.
    Instead, recover by turning toward the last confirmed image side, or by
    probing forward if the target was centered and the immediate view is clear.
    """
    if obs.candidates:
        return None
    if not _label_matches(obs.target, label):
        return None
    locked_side = side if side in ('left', 'right', 'center') else 'center'
    base = ('target_lock: last confirmed target "%s" was %.2fm on %s %d step(s) ago; '
            'try to reacquire it before generic semantic search'
            % (label, float(distance_m), locked_side, int(age_steps)))
    turn = abs(float(turn_step_rad))
    if locked_side == 'left':
        return Action(TURN, turn_yaw_rad=turn, arg_label=label, rationale=base)
    if locked_side == 'right':
        return Action(TURN, turn_yaw_rad=-turn, arg_label=label, rationale=base)
    return Action(DRIVE_FORWARD, forward_dist_m=max(0.05, float(forward_dist_m)),
                  arg_label=label,
                  rationale=(base + '; target was centered, probe forward; '
                             'navigation costmaps decide whether the short recovery is safe'))


def premature_done_to_continue(action: Action, obs: Observation) -> Optional[Action]:
    """DONE is allowed only when a strict target candidate is currently close.

    A VLM often says DONE after a failed/probing context approach because "no more
    useful information" is visible. That is unsafe for the real robot: absence of
    a strict target means keep searching, not finish.
    """
    if action.kind != DONE:
        return None
    strict = _strict_known_target_candidates(obs)
    if strict:
        closest = min(strict, key=lambda c: float(c.distance_m))
        if float(closest.distance_m) <= STRICT_TARGET_DONE_DIST_M:
            return None

    best = best_context_mark(obs)
    base = ('done_guard: target "%s" is not confirmed by a close strict detection; '
            'continuing search instead of DONE' % obs.target)
    if action.rationale:
        base += '; original rationale: ' + action.rationale
    if best is None:
        return Action(DETECT_ALL, rationale=base + '; refresh detections')
    if best.side == 'left':
        return Action(TURN, turn_yaw_rad=CONTEXT_EXPLORE_TURN_RAD,
                      arg_label=best.label, rationale=base)
    if best.side == 'right':
        return Action(TURN, turn_yaw_rad=-CONTEXT_EXPLORE_TURN_RAD,
                      arg_label=best.label, rationale=base)
    return Action(DRIVE_FORWARD, forward_dist_m=CONTEXT_EXPLORE_FORWARD_M,
                  arg_label=best.label,
                  rationale=(base + '; centered context, probe forward; '
                             'navigation costmaps decide whether the short motion is safe'))


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
                 reached_dist_m: float = 0.8, max_approaches: int = 8,
                 semantic_forward_m: float = 0.4,
                 lost_after_approach_done_dist_m: float = 1.8):
        self.scan_turn_limit = scan_turn_limit
        self.turn_step_rad = turn_step_rad
        self.reached_dist_m = reached_dist_m
        self.max_approaches = max_approaches
        self.semantic_forward_m = semantic_forward_m
        self.lost_after_approach_done_dist_m = lost_after_approach_done_dist_m
        self._scans = 0
        self._looked = False
        self._approaches = 0             # consecutive DRIVE_TO_VISIBLE toward this target
        self._last_approach_start_dist = None

    def _semantic_explore_action(self, obs: Observation) -> Optional[Action]:
        best = best_context_mark(obs)
        if best is None:
            return None
        detail = 'semantic_explore: target "%s" not visible; context mark %d "%s" is %s on %s' % (
            obs.target, best.mark_id, best.label, best.relevance, best.side)
        if best.side == 'left':
            return Action(TURN, turn_yaw_rad=self.turn_step_rad, rationale=detail)
        if best.side == 'right':
            return Action(TURN, turn_yaw_rad=-self.turn_step_rad, rationale=detail)
        return Action(DRIVE_FORWARD, forward_dist_m=self.semantic_forward_m,
                      rationale=(detail + '; move forward to inspect it; '
                                 'navigation costmaps decide whether the short motion is safe'))

    def plan(self, obs: Observation) -> Action:
        # 1) target visible.
        matches = [c for c in obs.candidates if _label_matches(obs.target, c.label)]
        if matches:
            best = max(matches, key=lambda c: c.score)
            self._scans = 0
            if not distance_is_known(best.distance_m):
                if (self._approaches > 0
                        and self._last_approach_start_dist is not None
                        and self._last_approach_start_dist
                        <= self.lost_after_approach_done_dist_m):
                    self._approaches = 0
                    self._last_approach_start_dist = None
                    return Action(DONE, rationale='target "%s" still visible after approach '
                                  'but depth is unknown -> stopping' % obs.target)
                self._looked = True
                if best.side == 'left':
                    return Action(TURN, turn_yaw_rad=TARGET_PROBE_TURN_RAD,
                                  rationale='target "%s" visible on the left but depth unknown; '
                                  'turn toward it' % obs.target)
                if best.side == 'right':
                    return Action(TURN, turn_yaw_rad=-TARGET_PROBE_TURN_RAD,
                                  rationale='target "%s" visible on the right but depth unknown; '
                                  'turn toward it' % obs.target)
                return Action(DRIVE_FORWARD, forward_dist_m=TARGET_PROBE_FORWARD_M,
                              rationale='target "%s" visible ahead but depth unknown; '
                              'move forward cautiously to get depth' % obs.target)
            self._looked = False
            # arrived: within the RealSense reached range -> done.
            if best.distance_m <= self.reached_dist_m:
                self._approaches = 0
                self._last_approach_start_dist = None
                return Action(DONE, rationale='target "%s" reached (%.2fm)'
                              % (obs.target, best.distance_m))
            # safety bound so a non-converging approach can't loop forever.
            if self._approaches >= self.max_approaches:
                self._approaches = 0
                self._last_approach_start_dist = None
                return Action(DONE, rationale='target "%s" approached %dx without closing in '
                              '(%.2fm) -> stopping' % (obs.target, self.max_approaches,
                                                       best.distance_m))
            # otherwise keep driving up to it.
            self._approaches += 1
            self._last_approach_start_dist = float(best.distance_m)
            return Action(DRIVE_TO_VISIBLE, mark_id=best.mark_id, arg_label=best.label,
                          rationale='target "%s" visible as mark %d (%.2fm)'
                          % (obs.target, best.mark_id, best.distance_m))
        # 2) not in view but we WERE just driving up to it -> at point-blank it overflows
        #    the frame and YOLOE drops it: treat that as arrived only if the last
        #    confirmed target range was already close. Far bounded approaches must
        #    keep searching/re-observing.
        if self._approaches > 0:
            last_dist = self._last_approach_start_dist
            self._approaches = 0
            self._last_approach_start_dist = None
            if (last_dist is not None
                    and last_dist <= self.lost_after_approach_done_dist_m):
                return Action(DONE, rationale='target reached (dropped out of frame at close range)')
        # 3) target absent, but there are semantically useful scene cues.
        semantic_action = self._semantic_explore_action(obs)
        if semantic_action is not None:
            self._looked = True
            self._scans = 0
            return semantic_action
        # 4) nothing matching in view -> one broad look before blind scanning.
        if not self._looked:
            self._looked = True
            return Action(DETECT_ALL, rationale='no target in view; detect all objects')
        # 5) rotate to bring new things into view.
        if self._scans < self.scan_turn_limit:
            self._scans += 1
            return Action(TURN, turn_yaw_rad=self.turn_step_rad,
                          rationale='blind_scan: scan-rotate %d/%d (no target)'
                          % (self._scans, self.scan_turn_limit))
        # 6) exhausted -> finish.
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
                           'distance_m': distance_for_options(c.distance_m),
                           'side': c.side,
                           'center_x_norm': round(float(c.center_x_norm), 3),
                           'source': c.source}
                          for c in obs.candidates],
        'context_marks': [{'mark_id': c.mark_id, 'label': c.label,
                           'score': round(c.score, 3),
                           'distance_m': distance_for_options(c.distance_m),
                           'side': c.side,
                           'center_x_norm': round(float(c.center_x_norm), 3),
                           'relevance': c.relevance}
                          for c in obs.context_marks],
        'notes': obs.notes_facts,
        'step_index': obs.step_index,
    }
    if obs.corridor_scan:
        opts['corridor_scan'] = obs.corridor_scan
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
    if name == 'TARGET_PROBE':
        mark_id = int(resp.get('mark_id', 0) or 0)
        repaired = target_probe_action(obs, mark_id=mark_id,
                                       rationale=str(resp.get('rationale', '') or ''))
        if repaired is not None:
            return repaired, 'OK'
        return None, 'TARGET_PROBE requires a visible target mark with unknown distance'
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
    repaired = context_mark_to_semantic_explore(act, obs)
    if repaired is not None:
        return repaired, 'OK'
    repaired = unknown_depth_target_to_probe(act, obs)
    if repaired is not None:
        return repaired, 'OK'
    repaired = edge_target_to_recenter(act, obs)
    if repaired is not None:
        return repaired, 'OK'
    repaired = context_forward_to_directional_explore(act, obs)
    if repaired is not None:
        return repaired, 'OK'
    repaired = context_turn_to_directional_explore(act, obs)
    if repaired is not None:
        return repaired, 'OK'
    repaired = visible_target_to_approach(act, obs)
    if repaired is not None:
        return repaired, 'OK'
    repaired = premature_done_to_continue(act, obs)
    if repaired is not None:
        return repaired, 'OK'
    ok, reason = validate_action(act, obs)
    if not ok:
        return None, reason
    return act, 'OK'
