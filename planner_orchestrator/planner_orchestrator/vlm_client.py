"""Pluggable VLM client for the Planner Orchestrator (Phase 4).

Mock-first + pluggable. `MockVlmClient` runs the deterministic `MockPlanner`
(no network) so the whole orchestrator loop runs in sim/CI. `OpenAICompatibleClient`
talks to ANY OpenAI-compatible vision chat API (base_url + api_key + model, e.g.
Qwen3-VL) using only the Python stdlib (urllib) -- no extra deps. Both share the
same Observation -> Action contract (planner_logic), so swapping them changes
nothing upstream. The VLM is never on the reactive path: the orchestrator calls
plan() with a per-call timeout, wrapped in a circuit-breaker, and adopts the
result only at a commit point.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

# Environment variables the real VLM credentials can be supplied through. The
# operator populates these (e.g. in the launch environment or a sourced env file)
# instead of putting secrets in launch files / ROS params. An explicit non-empty
# ROS param still takes precedence; env only fills in what the param leaves blank.
ENV_BASE_URL = 'VLM_BASE_URL'
ENV_API_KEY = 'VLM_API_KEY'
ENV_MODEL = 'VLM_MODEL'

from planner_orchestrator.planner_logic import (
    Action, MockPlanner, Observation, build_vlm_options, parse_vlm_action,
)

SYSTEM_PROMPT = (
    "You are the planner for a mobile robot searching for a target object. Each turn "
    "you receive: the target; the live camera image (1st image, with numbered marks "
    "on detected objects); a top-down SLAM occupancy map (2nd image, if available); "
    "the currently visible detections (each with a mark_id, its detector class label, "
    "a confidence score, and distance_m -- the RealSense range to it in meters, or "
    "null when depth is unknown); optional context_marks (non-target objects detected "
    "for scene understanding, each with label, side=left/center/right, distance_m and "
    "semantic relevance); optional corridor_scan entries captured during the initial "
    "forward/right/left sweep; and your running notes. Use known distance_m values "
    "to size DRIVE_FORWARD and the map to avoid obstacles and unexplored dead-ends. "
    "The current visible_marks/context_marks are authoritative for this step; notes "
    "are history only and must not override the live camera. "
    "Choose exactly ONE action and reply with a "
    "single JSON object, no prose. Actions: "
    "TURN -- rotate in place; set turn_yaw_rad (radians, + = left). Use meaningful "
    "turns around 0.6 rad; do not use tiny turns because Nav2 may treat them as "
    "already reached and the camera view will not change. "
    "DRIVE_FORWARD -- drive straight; set forward_dist_m (meters, negative = backward). "
    "DRIVE_TO_VISIBLE -- drive to a detected object using the navigation stack; set "
    "mark_id, which MUST be one of the listed visible marks and must have a non-null "
    "positive distance_m. "
    "DETECT_ALL -- run the detector over the whole view and add every object and its "
    "class to your notes (recognises common everyday classes: people, vehicles, "
    "furniture, animals, household items). "
    "DONE -- the target is reached or the mission is complete. Choose DONE only when "
    "a strict visible mark with source='target' confirms the target at close range "
    "(roughly within ~0.6-0.8 m), or right after a successful DRIVE_TO_VISIBLE has "
    "brought a strict target up to point-blank range. Never choose DONE just because "
    "the target is absent, distance_m is null/unknown, or only context_marks were seen. "
    "Search behavior: if visible_marks contains a strict target mark with source='target', "
    "use target_approach and choose DRIVE_TO_VISIBLE for that mark unless it is already "
    "close enough for DONE. "
    "If the target is visible but distance_m is null/unknown, do NOT stop and do NOT "
    "use DRIVE_TO_VISIBLE yet. Choose the action TURN toward the target side "
    "(left/right), or choose DRIVE_FORWARD about 0.5-0.6 m if it is centered and "
    "the map looks free, so the target can enter reliable depth range. This is the "
    "target_probe behavior, but TARGET_PROBE is NOT an action name; start only the "
    "rationale with 'target_probe:'. "
    "Exploration priority when the target is absent: corridors/free space come first. "
    "Actively move through open free space shown on the SLAM map; do not try to drive "
    "nose-first into a cabinet, desk, shelf, or other context object just because it "
    "is semantically related to the target. Context objects are ONLY cues for choosing "
    "which corridor/free region is most relevant; they are not destinations and not "
    "objects to approach. "
    "The map is not decorative: white connected corridors/free regions are places "
    "to explore, gray unknown beyond them is useful frontier, and black cells are "
    "obstacles. Prefer DRIVE_FORWARD about 0.45-0.6 m along a clear corridor/free "
    "direction after at most one meaningful TURN used to align with that corridor "
    "or with a semantic cue. If notes show the initial_scan turns are already done "
    "and the target is still absent, stop scanning in place and choose an active "
    "corridor exploration action. Do not spend many steps rotating around the same local "
    "furniture patch. If the live camera and depth show a reliable close centered "
    "desk/table/cabinet or other furniture physically blocking the forward direction "
    "(roughly <0.8 m), or a very close left/right furniture edge that may still be "
    "inside the robot's forward swept path (roughly <0.55 m), that direction is "
    "not a corridor probe: do NOT DRIVE_FORWARD into the blocker just because the "
    "map looks white there. If no such close forward/swept-path blocker is visible "
    "and the SLAM map shows connected free space ahead, "
    "DRIVE_FORWARD remains the preferred corridor-exploration action. "
    "When the forward direction is blocked, TURN toward the clearest adjacent "
    "left/right corridor instead. "
    "After the initial forward/right/left sweep, compare corridor_scan entries: "
    "prefer a real free/unknown corridor on the SLAM map first; if multiple corridors "
    "are similarly open, prefer the corridor whose recorded context objects are most "
    "semantically relevant to the target. Keep exploring the chosen corridor with "
    "short DRIVE_FORWARD steps until the target appears, navigation fails, or a close "
    "obstacle blocks the path. "
    "If the target is absent but context_marks contains semantically relevant objects "
    "near one side of the image (for example office furniture while searching for an "
    "office chair), use semantic_explore: use those objects only to pick an open "
    "corridor/free region on the map. Choose one meaningful TURN to align with that "
    "corridor if needed, then DRIVE_FORWARD through the free corridor. Do not "
    "DRIVE_TO_VISIBLE a context object and do not drive nose-first into centered "
    "furniture; move through free space beside/beyond it. Start the rationale with "
    "'semantic_explore:'. Do not keep rotating in place around the same local furniture: "
    "after one meaningful semantic_explore TURN, prefer DRIVE_FORWARD about 0.45-0.6 m "
    "into a clear corridor/free area to change viewpoint. Only suppress that "
    "forward probe when a close forward/swept-path blocker is visible in the "
    "live camera/depth. "
    "Target-like context detections in context_marks are still "
    "search cues only: use them to choose a corridor or viewpoint, but do not "
    "DRIVE_TO_VISIBLE to their mark_id and do not DONE from context alone. "
    "If neither target nor useful context is visible, use blind_scan "
    "(usually DETECT_ALL once, then TURN); start the rationale with 'blind_scan:'. "
    "DRIVE_TO_VISIBLE may reference only visible_marks, never context_marks. "
    "Never invent a mark_id that is not listed, and never output map coordinates. "
    "JSON schema: "
    '{"action": str, "turn_yaw_rad": float, "forward_dist_m": float, "mark_id": int, '
    '"arg_label": str, "rationale": str}.'
)


TARGET_RESOLUTION_PROMPT = (
    "You normalize a user's robot-search target before perception starts. "
    "Decide whether the user query is already a concrete object name, a concrete "
    "object name with visual attributes, or a semantic/riddle-like description. "
    "If it is already a direct object name, preserve it. For example, "
    "'office chair' stays 'office chair'. If it has useful visual attributes, "
    "keep them in detection_query, but set canonical_target to the basic object "
    "name. For example, 'black office chair' -> canonical_target 'office chair', "
    "detection_query 'black office chair'. If it is a riddle/metaphor, infer the "
    "most likely common physical object and express it as a short English "
    "open-vocabulary detector phrase. Do not choose robot actions and do not "
    "invent scene facts. Return one JSON object only with this schema: "
    '{"canonical_target": str, "detection_query": str, "query_type": str, '
    '"aliases": [str], "reason": str}. '
    "Allowed query_type values: direct_object_name, object_with_attributes, "
    "semantic_description, ambiguous. If ambiguous, still provide the best short "
    "detector phrase."
)


@dataclass(frozen=True)
class TargetResolution:
    """Normalized mission target.

    raw_query is what the operator published. canonical_target is what the
    planner reasons about. detection_query is what the open-vocabulary detector
    should receive; it may keep useful visual attributes.
    """

    raw_query: str
    canonical_target: str
    detection_query: str
    query_type: str = 'direct_object_name'
    aliases: tuple = ()
    reason: str = ''

    @staticmethod
    def _clean_text(value, fallback, max_len=120):
        text = str(value or '').strip().strip('"').strip("'")
        if not text:
            text = str(fallback or '').strip()
        text = ' '.join(text.split())
        return text[:max_len].strip()

    @classmethod
    def passthrough(cls, raw_query, query_type='direct_object_name', reason='passthrough'):
        raw = cls._clean_text(raw_query, '')
        return cls(raw_query=raw, canonical_target=raw, detection_query=raw,
                   query_type=query_type, aliases=tuple(), reason=reason)

    @classmethod
    def from_json(cls, raw_query, data):
        if not isinstance(data, dict):
            return cls.passthrough(raw_query, query_type='resolver_invalid',
                                   reason='resolver returned non-object JSON')
        raw = cls._clean_text(raw_query, '')
        canonical = cls._clean_text(data.get('canonical_target'), raw)
        detection = cls._clean_text(data.get('detection_query'), canonical)
        query_type = cls._clean_text(data.get('query_type'), 'direct_object_name', 40)
        allowed = {
            'direct_object_name', 'object_with_attributes',
            'semantic_description', 'ambiguous',
        }
        if query_type not in allowed:
            query_type = 'ambiguous'
        aliases = data.get('aliases') or []
        if not isinstance(aliases, list):
            aliases = []
        clean_aliases = []
        for item in aliases[:8]:
            alias = cls._clean_text(item, '', 80)
            if alias and alias not in clean_aliases and alias not in (canonical, detection):
                clean_aliases.append(alias)
        reason = cls._clean_text(data.get('reason'), '', 240)
        return cls(raw_query=raw, canonical_target=canonical, detection_query=detection,
                   query_type=query_type, aliases=tuple(clean_aliases), reason=reason)


class VlmClient:
    """Observation (+ optional camera JPEG + optional map JPEG) -> Action. Raises on
    failure/timeout."""

    def plan(self, obs: Observation, image_jpeg: Optional[bytes] = None,
             map_jpeg: Optional[bytes] = None) -> Action:
        raise NotImplementedError

    def resolve_target_query(self, raw_query: str) -> TargetResolution:
        """Map a raw operator query/riddle to a concrete detector target."""
        return TargetResolution.passthrough(raw_query)

    def plan_sequence(self, obs: Observation, image_jpeg: Optional[bytes] = None,
                      map_jpeg: Optional[bytes] = None, n: int = 1) -> list:
        """A short plan of up to n atomic actions (replan-every-N). Default: one
        action (reactive); the real VLM client may override to plan n steps ahead
        from a single observation. The orchestrator executes the returned list,
        then replans with a fresh observation."""
        return [self.plan(obs, image_jpeg, map_jpeg)]


class MockVlmClient(VlmClient):
    """Deterministic stand-in (no network) -- drives the whole loop in sim/CI."""

    def __init__(self, **mock_kwargs):
        self._mp = MockPlanner(**mock_kwargs)

    def plan(self, obs: Observation, image_jpeg: Optional[bytes] = None,
             map_jpeg: Optional[bytes] = None) -> Action:
        return self._mp.plan(obs)


class OpenAICompatibleClient(VlmClient):
    """Calls an OpenAI-compatible /chat/completions vision endpoint (stdlib only)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 8.0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    @staticmethod
    def _image_part(jpeg: bytes) -> dict:
        b64 = base64.b64encode(jpeg).decode('ascii')
        return {'type': 'image_url',
                'image_url': {'url': 'data:image/jpeg;base64,' + b64}}

    def build_messages(self, obs: Observation, image_jpeg: Optional[bytes],
                       map_jpeg: Optional[bytes] = None) -> list:
        opts = build_vlm_options(obs)
        text = ('Target: %s\nOptions (JSON):\n%s\nReply with ONE JSON action.'
                % (obs.target, json.dumps(opts)))
        content = [{'type': 'text', 'text': text}]
        if image_jpeg:                       # 1st image: live camera (Set-of-Mark)
            content.append({'type': 'text', 'text': 'Live camera (numbered marks):'})
            content.append(self._image_part(image_jpeg))
        if map_jpeg:                         # 2nd image: top-down SLAM map
            content.append({'type': 'text', 'text': 'Top-down SLAM map:'})
            content.append(self._image_part(map_jpeg))
        return [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': content}]

    def parse_response(self, resp_text: str, obs: Observation) -> Action:
        """Extract the tool-call JSON from a chat-completions response + validate."""
        data = json.loads(resp_text)
        content = data['choices'][0]['message']['content']
        if isinstance(content, list):   # some servers return content as parts
            content = ''.join(p.get('text', '') for p in content if isinstance(p, dict))
        action_json = json.loads(content)
        act, reason = parse_vlm_action(action_json, obs)
        if act is None:
            raise ValueError('VLM action rejected: %s' % reason)
        return act

    def _post(self, body: dict) -> str:
        req = urllib.request.Request(
            self.base_url + '/chat/completions',
            data=json.dumps(body).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + self.api_key},
            method='POST')
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return r.read().decode('utf-8')

    def parse_target_resolution(self, resp_text: str, raw_query: str) -> TargetResolution:
        """Extract and sanitize the target-normalization JSON."""
        data = json.loads(resp_text)
        content = data['choices'][0]['message']['content']
        if isinstance(content, list):
            content = ''.join(p.get('text', '') for p in content if isinstance(p, dict))
        return TargetResolution.from_json(raw_query, json.loads(content))

    def resolve_target_query(self, raw_query: str) -> TargetResolution:
        raw = TargetResolution._clean_text(raw_query, '')
        body = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': TARGET_RESOLUTION_PROMPT},
                {'role': 'user', 'content': 'User target query: %s' % raw},
            ],
            'temperature': 0,
            'max_tokens': 256,
            'response_format': {'type': 'json_object'},
        }
        return self.parse_target_resolution(self._post(body), raw)

    def plan(self, obs: Observation, image_jpeg: Optional[bytes] = None,
             map_jpeg: Optional[bytes] = None) -> Action:
        body = {
            'model': self.model,
            'messages': self.build_messages(obs, image_jpeg, map_jpeg),
            'temperature': 0,
            'max_tokens': 256,
            'response_format': {'type': 'json_object'},
        }
        return self.parse_response(self._post(body), obs)


def resolve_credentials(base_url: str = '', api_key: str = '', model: str = ''):
    """Fill blank credentials from the environment (VLM_BASE_URL/VLM_API_KEY/
    VLM_MODEL). A non-empty argument (e.g. an explicit ROS param) always wins;
    env only supplies what the caller left empty. Returns (base_url, api_key,
    model). NOTE: secrets are read here at runtime only -- they are never logged."""
    base_url = base_url or os.environ.get(ENV_BASE_URL, '')
    api_key = api_key or os.environ.get(ENV_API_KEY, '')
    model = model or os.environ.get(ENV_MODEL, '')
    return base_url.strip(), api_key.strip(), model.strip()


def make_client(use_mock: bool = False, base_url: str = '', api_key: str = '',
                model: str = '', timeout_s: float = 8.0, **mock_kwargs) -> VlmClient:
    """Pluggable factory. Credentials come from the given args or, if blank, the
    environment (see resolve_credentials). Returns the real OpenAI-compatible
    client when a base_url is available and mock is not forced; otherwise the
    deterministic mock (so with no creds anywhere the loop still runs offline)."""
    base_url, api_key, model = resolve_credentials(base_url, api_key, model)
    if use_mock or not base_url:
        return MockVlmClient(**mock_kwargs)
    return OpenAICompatibleClient(base_url, api_key, model, timeout_s)
