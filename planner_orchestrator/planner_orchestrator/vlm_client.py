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
    "a confidence score, and distance_m -- the RealSense range to it in meters); and "
    "your running notes. Use distance_m to size DRIVE_FORWARD and the map to avoid "
    "obstacles and unexplored dead-ends. Choose exactly ONE action and reply with a "
    "single JSON object, no prose. Actions: "
    "TURN -- rotate in place; set turn_yaw_rad (radians, + = left). "
    "DRIVE_FORWARD -- drive straight; set forward_dist_m (meters, negative = backward). "
    "DRIVE_TO_VISIBLE -- drive to a detected object using the navigation stack; set "
    "mark_id, which MUST be one of the listed visible marks. "
    "DETECT_ALL -- run the detector over the whole view and add every object and its "
    "class to your notes (recognises common everyday classes: people, vehicles, "
    "furniture, animals, household items). "
    "DONE -- the target is reached or the mission is complete. Choose DONE once the "
    "target's distance_m shows you are close (roughly within ~0.6 m, the robot's "
    "approach standoff), or right after a DRIVE_TO_VISIBLE has brought you up to it and "
    "it now fills the camera / drops out of the detections at point-blank range. "
    "Never invent a mark_id that is not listed, and never output map coordinates. "
    "JSON schema: "
    '{"action": str, "turn_yaw_rad": float, "forward_dist_m": float, "mark_id": int, '
    '"arg_label": str, "rationale": str}.'
)


class VlmClient:
    """Observation (+ optional camera JPEG + optional map JPEG) -> Action. Raises on
    failure/timeout."""

    def plan(self, obs: Observation, image_jpeg: Optional[bytes] = None,
             map_jpeg: Optional[bytes] = None) -> Action:
        raise NotImplementedError

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
