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
import urllib.request
from typing import Optional

from planner_orchestrator.planner_logic import (
    Action, MockPlanner, Observation, build_vlm_options, parse_vlm_action,
)

SYSTEM_PROMPT = (
    "You are the high-level planner for a mobile robot looking for a target object. "
    "Each turn you receive the target, a camera image, and the REAL options: the "
    "visible detections (each with a mark_id) and the frontier options (each with an "
    "id). Choose exactly ONE atomic action and reply with a single JSON object, no "
    "prose. Allowed actions: TURN (rotate in place, set turn_yaw_rad), DRIVE_FORWARD "
    "(set forward_dist_m), DRIVE_TO_VISIBLE (set mark_id -- MUST be one of the listed "
    "visible marks), GO_TO_FRONTIER (set frontier_id -- MUST be one of the listed "
    "frontiers), GET_OBSERVATION, DONE, STOP. NEVER invent a mark_id or frontier_id "
    "that is not in the options, and NEVER output map coordinates. JSON schema: "
    '{"action": str, "turn_yaw_rad": float, "forward_dist_m": float, "mark_id": int, '
    '"frontier_id": int, "arg_label": str, "rationale": str}.'
)


class VlmClient:
    """Observation (+ optional JPEG) -> Action. Raises on failure/timeout."""

    def plan(self, obs: Observation, image_jpeg: Optional[bytes] = None) -> Action:
        raise NotImplementedError

    def plan_sequence(self, obs: Observation, image_jpeg: Optional[bytes] = None,
                      n: int = 1) -> list:
        """A short plan of up to n atomic actions (replan-every-N). Default: one
        action (reactive); the real VLM client may override to plan n steps ahead
        from a single observation. The orchestrator executes the returned list,
        then replans with a fresh observation."""
        return [self.plan(obs, image_jpeg)]


class MockVlmClient(VlmClient):
    """Deterministic stand-in (no network) -- drives the whole loop in sim/CI."""

    def __init__(self, **mock_kwargs):
        self._mp = MockPlanner(**mock_kwargs)

    def plan(self, obs: Observation, image_jpeg: Optional[bytes] = None) -> Action:
        return self._mp.plan(obs)


class OpenAICompatibleClient(VlmClient):
    """Calls an OpenAI-compatible /chat/completions vision endpoint (stdlib only)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 8.0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def build_messages(self, obs: Observation, image_jpeg: Optional[bytes]) -> list:
        opts = build_vlm_options(obs)
        text = ('Target: %s\nOptions (JSON):\n%s\nReply with ONE JSON action.'
                % (obs.target, json.dumps(opts)))
        content = [{'type': 'text', 'text': text}]
        if image_jpeg:
            b64 = base64.b64encode(image_jpeg).decode('ascii')
            content.append({'type': 'image_url',
                            'image_url': {'url': 'data:image/jpeg;base64,' + b64}})
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

    def plan(self, obs: Observation, image_jpeg: Optional[bytes] = None) -> Action:
        body = {
            'model': self.model,
            'messages': self.build_messages(obs, image_jpeg),
            'temperature': 0,
            'max_tokens': 256,
            'response_format': {'type': 'json_object'},
        }
        return self.parse_response(self._post(body), obs)


def make_client(use_mock: bool = True, base_url: str = '', api_key: str = '',
                model: str = '', timeout_s: float = 8.0, **mock_kwargs) -> VlmClient:
    """Pluggable factory: mock unless a real base_url is configured."""
    if use_mock or not base_url:
        return MockVlmClient(**mock_kwargs)
    return OpenAICompatibleClient(base_url, api_key, model, timeout_s)
