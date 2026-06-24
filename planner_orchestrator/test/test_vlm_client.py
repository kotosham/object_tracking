"""Unit tests for the pluggable VLM client (Phase 4). No network."""
import json

import pytest

from planner_orchestrator.planner_logic import (
    Candidate, FrontierOpt, Observation, DRIVE_TO_VISIBLE, GO_TO_FRONTIER, TURN,
)
from planner_orchestrator.vlm_client import (
    ENV_API_KEY, ENV_BASE_URL, ENV_MODEL,
    MockVlmClient, OpenAICompatibleClient, make_client, resolve_credentials,
)


@pytest.fixture(autouse=True)
def _clear_vlm_env(monkeypatch):
    """Make the whole module hermetic: a populated operator shell (VLM_* exported)
    must not change what these tests resolve. Individual tests opt back in."""
    for name in (ENV_BASE_URL, ENV_API_KEY, ENV_MODEL):
        monkeypatch.delenv(name, raising=False)


def test_make_client_mock_by_default():
    assert isinstance(make_client(use_mock=True), MockVlmClient)


def test_make_client_mock_when_no_base_url():
    assert isinstance(make_client(use_mock=False, base_url=''), MockVlmClient)


def test_make_client_real_when_configured():
    c = make_client(use_mock=False, base_url='http://x/v1', api_key='k', model='m')
    assert isinstance(c, OpenAICompatibleClient) and c.base_url == 'http://x/v1'


def test_make_client_uses_env_when_param_blank(monkeypatch):
    monkeypatch.setenv(ENV_BASE_URL, 'http://env/v1')
    monkeypatch.setenv(ENV_API_KEY, 'envkey')
    monkeypatch.setenv(ENV_MODEL, 'envmodel')
    c = make_client()                       # no args -> all from env, real client
    assert isinstance(c, OpenAICompatibleClient)
    assert c.base_url == 'http://env/v1' and c.api_key == 'envkey' and c.model == 'envmodel'


def test_make_client_param_overrides_env(monkeypatch):
    monkeypatch.setenv(ENV_BASE_URL, 'http://env/v1')
    c = make_client(base_url='http://param/v1', api_key='k', model='m')
    assert c.base_url == 'http://param/v1'   # explicit param wins over env


def test_make_client_use_mock_forces_mock_even_with_env(monkeypatch):
    monkeypatch.setenv(ENV_BASE_URL, 'http://env/v1')
    assert isinstance(make_client(use_mock=True), MockVlmClient)


def test_resolve_credentials_strips_and_falls_back(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, '  spacey-key  ')
    base, key, model = resolve_credentials(base_url='http://x/v1')
    assert base == 'http://x/v1' and key == 'spacey-key' and model == ''


def test_mock_client_drives_loop():
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9)])
    a = MockVlmClient().plan(obs)
    assert a.kind == DRIVE_TO_VISIBLE and a.mark_id == 2


def test_build_messages_includes_image_and_options():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9)],
                      frontiers=[FrontierOpt(7, 1.0, 5.0)])
    msgs = c.build_messages(obs, image_jpeg=b'\xff\xd8jpegbytes')
    assert msgs[0]['role'] == 'system'
    user = msgs[1]['content']
    kinds = [p['type'] for p in user]
    assert 'text' in kinds and 'image_url' in kinds
    img = [p for p in user if p['type'] == 'image_url'][0]
    assert img['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert 'bus' in user[0]['text']        # target + options serialized in


def test_build_messages_text_only_when_no_image():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    msgs = c.build_messages(Observation(target='bus'), image_jpeg=None)
    assert all(p['type'] == 'text' for p in msgs[1]['content'])


def test_parse_response_valid_tool_call():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='bus', frontiers=[FrontierOpt(7, 1.0, 5.0)])
    resp = json.dumps({'choices': [{'message': {'content':
           json.dumps({'action': 'GO_TO_FRONTIER', 'frontier_id': 7, 'rationale': 'explore'})}}]})
    act = c.parse_response(resp, obs)
    assert act.kind == GO_TO_FRONTIER and act.frontier_id == 7


def test_parse_response_handles_list_content():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='x')
    inner = json.dumps({'action': 'TURN', 'turn_yaw_rad': 0.5})
    resp = json.dumps({'choices': [{'message': {'content': [{'type': 'text', 'text': inner}]}}]})
    act = c.parse_response(resp, obs)
    assert act.kind == TURN and abs(act.turn_yaw_rad - 0.5) < 1e-9


def test_parse_response_rejects_hallucinated_id():
    import pytest
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus')])
    resp = json.dumps({'choices': [{'message': {'content':
           json.dumps({'action': 'DRIVE_TO_VISIBLE', 'mark_id': 99})}}]})
    with pytest.raises(ValueError):
        c.parse_response(resp, obs)
