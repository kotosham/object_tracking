"""Unit tests for the pluggable VLM client (Phase 4). No network."""
import json

import pytest

from planner_orchestrator.planner_logic import (
    Candidate, Observation, DRIVE_TO_VISIBLE, DETECT_ALL, TURN,
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
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=2.0)])
    a = MockVlmClient().plan(obs)
    assert a.kind == DRIVE_TO_VISIBLE and a.mark_id == 2


def test_build_messages_includes_image_and_options():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=2.0)])
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


def test_build_messages_attaches_map_as_second_image():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=2.0)],
                      map_text='occupancy map')
    msgs = c.build_messages(obs, image_jpeg=b'\xff\xd8camera', map_jpeg=b'\xff\xd8map')
    images = [p for p in msgs[1]['content'] if p['type'] == 'image_url']
    assert len(images) == 2                      # camera + map
    assert 'occupancy map' in msgs[1]['content'][0]['text']   # map described in opts


def test_parse_response_valid_tool_call():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=2.0)])
    resp = json.dumps({'choices': [{'message': {'content':
           json.dumps({'action': 'DRIVE_TO_VISIBLE', 'mark_id': 2, 'rationale': 'approach'})}}]})
    act = c.parse_response(resp, obs)
    assert act.kind == DRIVE_TO_VISIBLE and act.mark_id == 2


def test_parse_response_detect_all():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    resp = json.dumps({'choices': [{'message': {'content':
           json.dumps({'action': 'DETECT_ALL', 'rationale': 'survey'})}}]})
    act = c.parse_response(resp, Observation(target='bus'))
    assert act.kind == DETECT_ALL


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


# --- replan_every_n > 1: пачка действий за один запрос ----------------------
# Дефолт replan_every_n=1, поэтому путь n=1 обязан остаться байт-в-байт прежним:
# первый тест ниже сторожит именно это (без него «оптимизация» тихо поменяла бы
# промпт и сделала бы прогоны до/после несравнимыми).

def _seq_resp(payload):
    return json.dumps({'choices': [{'message': {'content': json.dumps(payload)}}]})


def test_build_messages_n1_has_no_sequence_override():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='bus')
    assert c.build_messages(obs, None, None, n=1) == c.build_messages(obs, None)


def test_build_messages_n_gt_1_adds_sequence_override():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    msgs = c.build_messages(Observation(target='bus'), None, None, n=3)
    assert len(msgs) == 3 and msgs[1]['role'] == 'system'
    assert 'actions' in msgs[1]['content']


def test_parse_sequence_returns_actions_in_order():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    acts = c.parse_sequence_response(_seq_resp({'actions': [
        {'action': 'TURN', 'turn_yaw_rad': 0.5},
        {'action': 'DETECT_ALL'},
    ]}), Observation(target='bus'), 3)
    assert [a.kind for a in acts] == [TURN, DETECT_ALL]


def test_parse_sequence_truncates_to_n():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    acts = c.parse_sequence_response(_seq_resp({'actions': [
        {'action': 'TURN', 'turn_yaw_rad': 0.1},
        {'action': 'TURN', 'turn_yaw_rad': 0.2},
        {'action': 'TURN', 'turn_yaw_rad': 0.3},
    ]}), Observation(target='bus'), 2)
    assert len(acts) == 2


def test_parse_sequence_stops_after_done():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    acts = c.parse_sequence_response(_seq_resp({'actions': [
        {'action': 'DONE'}, {'action': 'TURN', 'turn_yaw_rad': 0.1},
    ]}), Observation(target='bus'), 3)
    assert len(acts) == 1 and acts[0].name == 'DONE'


def test_parse_sequence_accepts_single_action_object():
    """Модель вправе ответить одиночным действием старой схемы — это не отказ."""
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    acts = c.parse_sequence_response(
        _seq_resp({'action': 'TURN', 'turn_yaw_rad': 0.4}), Observation(target='bus'), 3)
    assert [a.kind for a in acts] == [TURN]


def test_parse_sequence_keeps_valid_prefix_on_bad_tail():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=2.0)])
    acts = c.parse_sequence_response(_seq_resp({'actions': [
        {'action': 'TURN', 'turn_yaw_rad': 0.3},
        {'action': 'DRIVE_TO_VISIBLE', 'mark_id': 99},
    ]}), obs, 3)
    assert [a.kind for a in acts] == [TURN]


def test_parse_sequence_raises_when_first_action_invalid():
    """Первое действие невалидно -> отказ, иначе circuit-breaker его не увидит."""
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus')])
    with pytest.raises(ValueError):
        c.parse_sequence_response(
            _seq_resp({'actions': [{'action': 'DRIVE_TO_VISIBLE', 'mark_id': 99}]}), obs, 3)


def test_parse_sequence_raises_on_empty_list():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    with pytest.raises(ValueError):
        c.parse_sequence_response(_seq_resp({'actions': []}), Observation(target='bus'), 3)


def test_plan_sequence_n1_uses_single_action_path(monkeypatch):
    """n=1 не должен ходить в ветку пачки: тот же body, что и у plan()."""
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    seen = {}

    def fake_post(body):
        seen.update(body)
        return _seq_resp({'action': 'TURN', 'turn_yaw_rad': 0.5})

    monkeypatch.setattr(c, '_post', fake_post)
    acts = c.plan_sequence(Observation(target='bus'), None, None, n=1)
    assert [a.kind for a in acts] == [TURN]
    assert seen['max_tokens'] == 256 and len(seen['messages']) == 2


def test_plan_sequence_scales_token_budget(monkeypatch):
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    seen = {}

    def fake_post(body):
        seen.update(body)
        return _seq_resp({'actions': [{'action': 'TURN', 'turn_yaw_rad': 0.1}]})

    monkeypatch.setattr(c, '_post', fake_post)
    c.plan_sequence(Observation(target='bus'), None, None, n=4)
    assert seen['max_tokens'] == 256 * 4 and len(seen['messages']) == 3


def test_mock_client_ignores_n(monkeypatch):
    """MockPlanner вперёд не планирует — пачка обязана остаться из одного шага."""
    acts = MockVlmClient().plan_sequence(
        Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=2.0)]),
        None, None, n=5)
    assert len(acts) == 1
