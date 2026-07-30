"""Unit tests for the pluggable VLM client (Phase 4). No network."""
import json

import pytest

from planner_orchestrator.planner_logic import (
    Candidate, Observation, DRIVE_TO_VISIBLE, DETECT_ALL, TURN,
)
from planner_orchestrator.vlm_client import (
    ENV_API_KEY, ENV_BASE_URL, ENV_MODEL, FEWSHOT, SYSTEM_PROMPT,
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
    user = msgs[-1]['content']
    kinds = [p['type'] for p in user]
    assert 'text' in kinds and 'image_url' in kinds
    img = [p for p in user if p['type'] == 'image_url'][0]
    assert img['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert 'bus' in user[0]['text']        # target + options serialized in


def test_build_messages_text_only_when_no_image():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    msgs = c.build_messages(Observation(target='bus'), image_jpeg=None)
    assert all(p['type'] == 'text' for p in msgs[-1]['content'])


def test_build_messages_attaches_map_as_second_image():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=2.0)],
                      map_text='occupancy map')
    msgs = c.build_messages(obs, image_jpeg=b'\xff\xd8camera', map_jpeg=b'\xff\xd8map')
    images = [p for p in msgs[-1]['content'] if p['type'] == 'image_url']
    assert len(images) == 2                      # camera + map
    assert 'occupancy map' in msgs[-1]['content'][0]['text']   # map described in opts


def test_system_prompt_states_the_role_and_every_input():
    """Модель должна знать, ЧЕМ она управляет и ЧТО ей дают. Без этого она
    получала три источника данных и не понимала, что с ними делать: смотрела на
    карту SLAM, не зная, что серое — единственное место, где может быть
    ненайденное, и на free_ahead_m, не зная, что это измерение, а не оценка."""
    p = SYSTEM_PROMPT.lower()
    assert 'wheeled robot' in p and 'find the target' in p
    for source in ('camera image', 'visible_marks', 'depth camera',
                   'free_ahead_m', 'objects_found', 'notes'):
        assert source.lower() in p, source
    # смысл цветов карты, а не только сами цвета
    assert 'gray = never seen' in p and 'already seen' in p
    # задний ход и потолок поворотов подряд — то, чего в промпте не было
    assert 'no rear sensor' in p and 'two turns in a row' in p
    # ReAct: думать раньше, чем действовать, и порядок ключей это закрепляет
    assert p.index('"think"') < p.index('"action"')


def test_fewshot_examples_use_the_real_request_shape():
    """Пример в другой форме учит отвечать на запрос, которого не будет.
    Проверяем, что реплика примера собирается той же функцией и той же формы,
    что настоящая."""
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    msgs = c.build_messages(Observation(target='bus'), None)
    real = msgs[-1]['content'][0]['text']
    shots = [m['content'] for m in msgs if m['role'] == 'user'][:-1]
    assert len(shots) == len(FEWSHOT) >= 4
    for shot in shots:
        assert shot.startswith('Target: ')
        assert 'Options (JSON):' in shot
        assert shot.splitlines()[-1] == real.splitlines()[-1]   # тот же хвост
        json.loads(shot.splitlines()[2])                        # опции — валидный JSON
    replies = [m['content'] for m in msgs if m['role'] == 'assistant']
    for reply in replies:
        parsed = json.loads(reply)
        assert 'think' in parsed and 'action' in parsed
        assert list(parsed).index('think') == 0                 # think идёт ПЕРВЫМ


def test_fewshot_covers_the_failure_modes_seen_in_runs():
    """Каждый пример существует из-за конкретной наблюдавшейся поломки: упёрлась
    в стену, крутится на месте, получила отказ исполнителя, ищет не в той
    комнате. Проверяем, что все четыре ответа в наборе есть."""
    replies = [json.loads(r) for _, r in FEWSHOT]
    opts = [o for o, _ in FEWSHOT]
    wall = [r for o, r in zip(opts, replies) if o.get('free_ahead_m', 9) < 0.5]
    assert any(r['action'] == 'TURN' for r in wall)          # у стены — поворот
    spin = [r for o, r in zip(opts, replies)
            if sum('TURN' in n for n in o.get('notes', [])) >= 2]
    assert spin and all(r['action'] == 'DRIVE_FORWARD' for r in spin)
    blocked = [r for o, r in zip(opts, replies)
               if any('blocked' in n for n in o.get('notes', []))]
    assert blocked and all(r['forward_dist_m'] < 0 or r['action'] == 'TURN'
                           for r in blocked)                 # отказ -> смена приёма
    assert any(o.get('objects_found') for o in opts)         # память о найденном


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


def _system_count(messages):
    """Сколько системных сообщений в диалоге.

    Проверять размер списка целиком больше нельзя: между системным промптом и
    реальным запросом лежат few-shot пары, и их число — деталь промпта, а не
    контракт. Значимо ровно одно: добавился ли SEQUENCE_PROMPT.
    """
    return sum(1 for m in messages if m.get('role') == 'system')


def test_build_messages_n1_has_no_sequence_override():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    obs = Observation(target='bus')
    assert c.build_messages(obs, None, None, n=1) == c.build_messages(obs, None)


def test_build_messages_n_gt_1_adds_sequence_override():
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    msgs = c.build_messages(Observation(target='bus'), None, None, n=3)
    assert _system_count(msgs) == 2                # базовый + SEQUENCE_PROMPT
    assert msgs[-1]['role'] == 'user'
    systems = [m['content'] for m in msgs if m['role'] == 'system']
    assert any('"actions"' in s for s in systems)  # переопределение формата пришло


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
    assert seen['max_tokens'] == 256
    assert _system_count(seen['messages']) == 1   # без SEQUENCE_PROMPT


def test_plan_sequence_scales_token_budget(monkeypatch):
    c = OpenAICompatibleClient('http://x/v1', 'k', 'qwen')
    seen = {}

    def fake_post(body):
        seen.update(body)
        return _seq_resp({'actions': [{'action': 'TURN', 'turn_yaw_rad': 0.1}]})

    monkeypatch.setattr(c, '_post', fake_post)
    c.plan_sequence(Observation(target='bus'), None, None, n=4)
    assert seen['max_tokens'] == 256 * 4
    assert _system_count(seen['messages']) == 2   # + SEQUENCE_PROMPT


def test_mock_client_ignores_n(monkeypatch):
    """MockPlanner вперёд не планирует — пачка обязана остаться из одного шага."""
    acts = MockVlmClient().plan_sequence(
        Observation(target='bus', candidates=[Candidate(2, 'bus', 0.9, distance_m=2.0)]),
        None, None, n=5)
    assert len(acts) == 1
