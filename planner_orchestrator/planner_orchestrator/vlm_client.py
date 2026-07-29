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
    DONE, Action, MockPlanner, Observation, build_vlm_options, parse_vlm_action,
)

# Промпт намеренно короткий и построен как ReAct: сначала "think", потом
# действие. Порядок ключей в JSON — часть замысла, а не оформление: модель
# генерирует слева направо, поэтому "think" ПЕРЕД "action" заставляет её сначала
# рассудить и лишь затем выбрать, а обратный порядок превращает объяснение в
# оправдание уже сделанного выбора.
#
# Прежняя версия была вдвое длиннее и при этом умалчивала ровно о том, что робот
# делал неправильно: не объясняла, когда поворачивать, зачем задний ход и какой
# шаг вперёд имеет смысл. Отсюда наблюдавшееся поведение — десятки DRIVE_FORWARD
# подряд в упор в стену, без единого TURN.
SYSTEM_PROMPT = (
    "You drive a small wheeled robot through rooms to find one target object.\n"
    "Each turn you get: the target, the camera view (numbered marks on detections), "
    "a top-down map, the list of visible marks, and your notes.\n"
    "\n"
    "ACTIONS\n"
    "TURN turn_yaw_rad      rotate in place. + = left, - = right. 1.57 = 90 deg.\n"
    "DRIVE_FORWARD forward_dist_m   drive straight. Use 0.3..1.5 m.\n"
    "DRIVE_TO_VISIBLE mark_id       let the nav stack drive to a listed mark.\n"
    "DETECT_ALL             name every object in view and store it in notes.\n"
    "DONE                   target reached.\n"
    "\n"
    "RULES\n"
    "- Wall or obstacle ahead: do NOT drive forward. TURN (1.57 or -1.57) and look.\n"
    "- Target not visible: explore. TURN to scan, DRIVE_FORWARD into open space, "
    "DETECT_ALL to record what is around. Note where each room and object is; that map "
    "of the place is how you find the target later.\n"
    "- Forward steps below 0.25 m do nothing at all -- the robot will not move. Never "
    "ask for them. If only a few centimetres are free ahead, you are AT a wall: turn.\n"
    "- Wedged with a wall in front and no room to turn? Back off first: negative "
    "forward_dist_m, -0.3 to -0.5, then turn. That is what reverse is for. Never "
    "reverse to explore -- there is no rear sensor, you are blind backwards, and the "
    "step is capped at 0.5 m.\n"
    "- Turning does not move you. After AT MOST two turns in a row you MUST drive: "
    "pick the most open direction you have just seen and DRIVE_FORWARD. Spinning in "
    "place forever fails the mission exactly as surely as ramming a wall.\n"
    "- Read your notes: they list what you already did. Repeating the same action that "
    "changed nothing is the main way this mission fails. Alternate look and move -- "
    "turn to see, then drive to get there.\n"
    "- DRIVE_TO_VISIBLE needs a mark_id from the list with a non-null distance_m. Never "
    "invent one. Never output map coordinates.\n"
    "- DONE only when the target's distance_m is about 0.6 m or less, or right after "
    "DRIVE_TO_VISIBLE brought you up to it. Never DONE on unknown distance.\n"
    "\n"
    "REPLY\n"
    'One JSON object, no prose, "think" first (one short sentence: what you see, what '
    "you will do):\n"
    '{"think": str, "action": str, "turn_yaw_rad": float, "forward_dist_m": float, '
    '"mark_id": int, "arg_label": str}'
)

# Few-shot. Три примера покрывают три состояния, в которых модель ошибалась:
# упёрлась в стену, ничего не видит, цель видна и близко. Пары «пользователь ->
# ответ» кладутся в диалог как обычные сообщения — так модель видит не описание
# формата, а его употребление.
FEWSHOT = (
    ('Target: chair\nVisible marks: none\nNotes: wall ahead at 0.4 m',
     '{"think": "Wall right in front, nothing visible, so driving forward is '
     'pointless -- turn left and scan.", "action": "TURN", "turn_yaw_rad": 1.57, '
     '"forward_dist_m": 0.0, "mark_id": 0, "arg_label": ""}'),
    ('Target: chair\nVisible marks: none\nNotes: corridor open ahead, rooms unexplored',
     '{"think": "Open corridor and no detections here; move up it and look for new '
     'objects.", "action": "DRIVE_FORWARD", "turn_yaw_rad": 0.0, '
     '"forward_dist_m": 1.0, "mark_id": 0, "arg_label": ""}'),
    ('Target: chair\nVisible marks: none\nNotes: TURN -> ok; TURN -> ok',
     '{"think": "Two turns already and still nothing -- turning again just spins me. '
     'Drive into the open space I saw.", "action": "DRIVE_FORWARD", '
     '"turn_yaw_rad": 0.0, "forward_dist_m": 1.0, "mark_id": 0, "arg_label": ""}'),
    ('Target: chair\nVisible marks: [{"mark_id": 2, "label": "chair", '
     '"distance_m": 2.4}]\nNotes: -',
     '{"think": "The chair is mark 2 at 2.4 m -- let the nav stack take me to it.", '
     '"action": "DRIVE_TO_VISIBLE", "turn_yaw_rad": 0.0, "forward_dist_m": 0.0, '
     '"mark_id": 2, "arg_label": "chair"}'),
)

# Добавка к системному промпту при replan_every_n > 1. Идёт ОТДЕЛЬНЫМ сообщением
# после SYSTEM_PROMPT и явно переопределяет его требование «ровно одно действие»:
# переписывать сам SYSTEM_PROMPT нельзя, иначе путь n=1 (дефолт) перестал бы быть
# байт-в-байт прежним, и сравнивать прогоны до/после стало бы нечестно.
#
# Честное предупреждение модели про слепое исполнение здесь обязательно: свежие
# кадр, карта и список меток есть только у ПЕРВОГО действия пачки, остальные
# выполняются без новой перцепции. Без этой строки модель охотно планирует
# «доехать до метки 2, затем повернуть к метке 3», хотя метки к тому моменту уже
# пересчитаны и mark_id значит совсем другое.
SEQUENCE_PROMPT = (
    "REPLY FORMAT OVERRIDE (replaces the single-object reply rule above): reply "
    'with a single JSON object {"actions": [...]} whose "actions" is a list of 1 to '
    '%d actions, in execution order, each object using exactly the schema above. '
    'Only the FIRST action is chosen with fresh perception: the rest are executed '
    'blind, without a new camera image, map or detection list, so mark_id values may '
    'be stale by then. Therefore: put DRIVE_TO_VISIBLE only as the FIRST action, '
    'prefer short conservative sequences of TURN/DRIVE_FORWARD, and emit fewer than '
    '%d actions whenever the situation is uncertain. Anything after DONE is ignored.'
)


class VlmClient:
    """Observation (+ optional camera JPEG + optional map JPEG) -> Action. Raises on
    failure/timeout."""

    def plan(self, obs: Observation, image_jpeg: Optional[bytes] = None,
             map_jpeg: Optional[bytes] = None) -> Action:
        raise NotImplementedError

    def plan_sequence(self, obs: Observation, image_jpeg: Optional[bytes] = None,
                      map_jpeg: Optional[bytes] = None, n: int = 1) -> list:
        """A short plan of up to n atomic actions (replan-every-N).

        Базовая реализация ИГНОРИРУЕТ n и всегда возвращает одно действие — это
        честно для MockVlmClient, чей MockPlanner вперёд не планирует.
        OpenAICompatibleClient переопределяет метод и реально просит у модели до n
        действий за один запрос; см. его plan_sequence."""
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
                       map_jpeg: Optional[bytes] = None, n: int = 1) -> list:
        opts = build_vlm_options(obs)
        tail = ('Reply with ONE JSON action.' if n <= 1 else
                'Reply with a JSON object {"actions": [...]} of up to %d actions.' % n)
        text = ('Target: %s\nOptions (JSON):\n%s\n%s'
                % (obs.target, json.dumps(opts), tail))
        content = [{'type': 'text', 'text': text}]
        if image_jpeg:                       # 1st image: live camera (Set-of-Mark)
            content.append({'type': 'text', 'text': 'Live camera (numbered marks):'})
            content.append(self._image_part(image_jpeg))
        if map_jpeg:                         # 2nd image: top-down SLAM map
            content.append({'type': 'text', 'text': 'Top-down SLAM map:'})
            content.append(self._image_part(map_jpeg))
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        # Примеры идут ПОСЛЕ системного сообщения и ДО реального запроса, как
        # обычный диалог: так модель видит формат в употреблении, а не в описании.
        for shot_user, shot_reply in FEWSHOT:
            messages.append({'role': 'user', 'content': shot_user})
            messages.append({'role': 'assistant', 'content': shot_reply})
        if n > 1:
            messages.append({'role': 'system',
                             'content': SEQUENCE_PROMPT % (n, n)})
        messages.append({'role': 'user', 'content': content})
        return messages

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

    def parse_sequence_response(self, resp_text: str, obs: Observation,
                                n: int) -> list:
        """Ответ вида {"actions":[...]} -> список валидных Action.

        Невалидное действие НЕ роняет всю пачку: список обрезается по последнему
        валидному. Причина в цене — пачка стоит один запрос к модели, и выкидывать
        три корректных шага из-за четвёртого с несуществующим mark_id значит
        платить ещё раз за то, что уже получено. Но если невалидно ПЕРВОЕ действие,
        поднимается ValueError — ровно как в plan(), чтобы circuit-breaker увидел
        отказ и на третий раз ушёл в FLAT.
        """
        data = json.loads(resp_text)
        content = data['choices'][0]['message']['content']
        if isinstance(content, list):   # some servers return content as parts
            content = ''.join(p.get('text', '') for p in content if isinstance(p, dict))
        payload = json.loads(content)
        raw = None
        if isinstance(payload, dict):
            raw = payload.get('actions')
            # Модель могла проигнорировать override и прислать одиночное действие
            # старой схемы — это корректный ответ, а не сбой: принимаем как пачку из
            # одного, иначе каждый такой ответ считался бы отказом и открывал бы
            # circuit-breaker на ровном месте.
            if raw is None and 'action' in payload:
                raw = [payload]
        elif isinstance(payload, list):
            raw = payload
        if not isinstance(raw, list) or not raw:
            raise ValueError('VLM sequence rejected: нет непустого списка actions')

        actions = []
        for item in raw[:max(1, int(n))]:
            act, reason = parse_vlm_action(item, obs)
            if act is None:
                if not actions:
                    raise ValueError('VLM action rejected: %s' % reason)
                break
            actions.append(act)
            if act.kind == DONE:     # всё после DONE смысла не имеет
                break
        return actions

    def plan_sequence(self, obs: Observation, image_jpeg: Optional[bytes] = None,
                      map_jpeg: Optional[bytes] = None, n: int = 1) -> list:
        """До n атомарных действий за ОДИН запрос к модели (replan_every_n).

        n <= 1 идёт ровно прежним путём (plan(): тот же промпт, та же схема, тот же
        max_tokens) — это дефолт, и он обязан остаться неизменным, чтобы прогоны
        до и после этой правки были сравнимы.
        """
        n = max(1, int(n))
        if n == 1:
            return [self.plan(obs, image_jpeg, map_jpeg)]
        body = {
            'model': self.model,
            'messages': self.build_messages(obs, image_jpeg, map_jpeg, n=n),
            'temperature': 0,
            # Бюджет на пачку: 256 токенов это потолок ОДНОГО действия с rationale.
            # Оставить его для списка значило бы обрывать JSON на середине и ловить
            # JSONDecodeError вместо плана.
            'max_tokens': 256 * n,
            'response_format': {'type': 'json_object'},
        }
        return self.parse_sequence_response(self._post(body), obs, n)


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
