#!/usr/bin/env python3
"""Planner Orchestrator (Phase 4): VLM-mode planner over the FLAT executive.

Replans every N ATOMIC steps. Each step: build an Observation from the latest
detections + notes (+ the camera frame, the top-down SLAM map, the measured
forward clearance and the map coordinates of everything found so far), ask the
client (mock or OpenAI-compatible) for up to N atomic actions, and dispatch each:
  TURN                 -> GoToPose at a pose RELATIVE to the robot's real pose
  DRIVE_FORWARD        -> GoToPose ahead; when the way straight is blocked, a
                          free goal off the heading found on the SLAM map, so
                          Nav2 routes AROUND the obstacle instead of refusing
  DRIVE_TO_VISIBLE     -> ApproachDetection (drive to a detected object via Nav)
  DETECT_ALL           -> broad-vocab detector call -> objects + classes into notes
  DONE                 -> finish
Every motion step reports back what PHYSICALLY happened -- measured displacement
or rotation, plus the executive's refusal reason -- because an 'ok' the robot did
not earn is what kept the model repeating a move that changed nothing.
The vocabulary is deliberately small (raw motion + perception) so the VLM does its
own navigation reasoning -- a fair comparison against the FLAT policy. The VLM is
never on the reactive path; the executive owns motion + safety. A
per-call timeout + circuit-breaker degrade VLM->FLAT on loss. Mock-first: with
use_mock (or no credentials anywhere) the whole loop runs in sim/CI with no API
key. Trigger a mission by publishing the target on /vlm_mission (std_msgs/String);
abort one by publishing on /vlm_mission/cancel (std_msgs/Empty) -- the flag is
checked on step boundaries, so the skill already in flight finishes and no new
one is issued. Without it there was no abort at all: /vlm_mission is ignored
while a mission runs, so the loop always ran to max_steps.

Real-VLM credentials: set the ROS params vlm_base_url / vlm_api_key / vlm_model,
OR (preferred for secrets) export the environment variables VLM_BASE_URL /
VLM_API_KEY / VLM_MODEL -- env fills in any param left blank, so keys never need
to live in a launch file. A base_url from either source auto-engages the real
OpenAI-compatible client unless use_mock:=true is set explicitly.
"""
import json
import math
import os
import threading
import time
import uuid
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

# A plan plus the candidate pixel map captured at planning time, so DRIVE_TO_VISIBLE
# resolves mark_id->pixel against the SAME observation the VLM chose from -- even
# while a concurrent replan is already overwriting the live candidate state.
_PlanBundle = namedtuple('_PlanBundle', 'actions pixels')

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan
from std_msgs.msg import Empty, String
from tf2_ros import (Buffer, ConnectivityException, ExtrapolationException,
                     LookupException, TransformListener)

from ar_project_msgs.action import ApproachDetection, GoToPose, Stop
from object_tracking_msgs.action import DetectTarget
from object_tracking_msgs.msg import Notes

from fleet_comms.qos import detection_stream_nodeadline, media_besteffort

from ar_project_msgs.msg import Heartbeat
from fleet_comms.heartbeat import HeartbeatPublisher
from planner_orchestrator import orchestration as orch
from planner_orchestrator.planner_logic import (
    Candidate, CircuitBreaker, DegradationLatch, NotesBuffer, Observation,
    DETECT_ALL, DRIVE_FORWARD, DRIVE_TO_VISIBLE, TURN, distance_for_options,
    distance_is_known, format_distance,
)
from planner_orchestrator.vlm_client import make_client

try:
    import cv2
    import numpy as np
    from cv_bridge import CvBridge
    _HAVE_CV = True
except Exception:                       # mock mode needs no image pipeline
    _HAVE_CV = False

# Период обновления «что видит робот» и «вид сверху», пока миссия не идёт.
# 1 Гц совпадает с частотой кадра SSE у дашборда: чаще — впустую жечь JPEG-кодек
# и рендер карты на изображения, которые никто не успеет увидеть.
IDLE_VIEW_PERIOD_S = 1.0

# После скольких ОДИНАКОВЫХ действий подряд писать в заметки предупреждение о
# зацикливании. Три, а не два: два одинаковых шага — нормальный приём (довернуть
# на 180 двумя поворотами, проехать длинный коридор двумя рывками).
REPEAT_WARN_AFTER = 3

# Потолок одного заднего хода. Робот едет назад ВСЛЕПУЮ — камера и лидар
# смотрят вперёд, — поэтому шаг заведомо короткий: освободить место для
# разворота хватает, а вкатиться во что-то позади за 0.5 м трудно.
REVERSE_MAX_M = 0.5

# Радиус, который должен быть свободен вокруг ЦЕЛЕВОЙ точки объезда. Робот
# ~0.35 м в ширину, Nav2 раздувает препятствия примерно на 0.25 м; 0.30 —
# компромисс: цель заведомо не в стене, но и не отбрасываются проходы в дверях.
GOAL_CLEAR_RADIUS_M = 0.30

# Горизонтальное поле зрения цветной камеры RealSense D435 (~69°). Нужно только
# как запасной вариант, когда /camera_info ещё не пришёл: по нему считается
# азимут метки, чтобы положить найденный предмет на карту.
DEFAULT_HFOV_RAD = 1.204

# Порог уверенности, с которого детекция ЗАПОМИНАЕТСЯ на карте. Отдельный и
# заведомо высокий: DETECT_ALL работает с порогом 0.12 ради полноты списка на
# один шаг, и отмечать это на карте значило бы засеять её галлюцинациями,
# которые потом никуда не денутся — модель им верит, потому что «уже найдено».
DETECT_MEMORY_CONF = 0.55

# Ближе этого две детекции одного класса считаются одним предметом (метры).
# Пол-метра — примерно точность привязки: поза робота в карте, азимут пикселя и
# дальность каждый дают свою ошибку.
OBJECT_MERGE_M = 0.5

# Сколько предметов держим в памяти. Дальше вытесняем самые неуверенные: список
# уходит в промпт, и бесконечный он быть не может.
OBJECT_MEMORY_MAX = 30


def _parse_rooms(raw, logger=None):
    """'имя|x0,x1,y0,y1;имя|...' -> {name: (x_min, x_max, y_min, y_max)}.

    Не JSON намеренно: строка приходит через `-p name:=value`, то есть через
    YAML-разбор ROS-аргументов, и фигурные скобки с кавычками валят его целиком
    (проверено: оркестратор падал с «Failed to parse global arguments»). Здесь
    нет ни одного символа, особенного для YAML.

    Кривой кусок пропускаем молча, а не роняем узел: подписи на карте — удобство,
    и падать из-за них было бы обменом большого на малое.
    """
    text = (raw or '').strip()
    if not text:
        return {}
    out = {}
    for chunk in text.split(';'):
        chunk = chunk.strip()
        if not chunk or '|' not in chunk:
            continue
        name, _, box = chunk.partition('|')
        try:
            x0, x1, y0, y1 = (float(v) for v in box.split(','))
        except (TypeError, ValueError):
            continue
        out[name.strip()] = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    if not out and logger is not None:
        logger.warn('rooms_spec не разобран — карта пойдёт без подписей комнат')
    return out


def _yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _map_latched_qos():
    """QoS matching a SLAM /map publisher: reliable + transient_local + keep_last(1)."""
    q = QoSProfile(depth=1)
    q.history = HistoryPolicy.KEEP_LAST
    q.reliability = ReliabilityPolicy.RELIABLE
    q.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return q


class PlannerOrchestrator(Node):
    HEARTBEAT_PERIOD_S = 0.5

    def __init__(self):
        super().__init__('planner_orchestrator')
        self.heartbeat = HeartbeatPublisher(self, 'planner_orchestrator',
                                            period_s=self.HEARTBEAT_PERIOD_S)
        # ---- params ----
        self.declare_parameter('replan_every_n', 1)
        self.declare_parameter('use_mock', False)
        self.declare_parameter('vlm_base_url', '')
        self.declare_parameter('vlm_api_key', '')
        self.declare_parameter('vlm_model', '')
        self.declare_parameter('vlm_timeout_s', 8.0)
        self.declare_parameter('turn_step_rad', 0.6)
        self.declare_parameter('forward_step_m', 0.5)
        self.declare_parameter('approach_offset', 0.58)
        self.declare_parameter('max_steps', 60)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        # Relative VLM actions (TURN / DRIVE_FORWARD) only need a local metric
        # frame. In sim startup / degraded SLAM, map->base_link can be briefly
        # absent while odom->base_link is already valid, so fall back to odom
        # instead of dropping the motion.
        self.declare_parameter('motion_fallback_frame', 'odom')
        self.declare_parameter('tf_lookup_timeout_s', 1.0)
        self.declare_parameter('skill_wait_s', 5.0)
        self.declare_parameter('result_timeout_s', 90.0)
        # Epoch the orchestrator's skill goals carry: must match the executive's
        # CURRENT MissionState epoch (idle = 0) or the skills reject them as
        # zombies. In VLM mode the orchestrator drives the skills directly.
        self.declare_parameter('mission_epoch', 0)
        # Floor on per-step wall time so an instant-reached skill can't make the
        # loop hammer the executive (and gives observations time to refresh).
        self.declare_parameter('min_step_s', 0.5)
        # Phase 4.6 anytime/async replan: compute the NEXT plan concurrently while
        # the executive still runs the current action, adopt only at a commit-point
        # (the batch boundary) -> no idle / "wasted actions" between replans.
        self.declare_parameter('async_replan', True)
        # Phase 3 binding: pull real Set-of-Mark candidates from the edge detector
        # and feed the chosen mark's pixel to ApproachDetection on DRIVE_TO_VISIBLE.
        self.declare_parameter('detect_action_name', 'detect_target')
        self.declare_parameter('detect_timeout_s', 6.0)
        # Legacy override: if >0, applies one confidence floor to both target
        # detection and DETECT_ALL. Prefer the split thresholds below: they mirror
        # the diploma tracker (DINO strict for target, YOLOE permissive for overview).
        self.declare_parameter('detect_conf', 0.0)
        self.declare_parameter('target_detect_conf', 0.50)
        self.declare_parameter('detect_all_conf', 0.12)
        # Период холостой детекции: пока миссия не идёт, оркестратор раз в столько
        # секунд гоняет детектор широким словарём и отдаёт в дашборд кадр С РАЗМЕТКОЙ,
        # а не сырой. Иначе «что видит робот» до задания показывает картинку, по
        # которой нельзя понять, распознаёт ли детектор вообще хоть что-то, — а
        # именно это оператор и проверяет перед пуском. 0 = выключить (детектор
        # тогда в простое не трогается совсем).
        # 15, а не 5: показ обстановки в простое — не измерение, обновления раз в
        # 15 с оператору хватает. При 5 с детектор молотил YOLOE вхолостую почти
        # непрерывно (замер: прогон занимает ~5 с, то есть пауза между ними
        # исчезала), держал ядро CPU на пустых кадрах и отбирал его у Gazebo,
        # Nav2 и RTAB-Map в том же контейнере.
        self.declare_parameter('idle_detect_period_s', 15.0)
        # Пауза перед ПЕРВОЙ холостой детекцией, отсчитывается от старта узла.
        # Не косметика: первый вызов YOLOE грузит веса на GPU и надолго занимает
        # ядро, а оркестратор поднимается ровно тогда, когда lifecycle_manager
        # конфигурирует Nav2. Без паузы это измеримо ломало запуск — на этой машине
        # цепочка вставала после «Configuring controller_server» и не доходила до
        # активации ВООБЩЕ (проверено: преflight не зеленел и через 210 с, при 1000%+
        # CPU в контейнере). Украшение интерфейса не имеет права мешать подъёму стека.
        self.declare_parameter('idle_detect_warmup_s', 60.0)
        # Порог уверенности ДЛЯ ПОКАЗА в простое. Отдельный от detect_all_conf (0.12)
        # намеренно: тот низкий порог существует ради полноты списка для VLM, где
        # лишний кандидат безобиден. На картинке же всё, что около порога, живёт
        # ровно один кадр — метки появляются и исчезают между прогонами, номера
        # переприсваиваются, и оператор смотрит на мельтешение вместо обстановки.
        # Показываем только то, в чём детектор уверен. 0.25, а не выше: цели в
        # симуляции — плоские билборды, и YOLOE даёт по ним 0.26..0.40, так что
        # порог 0.35 срезал в flat_detect ВСЁ, включая сам билборд (замерено).
        # 0.25 отсекает дрожащий хвост 0.13..0.22 и оставляет устойчивые метки.
        self.declare_parameter('idle_detect_conf', 0.25)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('subscribe_camera_image', True)
        self.declare_parameter('camera_image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera_use_compressed_input', False)
        # Attach the top-down SLAM occupancy map as a 2nd image to the VLM. map_max_px
        # bounds the rendered map's longest side (kept small to limit tokens/latency).
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('send_map', True)
        self.declare_parameter('map_max_px', 384)
        # Executive-side forward-clearance clamp for DRIVE_FORWARD (see
        # orchestration.forward_clearance for the failure it closes). standoff is
        # the gap kept between the scan frame and the nearest obstacle after the
        # drive: the episode collision guard trips at 0.16 m, Nav2's inflation
        # settles around 0.25, so 0.40 stays clear of both. corridor_half_width
        # is half the robot's body width (~0.35 m wide) plus a small margin.
        # scan_topic '' disables the clamp (real robot without a front scan).
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('forward_standoff_m', 0.40)
        self.declare_parameter('forward_corridor_half_width_m', 0.25)
        # Ниже этого Nav2 не поедет ВООБЩЕ: цель попадает внутрь допуска
        # (nav2_params.yaml: xy_goal_tolerance 0.20), controller_server сразу
        # рапортует «Reached the goal», и робот не трогается с места. Прежний
        # порог отказа был 0.05 м, из-за чего в зазоре 0.05..0.20 команда
        # считалась выполненной, VLM получал «ok» и повторял её.
        # Наблюдалось у оператора ровно так: сорок шагов DRIVE_FORWARD +0.18m
        # подряд, «Reached the goal!» на каждом, робот стоит, кадр и карта не
        # меняются, миссия упирается в max_steps. 0.25 = допуск 0.20 плюс запас
        # на дискретность одометрии.
        self.declare_parameter('min_drive_m', 0.25)
        # Комнаты мира: 'имя|x0,x1,y0,y1;имя|...' в метрах. Подписываются на
        # карте, которую видит модель. АПРИОРНОЕ знание — робот его не выводит,
        # его передаёт консоль из worlds.yaml. Пусто = карта остаётся чистой
        # SLAM-сеткой, как раньше.
        self.declare_parameter('rooms_spec', '')
        # Поза начала кадра `map` в МИРОВЫХ координатах. Для SLAM это точка
        # старта робота, поэтому консоль передаёт сюда тот же spawn, что и
        # Gazebo. Нужна ровно затем, чтобы комнаты из worlds.yaml (мировые
        # координаты) легли на карту SLAM, а не на семь метров в сторону.
        self.declare_parameter('rooms_origin_x', 0.0)
        self.declare_parameter('rooms_origin_y', 0.0)
        self.declare_parameter('rooms_origin_yaw', 0.0)
        # Объезд для DRIVE_FORWARD. Команду «вперёд» исполняет Nav2 по
        # построенной карте: если точка прямо по курсу в стене, берётся
        # ближайшая свободная примерно в том же направлении, и маршрут до неё
        # Nav2 прокладывает сам. false = прежнее поведение (честный отказ).
        self.declare_parameter('nav_detour', True)
        self.declare_parameter('detour_max_bearing_rad', 1.05)
        # Запоминание найденных предметов на карте (см. DETECT_MEMORY_CONF).
        self.declare_parameter('detect_memory_conf', DETECT_MEMORY_CONF)
        self.declare_parameter('camera_info_topic',
                               '/camera/camera/color/camera_info')
        g = lambda n: self.get_parameter(n).value
        self.replan_n = max(1, int(g('replan_every_n')))
        self.turn_step = float(g('turn_step_rad'))
        self.fwd_step = float(g('forward_step_m'))
        self.approach_offset = float(g('approach_offset'))
        self.max_steps = int(g('max_steps'))
        self.map_frame = g('map_frame')
        self.robot_frame = g('robot_frame')
        self.motion_fallback_frame = g('motion_fallback_frame')
        self.tf_lookup_timeout_s = float(g('tf_lookup_timeout_s'))
        self.skill_wait_s = float(g('skill_wait_s'))
        self.result_timeout_s = float(g('result_timeout_s'))
        self.min_step_s = float(g('min_step_s'))
        self.detect_timeout_s = float(g('detect_timeout_s'))
        self.detect_conf = float(g('detect_conf'))
        self.target_detect_conf = float(g('target_detect_conf'))
        self.detect_all_conf = float(g('detect_all_conf'))
        self.idle_detect_period_s = float(g('idle_detect_period_s'))
        self.idle_detect_warmup_s = float(g('idle_detect_warmup_s'))
        self.idle_detect_conf = float(g('idle_detect_conf'))
        if self.detect_conf > 0.0:
            self.target_detect_conf = self.detect_conf
            self.detect_all_conf = self.detect_conf
        self.vlm_timeout_s = float(g('vlm_timeout_s'))
        self.camera_frame = g('camera_frame')
        self.subscribe_camera_image = bool(g('subscribe_camera_image')) and _HAVE_CV
        self.camera_image_topic = str(g('camera_image_topic'))
        self.camera_use_compressed_input = bool(g('camera_use_compressed_input'))
        self.send_map = bool(g('send_map')) and _HAVE_CV
        self.map_max_px = int(g('map_max_px'))
        self.scan_topic = str(g('scan_topic'))
        self.forward_standoff_m = float(g('forward_standoff_m'))
        self.min_drive_m = float(g('min_drive_m'))
        self.rooms = orch.rooms_to_map_frame(
            _parse_rooms(str(g('rooms_spec') or ''), self.get_logger()),
            float(g('rooms_origin_x')), float(g('rooms_origin_y')),
            float(g('rooms_origin_yaw')))
        self.nav_detour = bool(g('nav_detour'))
        self.detour_max_bearing_rad = float(g('detour_max_bearing_rad'))
        self.detect_memory_conf = float(g('detect_memory_conf'))
        self.forward_corridor_half_width_m = float(g('forward_corridor_half_width_m'))
        self.async_replan = bool(g('async_replan'))
        self._planner_pool = ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix='replan')
        # ОТДЕЛЬНЫЙ пул для детекции в простое, а не _planner_pool: тот занят
        # упреждающим replan во время миссии, и общая очередь означала бы, что
        # холостая картинка задерживает план. _call_action блокирует поток целиком
        # (ждёт Event), поэтому вызывать его прямо из таймера нельзя — таймер бы
        # встал вместе с исполнителем.
        self._idle_pool = ThreadPoolExecutor(max_workers=1,
                                             thread_name_prefix='idleview')
        self._idle_future = None
        # Первый холостой прогон разрешён только после прогрева (см. параметр).
        self._idle_detect_next = time.monotonic() + max(0.0,
                                                        self.idle_detect_warmup_s)
        self._idle_last_problem = ''    # последняя причина «кадр без разметки»
        self._idle_detect_started = False   # был ли хоть один прогон детекции

        self.client = make_client(use_mock=bool(g('use_mock')), base_url=g('vlm_base_url'),
                                  api_key=g('vlm_api_key'), model=g('vlm_model'),
                                  timeout_s=float(g('vlm_timeout_s')))
        # Where credentials came from -- a label only; the key/url are never logged.
        if g('vlm_base_url'):
            self._cred_src = 'param'
        elif os.environ.get('VLM_BASE_URL'):
            self._cred_src = 'env'
        else:
            self._cred_src = 'none'
        self.cb = CircuitBreaker()
        # Phase 5.1 seamless degradation: a zero-network FLAT fallback (MockPlanner)
        # the orchestrator latches onto when the VLM is lost (circuit-breaker open),
        # so the mission CONTINUES as FLAT instead of stopping.
        self._fallback = make_client(use_mock=True)
        self._degrade = DegradationLatch()
        self.notes = NotesBuffer()
        self._epoch = int(g('mission_epoch'))

        # ---- inputs ----
        self._pixel = None
        self._cam_bgr = None      # latest camera frame; encoded to JPEG lazily
        self._map = None                 # latest SLAM OccupancyGrid (for the VLM map)
        # guards the consistency of the (camera jpeg, /target_pixel) snapshot vs the
        # ROS executor threads that write them (_on_image / _on_pixel)
        self._lock = threading.Lock()
        self._bridge = CvBridge() if _HAVE_CV else None
        sub = ReentrantCallbackGroup()
        # /map нужен ДВУМ потребителям: картинке для VLM (send_map) и объезду
        # DRIVE_FORWARD по карте (nav_detour). Связывать их одним флагом нельзя:
        # send_map:=false — рекомендованный в RUNBOOK способ облегчить запрос к
        # медленному VLM, и он молча отключал бы объезд, оставляя в заметках
        # фразу «карта не показывает свободного места» при том, что карту вообще
        # не получали. _render_map отдельно проверяет send_map, так что подписка
        # сама по себе ничего в промпт не тащит.
        if self.send_map or (self.nav_detour and _HAVE_CV):
            self.create_subscription(OccupancyGrid, g('map_topic'), self._on_map,
                                     _map_latched_qos(), callback_group=sub)
        # BEST_EFFORT/no-deadline to match the detector/tracker's offered QoS (a
        # RELIABLE sub would receive nothing from a BEST_EFFORT publisher)
        self.create_subscription(PointStamped, '/target_pixel', self._on_pixel,
                                 detection_stream_nodeadline(), callback_group=sub)
        if self.subscribe_camera_image:
            image_type = CompressedImage if self.camera_use_compressed_input else Image
            self.create_subscription(image_type, self.camera_image_topic,
                                     self._on_image, media_besteffort(), callback_group=sub)
            self.get_logger().info(
                'planner_orchestrator camera input: topic=%s compressed=%s'
                % (self.camera_image_topic, self.camera_use_compressed_input))
        # Операторская отмена. Без неё прервать миссию было НЕЧЕМ: _on_mission
        # молча игнорирует всё, пока _busy, а цикл крутится до max_steps -- то
        # есть нажать "стоп" в интерфейсе означало бы остановить движение
        # skill'ом Stop и смотреть, как планировщик на следующем шаге выдаёт
        # новую цель и робот едет дальше. Это НЕ ограничение планировщика: флаг
        # проверяется между шагами и только прекращает миссию целиком, он не
        # влияет ни на один выбор модели.
        # Событие создаётся ДО обеих подписок: обработчики читают его, и порядок
        # "сначала состояние, потом подписка" исключает callback по недосозданному полю.
        self._cancel = threading.Event()
        self.create_subscription(String, '/vlm_mission', self._on_mission, 1,
                                 callback_group=sub)
        self.create_subscription(Empty, '/vlm_mission/cancel', self._on_cancel, 1,
                                 callback_group=sub)
        self._scan = None                # latest LaserScan (forward-clearance clamp)
        self._exec_note = None           # honest per-step execution detail for notes
        self._skill_reason = ''          # почему исполнитель отказал (в заметки)
        self._last_action_name = ''      # для счётчика повторов подряд
        self._same_action_run = 0
        # Найденные предметы: {(label, cell_x, cell_y): {'label','x','y','score'}}
        # в метрах кадра `map`. Живут в пределах миссии (сбрасываются в
        # _run_mission вместе с заметками) и рисуются на карте для модели.
        self._objects = {}
        self._objects_lock = threading.Lock()
        # Интринсики цветной камеры: по ним считается азимут метки, без него
        # предмет некуда положить на карту. До прихода /camera_info работает
        # запасной путь по DEFAULT_HFOV_RAD.
        self._cam_fx = None
        self._cam_cx = None
        camera_info_topic = str(g('camera_info_topic') or '')
        if camera_info_topic and _HAVE_CV:
            self.create_subscription(CameraInfo, camera_info_topic,
                                     self._on_camera_info, media_besteffort(),
                                     callback_group=sub)
        if self.scan_topic:
            self.create_subscription(LaserScan, self.scan_topic, self._on_scan,
                                     media_besteffort(), callback_group=sub)
        self.notes_pub = self.create_publisher(Notes, '/planner/notes', 1)

        # ---- monitoring outputs (mission dashboard) ----
        # /vlm/activity: structured human-readable trace of what the VLM saw,
        # decided and what actually happened (the same content that used to live
        # only in ephemeral console logs). TRANSIENT_LOCAL with a short history
        # so a late-joining monitor replays recent events. Consumed edge-locally.
        activity_qos = QoSProfile(depth=50)
        activity_qos.history = HistoryPolicy.KEEP_LAST
        activity_qos.reliability = ReliabilityPolicy.RELIABLE
        activity_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._activity_pub = self.create_publisher(String, '/vlm/activity', activity_qos)
        self._activity_seq = 0
        # Latest annotated Set-of-Mark frame ("what the robot sees + the mark ids
        # the VLM chose from") and the top-down map image actually sent to the VLM.
        self._setofmark_pub = self.create_publisher(CompressedImage, '/vlm/setofmark', 1)
        self._map_view_pub = self.create_publisher(CompressedImage, '/vlm/map_view', 1)
        # Живые виды, пока миссия НЕ идёт. Без этого таймера оба вида обновлялись
        # только внутри цикла миссии, и в простое оператор видел либо пустоту, либо
        # кадр и карту многоминутной давности с подписью «отправлено VLM N с назад» —
        # то есть не мог проверить, где робот и что он видит, ПЕРЕД тем как дать
        # задание. Во время миссии таймер молчит: там ценнее показывать ровно то
        # изображение, которое реально ушло в модель, а не более свежее.
        self._idle_view_timer = self.create_timer(IDLE_VIEW_PERIOD_S,
                                                  self._publish_idle_views)

        # ---- executive skill clients (loopback-style poll on a reentrant group) ----
        cg = ReentrantCallbackGroup()
        self._ac = {
            orch.SKILL_GO_TO_POSE: ActionClient(self, GoToPose, 'go_to_pose', callback_group=cg),
            orch.SKILL_APPROACH: ActionClient(self, ApproachDetection, 'approach_detection', callback_group=cg),
            orch.SKILL_STOP: ActionClient(self, Stop, 'stop', callback_group=cg),
        }
        self._detect = ActionClient(self, DetectTarget, g('detect_action_name'),
                                    callback_group=cg)
        # inject the chosen candidate's pixel for ApproachDetection (matches the
        # detector's /target_pixel QoS so the executive's subscriber receives it)
        self._pixel_pub = self.create_publisher(PointStamped, '/target_pixel',
                                                detection_stream_nodeadline())
        self._tf = Buffer()
        self._tfl = TransformListener(self._tf, self)
        self._busy = False
        self.get_logger().info(
            'planner_orchestrator up (Phase 4 VLM mode): client=%s creds=%s replan_every_n=%d. '
            'Publish target on /vlm_mission to start.'
            % (type(self.client).__name__, self._cred_src, self.replan_n))

    # ---- input callbacks ----
    def _on_map(self, msg):
        self._map = msg

    def _on_scan(self, msg):
        self._scan = msg

    def _on_camera_info(self, msg):
        # k = [fx 0 cx; 0 fy cy; 0 0 1]. Нули значат «камера ещё не
        # откалибрована» — такие сообщения пропускаем, иначе деление на fx=0.
        try:
            fx, cx = float(msg.k[0]), float(msg.k[2])
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if fx > 1.0:
            self._cam_fx, self._cam_cx = fx, cx

    def _on_pixel(self, msg):
        with self._lock:
            self._pixel = msg

    def _on_image(self, msg):
        try:
            if self.camera_use_compressed_input:
                arr = np.frombuffer(msg.data, np.uint8)
                cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                cv = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            if cv is None:
                return
            # Store the frame; encode to JPEG lazily in _camera_jpeg. The camera
            # arrives at 6-15 Hz but a JPEG is needed at most once per replan (and
            # usually superseded by the detector's annotated frame), so encoding
            # every frame here just burned CPU.
            with self._lock:
                self._cam_bgr = cv
        except Exception as e:
            self.get_logger().warn('image decode failed: %s' % e, throttle_duration_sec=5.0)

    def _on_mission(self, msg):
        target = (msg.data or '').strip()
        if not target or self._busy:
            return
        self._busy = True
        self._cancel.clear()
        threading.Thread(target=self._run_mission, args=(target,), daemon=True).start()

    def _on_cancel(self, _msg):
        """Оператор прервал миссию. Флаг проверяется на границах шагов, поэтому
        текущее уже выданное skill-действие доигрывается до конца (прерывать его
        на полпути небезопасно -- исполнитель владеет движением), а нового не
        будет. Движение при этом гасится сразу через skill Stop."""
        if not self._busy:
            return
        self._cancel.set()
        self.get_logger().warn('VLM mission cancel requested by operator')
        self._activity('mission_cancel_requested')
        self._dispatch_stop()

    # ---- monitoring trace ----
    def _activity(self, event, **data):
        """Publish one structured VLM activity event on /vlm/activity (JSON) for
        the mission dashboard. Best-effort: monitoring must never break planning."""
        try:
            self._activity_seq += 1
            payload = {'seq': self._activity_seq, 'event': event, 'stamp': time.time()}
            payload.update(data)
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False, default=str)
            self._activity_pub.publish(msg)
        except Exception:                             # pragma: no cover
            pass

    def _publish_view(self, pub, jpeg_bytes):
        """Publish a JPEG byte string as CompressedImage (dashboard view topics)."""
        if not jpeg_bytes:
            return
        try:
            m = CompressedImage()
            m.header.stamp = self.get_clock().now().to_msg()
            m.format = 'jpeg'
            m.data = bytes(jpeg_bytes)
            pub.publish(m)
        except Exception:                             # pragma: no cover
            pass

    # ---- observation ----
    def _camera_jpeg(self):
        # Encode on demand (rare: ~once per replan). _on_image always stores a
        # fresh frame object, so the reference we grab under the lock is safe to
        # encode after releasing it.
        with self._lock:
            frame = self._cam_bgr
        if frame is None or not _HAVE_CV:
            return None
        ok, buf = cv2.imencode('.jpg', frame)
        return buf.tobytes() if ok else None

    def _observation(self, target, step_index):
        """Pull candidates + the matching VLM image together, then build the
        Observation. Returns (obs, pixels, jpeg, map_jpeg) so the plan uses a
        CONSISTENT (candidate ids, camera image) pair even while the camera/replan
        threads run; the top-down SLAM map is rendered alongside (or None)."""
        cands, pixels, jpeg = self._refresh_candidates(target)
        map_jpeg, map_text = self._render_map()
        obs = Observation(target=target, candidates=cands,
                          notes_facts=self.notes.facts, step_index=step_index,
                          map_text=map_text,
                          free_ahead_m=self._forward_clearance_m(),
                          objects_found=self._objects_for_prompt())
        return obs, pixels, jpeg, map_jpeg

    def _refresh_candidates(self, target):
        """Query the edge DetectTarget service. Returns (candidates, pixels, jpeg)
        captured together: jpeg = the annotated Set-of-Mark frame when available,
        else the latest camera frame -- so the VLM image always matches the candidate
        ids (and the camera callback can't clobber the annotated frame mid-plan).
        Falls back to a single /target_pixel candidate when the detector is absent."""
        if self._detect.wait_for_server(timeout_sec=1.0):
            g = DetectTarget.Goal()
            g.request_id = self._goal_id()
            g.mission_epoch = self._epoch
            g.query = target
            g.render_setofmark = True
            g.conf_threshold = self.target_detect_conf
            res = self._call_action(self._detect, g, self.detect_timeout_s)
            if res is not None and getattr(res, 'candidates', None):
                self._remember_objects(res.candidates)
                cands, pix = [], {}
                for c in res.candidates:
                    cands.append(Candidate(mark_id=int(c.mark_id), label=c.label,
                                           score=float(c.confidence),
                                           distance_m=float(c.pixel.z)))  # z = depth_m
                    pix[int(c.mark_id)] = c.pixel    # Point: x=u, y=v, z=depth_m
                jpeg = bytes(res.annotated.data) if res.annotated.data else self._camera_jpeg()
                # Dashboard view: what the robot sees + the mark ids offered to the VLM.
                self._publish_view(self._setofmark_pub, jpeg)
                return cands, pix, jpeg
            # Detector answered but found NOTHING -> report honestly empty. Must NOT fall
            # back to /target_pixel here: during DRIVE_TO_VISIBLE the orchestrator keeps
            # republishing the chosen pixel on /target_pixel, which would otherwise leak
            # back as a PHANTOM stale detection (target still "1.7 m away" after we drove
            # right up to it and YOLOE lost it at close range).
            #
            # Кадр в дашборд отдаём ДАЖЕ когда детектор не нашёл ничего. Раньше
            # публикация стояла только в ветке «есть кандидаты», и панель «что видит
            # робот» оставалась пустой ровно в том случае, когда оператору нужнее
            # всего посмотреть на картинку — цель не найдена, и вопрос «а что вообще
            # в кадре?» без изображения не разрешить.
            empty_jpeg = self._camera_jpeg()
            self._publish_view(self._setofmark_pub, empty_jpeg)
            return [], {}, empty_jpeg
        # Detector server absent -> last-resort single /target_pixel candidate (lets the
        # orchestrator also run against the continuous rgb_tracker instead of the service).
        return self._fallback_candidates(target)

    def _fallback_candidates(self, target):
        with self._lock:
            px = self._pixel
        jpeg = self._camera_jpeg()
        if px is not None:
            return ([Candidate(mark_id=1, label=target, score=1.0,
                               distance_m=float(px.point.z))], {1: px.point}, jpeg)
        return [], {}, jpeg

    # ---- память о найденных предметах (отметки на карте SLAM) ----------------
    def _remember_objects(self, detections):
        """Положить уверенные детекции на карту: пиксель + глубина -> метры кадра
        `map`, рядом с позой робота в момент съёмки.

        Зачем: без этого найденное жило только строкой в заметках, откуда
        вытеснялось через пару десятков шагов. Робот честно видел кровать,
        сообщал о ней и через двадцать шагов не знал ни что видел её, ни где.
        Порог намеренно высокий (detect_memory_conf, 0.55): отметка ставится
        НАВСЕГДА в пределах миссии, и ошибка в ней дороже пропуска — модель
        поверит собственной памяти и уедет искать несуществующий унитаз.

        Геометрия — в orch.detection_map_xy. Смещение камеры относительно
        base_link (единицы сантиметров) не учитывается: оно заведомо меньше
        ошибки самой привязки.

        Поза берётся ТЕКУЩАЯ, а не на момент кадра. Детекция случается на
        границе шага, когда робот стоит, так что расхождения нет; но при
        async_replan=true (в запуске выключен) перепланирование идёт параллельно
        движению, и тогда отметка уедет на путь, пройденный за время инференса.
        """
        pose = self._robot_pose()
        if pose is None or not detections:
            return
        rx, ry, ryaw, _ = pose
        fx, cx = self._cam_fx, self._cam_cx
        with self._lock:
            frame = self._cam_bgr
        width = frame.shape[1] if frame is not None else None
        added = []
        for det in detections:
            try:
                score = float(det.confidence)
                depth = float(det.pixel.z)
                u = float(det.pixel.x)
                label = str(det.label or '').strip()
            except (AttributeError, TypeError, ValueError):
                continue
            if not label or score < self.detect_memory_conf:
                continue
            if not distance_is_known(depth):
                continue
            if fx and cx is not None:
                use_fx, use_cx = fx, cx
            elif width:
                # Запасной путь без /camera_info: fx = (w/2) / tan(hfov/2).
                use_fx = (width / 2.0) / math.tan(DEFAULT_HFOV_RAD / 2.0)
                use_cx = width / 2.0
            else:
                continue
            ox, oy = orch.detection_map_xy(rx, ry, ryaw, depth, u, use_fx, use_cx)
            key = (label, round(ox / OBJECT_MERGE_M), round(oy / OBJECT_MERGE_M))
            with self._objects_lock:
                old = self._objects.get(key)
                if old is None or score > old['score']:
                    self._objects[key] = {'label': label, 'x': ox, 'y': oy,
                                          'score': score}
                    if old is None:
                        added.append('%s at map (%.1f, %.1f)' % (label, ox, oy))
                if len(self._objects) > OBJECT_MEMORY_MAX:
                    worst = min(self._objects, key=lambda k: self._objects[k]['score'])
                    self._objects.pop(worst, None)
        if added:
            self.get_logger().info('на карту добавлено: ' + '; '.join(added))

    def _objects_snapshot(self):
        with self._objects_lock:
            return sorted(self._objects.values(),
                          key=lambda o: (o['label'], o['x'], o['y']))

    def _objects_for_prompt(self):
        """Найденное для JSON-опций: метка + координаты карты, без score."""
        return [{'label': o['label'], 'x': round(o['x'], 2), 'y': round(o['y'], 2)}
                for o in self._objects_snapshot()]

    def _lookup_robot_pose(self, target_frame, timeout_s=0.0):
        try:
            if timeout_s > 0.0:
                tf = self._tf.lookup_transform(
                    target_frame, self.robot_frame, rclpy.time.Time(),
                    timeout=Duration(seconds=float(timeout_s)))
            else:
                tf = self._tf.lookup_transform(
                    target_frame, self.robot_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.x, t.y, yaw, target_frame)

    def _robot_pose(self):
        return self._lookup_robot_pose(self.map_frame, timeout_s=0.0)

    def _motion_pose(self):
        frames = [self.map_frame]
        if self.motion_fallback_frame and self.motion_fallback_frame not in frames:
            frames.append(self.motion_fallback_frame)
        for frame in frames:
            pose = self._lookup_robot_pose(frame, timeout_s=self.tf_lookup_timeout_s)
            if pose is not None:
                if frame != self.map_frame:
                    self.get_logger().warn(
                        'no %s->%s TF; using %s->%s for relative motion'
                        % (self.map_frame, self.robot_frame, frame, self.robot_frame),
                        throttle_duration_sec=5.0)
                return pose
        return None

    def _publish_idle_views(self):
        """Свежие «что видит робот» и «вид сверху» в простое (см. _idle_view_timer).

        Исключения глушатся: это украшение интерфейса, и упавший таймер rclpy больше
        не вызовет — уронить им узел, который в этот момент обязан принимать миссию,
        нельзя.
        """
        if self._busy:
            return
        try:
            map_jpeg, _ = self._render_map()
            if map_jpeg:
                self._publish_view(self._map_view_pub, map_jpeg)

            # Кадр С РАЗМЕТКОЙ детектора, если подошёл срок и предыдущий прогон уже
            # закончился. Детекция уходит в отдельный поток: она ходит в action-сервер
            # и блокирует вызывающего до ответа.
            if self.idle_detect_period_s > 0.0:
                busy_run = self._idle_future is not None and not self._idle_future.done()
                if not busy_run and time.monotonic() >= self._idle_detect_next:
                    self._idle_detect_next = (time.monotonic()
                                              + self.idle_detect_period_s)
                    self._idle_future = self._idle_pool.submit(self._idle_detect_once)
                    self._idle_detect_started = True
                    return          # кадр опубликует сам прогон детекции
                if busy_run:
                    return          # не перетираем размеченный кадр сырым
                if self._idle_detect_started:
                    # Прогон детекции уже был: кадр в дашборде обновляет ТОЛЬКО он.
                    # Иначе таймер (1 Гц) затирал бы размеченный кадр сырым четыре
                    # раза из пяти, и разметка мелькала бы на секунду — ровно то,
                    # из-за чего «отработанной детекции» в дашборде было не видно,
                    # хотя детектор её исправно возвращал.
                    return

            # Досюда доходим, только пока холостая детекция выключена или ещё не
            # стартовала (прогрев): пустая панель в эту минуту хуже сырого кадра.
            jpeg = self._camera_jpeg()
            if jpeg:
                self._publish_view(self._setofmark_pub, jpeg)
        except Exception:                              # noqa: BLE001
            pass

    def shutdown_pools(self):
        """Остановить фоновые пулы (см. вызов в main)."""
        self._cancel.set()
        for pool in (self._idle_pool, self._planner_pool):
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:          # cancel_futures появился в 3.9
                pool.shutdown(wait=False)
            except Exception:          # noqa: BLE001
                pass

    def _idle_detect_once(self):
        """Один холостой прогон детектора -> размеченный кадр в дашборд.

        Широкий словарь (пустой query), как у DETECT_ALL: оператор перед пуском
        смотрит, что робот вообще различает, а не ищет конкретную цель — её ещё
        не задали. Результат НЕ попадает ни в notes, ни в ленту активности: это
        не шаг миссии, и подмешивать его в журнал прогона нельзя.
        """
        problem = ''
        try:
            if self._busy:                # миссия стартовала, пока мы стояли в очереди
                return
            annotated = None
            # Таймаут как в _observation (1.0 с), а не короче: под нагрузкой
            # (Gazebo + SLAM + Nav2) обнаружение сервера занимает заметно больше,
            # чем на холостой машине, и слишком жадный таймаут молча уводил в
            # ветку «детектора нет» — кадр публиковался сырым, а сервер при этом
            # был жив и не получал НИ ОДНОГО запроса.
            if not self._detect.wait_for_server(timeout_sec=1.0):
                problem = 'detect_target_server не отвечает'
            else:
                g = DetectTarget.Goal()
                g.request_id = self._goal_id()
                g.mission_epoch = self._epoch
                g.query = ''
                g.render_setofmark = True
                g.conf_threshold = self.idle_detect_conf
                res = self._call_action(self._detect, g, self.detect_timeout_s)
                if res is None:
                    problem = 'детектор не ответил за %.1f с' % self.detect_timeout_s
                elif getattr(res, 'annotated', None) is None or not res.annotated.data:
                    problem = 'детектор ответил без размеченного кадра'
                else:
                    annotated = bytes(res.annotated.data)
            # Детектора нет или он промолчал — лучше сырой кадр, чем пустая панель.
            jpeg = annotated or self._camera_jpeg()
            if jpeg and not self._busy:
                self._publish_view(self._setofmark_pub, jpeg)
        except Exception as exc:                       # noqa: BLE001
            problem = '%s: %s' % (type(exc).__name__, exc)
        # Логируем ТОЛЬКО смену причины: молчаливый except здесь уже один раз стоил
        # часа диагностики (кадр шёл, а детектор не получал запросов), но писать в
        # журнал каждые несколько секунд одно и то же — значит утопить в шуме лог
        # миссии, ради которого журнал и читают.
        if problem != self._idle_last_problem:
            self._idle_last_problem = problem
            if problem:
                self.get_logger().warn('холостая детекция: %s '
                                       '(в дашборд идёт кадр без разметки)' % problem)

    def _render_map(self):
        """Render the latest SLAM OccupancyGrid to a compact top-down JPEG with the
        robot drawn on it (white=free, black=obstacle, gray=unknown; red dot+line =
        robot pose+heading), plus a text description. North-up, metric. Returns
        (jpeg_bytes | None, description | '').

        Поверх сетки — зелёные рамки и названия комнат из плана здания и синие
        точки с подписями для уже найденных предметов. И то и другое дублируется
        текстом в описании: надпись на картинке даёт пространственную привязку,
        текст переживает сжатие JPEG. Всё в кадре `map`, включая комнаты (их
        сдвигает orch.rooms_to_map_frame — приходят они в мировых координатах)."""
        grid = self._map
        if not self.send_map or grid is None or not _HAVE_CV:
            return None, ''
        w, h = int(grid.info.width), int(grid.info.height)
        res = float(grid.info.resolution)
        if w <= 0 or h <= 0 or res <= 0.0:
            return None, ''
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        try:
            data = np.asarray(grid.data, dtype=np.int16).reshape(h, w)
        except (ValueError, TypeError):
            return None, ''
        img = np.full((h, w), 127, dtype=np.uint8)        # unknown (-1)
        img[(data >= 0) & (data < 50)] = 255              # free
        img[data >= 50] = 0                               # occupied
        n_unknown = int(np.count_nonzero(data < 0))
        n_occ = int(np.count_nonzero(data >= 50))
        n_free = w * h - n_unknown - n_occ
        # Освоенность комнат считается ДО расширения холста: ox/oy/w/h ниже
        # переопределяются под габариты здания, а `data` остаётся исходной
        # сеткой SLAM. Это перевод «серого» из картинки в факт: «в спальне не
        # были совсем» — то, на что модель может опереться, глядя на подписи.
        explored_by_room = self._room_exploration(data, ox, oy, res, w, h)
        # Холст расширяется до габаритов ЗДАНИЯ, если комнаты известны. Без
        # этого видна только уже исследованная часть: SLAM-сетка растёт по мере
        # разведки, и подписи комнат, куда робот ещё не заходил, просто не
        # попадали в кадр — а именно они и нужны, чтобы решить, куда ехать.
        # Дорисованная область остаётся серой (=unknown): это честно, там
        # действительно ничего не измерено.
        if self.rooms:
            pad = 1.0
            rx0 = min(min(v[0] for v in self.rooms.values()) - pad, ox)
            rx1 = max(max(v[1] for v in self.rooms.values()) + pad, ox + w * res)
            ry0 = min(min(v[2] for v in self.rooms.values()) - pad, oy)
            ry1 = max(max(v[3] for v in self.rooms.values()) + pad, oy + h * res)
            nw = max(w, int(math.ceil((rx1 - rx0) / res)))
            nh = max(h, int(math.ceil((ry1 - ry0) / res)))
            if (nw, nh) != (w, h) and nw * nh <= 4000 * 4000:
                canvas = np.full((nh, nw), 127, dtype=np.uint8)
                cx0 = int(round((ox - rx0) / res))
                cy0 = int(round((oy - ry0) / res))
                cx0 = max(0, min(cx0, nw - w))
                cy0 = max(0, min(cy0, nh - h))
                canvas[cy0:cy0 + h, cx0:cx0 + w] = img
                img, w, h = canvas, nw, nh
                ox, oy = rx0, ry0
        # OccupancyGrid origin is bottom-left; image row 0 is top -> flip to north-up.
        img = cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_GRAY2BGR)
        pose = self._robot_pose()
        robot_xy = (pose[0], pose[1]) if pose else (ox + w * res / 2.0, oy + h * res / 2.0)
        if pose is not None:
            cx = int((pose[0] - ox) / res)
            cy = int((pose[1] - oy) / res)
            if 0 <= cx < w and 0 <= cy < h:
                py = h - 1 - cy                            # world->flipped image row
                r = max(2, w // 80)
                cv2.circle(img, (cx, py), r, (0, 0, 255), -1)
                ll = max(6, w // 12)
                hx = int(cx + math.cos(pose[2]) * ll)
                hy = int(py - math.sin(pose[2]) * ll)      # screen y is down
                cv2.line(img, (cx, py), (hx, hy), (0, 0, 255), 2)
        scale = self.map_max_px / float(max(w, h))
        if scale < 1.0:                                    # cap size; keep cells crisp
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_NEAREST)
        else:
            scale = 1.0
        # Подписи комнат и найденные предметы — ПОСЛЕ масштабирования: иначе
        # текст ужимается вместе с картинкой и на выходе нечитаем.
        rooms_line = self._draw_rooms(img, ox, oy, res, h, scale, explored_by_room)
        objects_line = self._draw_objects(img, ox, oy, res, h, scale)
        ok, buf = cv2.imencode('.jpg', img)
        if not ok:
            return None, ''
        text = orch.describe_occupancy_grid(
            w, h, res, robot_xy, n_free, n_occ, n_unknown)
        if rooms_line:
            text = text + ' ' + rooms_line
        if objects_line:
            text = text + ' ' + objects_line
        return buf.tobytes(), text

    def _room_exploration(self, data, ox, oy, res, w, h):
        """{имя комнаты: доля НЕ осмотренных клеток} по исходной сетке SLAM.

        Клетки за пределами сетки считаются неизвестными — так и есть, сетка
        растёт только там, где датчики что-то видели. Комната, целиком лежащая
        вне сетки, получает 1.0, то есть «не были совсем».
        """
        out = {}
        if not self.rooms:
            return out
        for name, (x0, x1, y0, y1) in self.rooms.items():
            cx0 = int(math.floor((x0 - ox) / res))
            cx1 = int(math.ceil((x1 - ox) / res))
            cy0 = int(math.floor((y0 - oy) / res))
            cy1 = int(math.ceil((y1 - oy) / res))
            total = max(1, (cx1 - cx0) * (cy1 - cy0))
            ix0, ix1 = max(0, cx0), min(w, cx1)
            iy0, iy1 = max(0, cy0), min(h, cy1)
            if ix1 <= ix0 or iy1 <= iy0:
                out[name] = 1.0
                continue
            known = int(np.count_nonzero(data[iy0:iy1, ix0:ix1] >= 0))
            out[name] = max(0.0, 1.0 - known / float(total))
        return out

    def _draw_rooms(self, img, ox, oy, res, grid_h, scale, explored=None):
        """Подписать комнаты на карте и вернуть их же строкой для промпта.

        Две формы одного и того же намеренно: надпись на картинке даёт модели
        пространственную привязку («туалет — вон та комната сверху»), а строка
        текстом переживает любое качество JPEG и читается вернее, чем мелкие
        буквы. Что-то одно регулярно теряется.

        К каждой комнате приписывается, насколько она осмотрена. Одной легенды
        «серое = не были» мало: по картинке 384 px модель не считает доли, а
        решение «куда ехать» — это именно выбор наименее осмотренной комнаты.
        """
        if not self.rooms:
            return ''
        explored = explored or {}
        parts = []
        for name in sorted(self.rooms):
            x0, x1, y0, y1 = self.rooms[name]
            p0 = self._to_px(x0, y1, ox, oy, res, grid_h, scale)   # верхний-левый
            p1 = self._to_px(x1, y0, ox, oy, res, grid_h, scale)   # нижний-правый
            cv2.rectangle(img, p0, p1, (0, 140, 0), 1)
            tx, ty = self._to_px((x0 + x1) / 2.0, (y0 + y1) / 2.0,
                                 ox, oy, res, grid_h, scale)
            self._put_label(img, str(name), tx, ty, scale, (0, 120, 0))
            unknown = explored.get(name)
            if unknown is None:
                state = ''
            elif unknown >= 0.7:
                state = ', NOT visited yet'
            elif unknown >= 0.25:
                state = ', %.0f%% of it still unseen' % (100.0 * unknown)
            else:
                state = ', already searched'
            parts.append('%s at x %.1f..%.1f, y %.1f..%.1f%s'
                         % (name, x0, x1, y0, y1, state))
        return ('Rooms (from the building plan, in your own map coordinates): '
                + '; '.join(parts) + '.')

    def _draw_objects(self, img, ox, oy, res, grid_h, scale):
        """Отметить найденные предметы на карте и вернуть их же строкой.

        Отметка ставится там, где предмет РЕАЛЬНО стоит в кадре `map`, а не там,
        где робот его увидел: модель спрашивает «где я это видел», и ответом
        должно быть место предмета. Синий кружок с подписью, чтобы не спутать с
        красной позой робота и зелёными рамками комнат.
        """
        objs = self._objects_snapshot()
        if not objs:
            return ''
        parts = []
        for o in objs:
            px, py = self._to_px(o['x'], o['y'], ox, oy, res, grid_h, scale)
            if not (0 <= px < img.shape[1] and 0 <= py < img.shape[0]):
                continue
            cv2.circle(img, (px, py), max(2, img.shape[1] // 110), (255, 90, 0), -1)
            self._put_label(img, o['label'], px, py - 8, scale, (200, 60, 0))
            parts.append('%s at (%.1f, %.1f)' % (o['label'], o['x'], o['y']))
        if not parts:
            return ''
        return ('Objects you have already found (blue dots on the map): '
                + '; '.join(parts) + '.')

    @staticmethod
    def _to_px(wx, wy, ox, oy, res, grid_h, scale):
        """Метры кадра `map` -> пиксель отмасштабированной картинки."""
        cx = (wx - ox) / res
        cy = (wy - oy) / res
        return int(cx * scale), int((grid_h - 1 - cy) * scale)

    @staticmethod
    def _put_label(img, text, tx, ty, scale, color):
        """Подпись с белой подложкой: тонкий текст без неё теряется и на белом
        свободном месте, и на сером неизвестном."""
        fs = max(0.35, 0.45 * scale) if scale < 1.0 else 0.45
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        org = (max(0, tx - tw // 2), max(th, ty + th // 2))
        cv2.rectangle(img, (org[0] - 2, org[1] - th - 2),
                      (org[0] + tw + 2, org[1] + 3), (255, 255, 255), -1)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1,
                    cv2.LINE_AA)

    # ---- anytime/async mission loop (Phase 4.6): replan overlaps execution ----
    def _compute_plan(self, target, step):
        """Build an observation and ask the planner for up to N atomic actions.
        Runs either inline (bootstrap) or on the planner pool concurrently with
        execution. Returns a _PlanBundle (actions + the candidate pixel snapshot
        the VLM chose from); empty actions on VLM failure (circuit-breaker fed)."""
        # (obs, pixels, jpeg, map_jpeg) captured together -> the plan's DRIVE_TO_VISIBLE
        # pixels and the VLM images are a consistent set (race-free vs camera/replan).
        obs, pixels, jpeg, map_jpeg = self._observation(target, step)
        # Dashboard views: the map image actually sent to the VLM this cycle.
        self._publish_view(self._map_view_pub, map_jpeg)
        # Phase 5.1: pick VLM or the latched FLAT fallback (once the breaker opens).
        client = self._degrade.select(self.client, self._fallback, self.cb.is_open)
        if self._degrade.just_degraded():
            self.get_logger().error('circuit-breaker OPEN -> degrade VLM->FLAT (mission '
                                     'continues as FLAT, DEGRADED)')
            self.notes.add_fact('DEGRADED: VLM lost -> continuing in FLAT fallback')
            self._activity('degraded', step=step,
                           detail='VLM circuit-breaker OPEN -> continuing in FLAT fallback')
        best = max(obs.candidates, key=lambda c: c.score, default=None)
        det = ('' if best is None else " best='%s' conf=%.2f @%s"
               % (best.label, best.score, format_distance(best.distance_m)))
        self.get_logger().info(
            'observe@step %d: %d detection(s)%s, notes=%d, map=%s -> asking %s'
            % (step, len(obs.candidates), det, len(obs.notes_facts),
               'yes' if map_jpeg else 'no', type(client).__name__))
        self._activity(
            'observe', step=step, n_detections=len(obs.candidates),
            detections=[{'mark_id': c.mark_id, 'label': c.label,
                         'score': round(float(c.score), 3),
                         'distance_m': distance_for_options(c.distance_m)}
                        for c in obs.candidates],
            notes=len(obs.notes_facts), map='yes' if map_jpeg else 'no',
            client=type(client).__name__)
        vlm_t0 = time.monotonic()
        try:
            actions = list(client.plan_sequence(obs, jpeg, map_jpeg, n=self.replan_n))
            self.cb.record_success() if actions else self.cb.record_failure()
            self.get_logger().info('plan@step %d: VLM returned %d action(s): %s'
                                   % (step, len(actions),
                                      ', '.join(self._action_brief(a) for a in actions) or '-'))
            self._activity(
                'plan', step=step,
                latency_ms=round((time.monotonic() - vlm_t0) * 1e3, 1),
                actions=[{'action': self._action_brief(a),
                          'rationale': a.rationale or ''} for a in actions])
        except Exception as e:
            self.cb.record_failure()
            self.get_logger().warn('plan failed (%s); cb_open=%s' % (e, self.cb.is_open))
            self._activity('plan_failed', step=step, error=str(e),
                           cb_open=bool(self.cb.is_open))
            actions = []
        # Real per-component health (was: heartbeat always OK). Latency feeds the
        # p99 budget; DEGRADED reflects the breaker/latch and resets once healthy.
        self.heartbeat.set_latency_ms((time.monotonic() - vlm_t0) * 1e3)
        self.heartbeat.set_status(
            Heartbeat.DEGRADED if (self.cb.is_open or self._degrade.degraded)
            else Heartbeat.OK)
        return _PlanBundle(actions, pixels)

    def _next_bundle(self, pending, target, step):
        """Adopt the concurrently-computed plan at the commit-point (no idle if it
        finished during execution), or compute inline when async is off."""
        if pending is not None:
            try:
                return pending.result()
            except Exception:
                return _PlanBundle([], {})
        return self._compute_plan(target, step)

    def _run_mission(self, target):
        self.get_logger().info('VLM mission start: target="%s"' % target)
        self._activity('mission_start', target=target,
                       client=type(self.client).__name__, creds=self._cred_src,
                       replan_every_n=self.replan_n, max_steps=self.max_steps)
        self.notes = NotesBuffer()
        self.cb = CircuitBreaker()
        self._degrade = DegradationLatch()   # fresh mission retries the VLM
        # Память о найденном — ровно на одну миссию: предметы могли переставить,
        # а робота вернуть в исходную точку кнопкой сброса, и старые отметки
        # тогда врут увереннее, чем помогают.
        with self._objects_lock:
            self._objects = {}
        self._last_action_name, self._same_action_run = '', 0
        step = 0
        pending = None
        try:
            bundle = self._compute_plan(target, step)   # bootstrap (the only idle point)
            while rclpy.ok() and step < self.max_steps:
                if self._cancel.is_set():
                    break
                if not bundle.actions:
                    # Degradation (cb open) does NOT stop the mission -- _compute_plan
                    # has already switched to the FLAT fallback. Only stop if even the
                    # fallback yields nothing; otherwise retry (transient empty plan).
                    if self._degrade.degraded:
                        self.get_logger().error('FLAT fallback produced no action -> stopping')
                        self._dispatch_stop()
                        break
                    bundle = self._next_bundle(pending, target, step)
                    pending = None
                    time.sleep(self.min_step_s)   # rate-limit transient empty-plan retries
                    continue
                terminate = False
                for i, action in enumerate(bundle.actions):
                    if self._cancel.is_set():
                        terminate = True
                        break
                    self.get_logger().info('step %d: %s -- %s'
                                           % (step, self._action_brief(action),
                                              action.rationale or ''))
                    self._activity('step_start', step=step,
                                   action=self._action_brief(action),
                                   rationale=action.rationale or '')
                    if orch.is_terminal(action.kind):       # DONE
                        self.get_logger().info('VLM mission finished: %s' % action.name)
                        self._publish_notes(target)
                        step += 1            # count the terminal action too
                        terminate = True
                        break
                    # anytime: launch the NEXT replan while this (last-of-batch) action
                    # executes, so it is ready at the commit-point -> no wasted idle.
                    if orch.should_launch_lead_replan(i, len(bundle.actions),
                                                      self.async_replan, pending is not None):
                        pending = self._planner_pool.submit(self._compute_plan, target, step + 1)
                    t0 = time.monotonic()
                    self._exec_note = None
                    ok = self._dispatch(action, bundle.pixels)
                    # honest execution detail (e.g. a clamped DRIVE_FORWARD): the
                    # model must learn what PHYSICALLY happened, not just ok/failed
                    detail = self._exec_note
                    self._exec_note = None
                    self.notes.add_fact('%s%s -> %s%s' % (
                        action.name,
                        (' ' + action.rationale) if action.rationale else '',
                        'ok' if ok else 'failed',
                        (' | ' + detail) if detail else ''))
                    # Зацикливание модель по своим заметкам не опознаёт: каждый
                    # отдельный шаг отчитывается «ok», и серия из семнадцати
                    # поворотов подряд выглядит как семнадцать успехов. Считаем
                    # повторы сами и говорим прямым текстом — это наблюдение,
                    # на которое планировщику есть чем ответить.
                    if action.name == self._last_action_name:
                        self._same_action_run += 1
                    else:
                        self._last_action_name, self._same_action_run = action.name, 1
                    if self._same_action_run >= REPEAT_WARN_AFTER:
                        self.notes.add_fact(
                            'WARNING: %s repeated %d times in a row and the view is '
                            'not changing -- you are stuck in a loop, choose a '
                            'DIFFERENT action'
                            % (action.name, self._same_action_run))
                    self._activity('step_result', step=step,
                                   action=self._action_brief(action),
                                   result='ok' if ok else 'failed',
                                   detail=detail or '',
                                   duration_s=round(time.monotonic() - t0, 2))
                    self._publish_notes(target)
                    step += 1
                    dt = time.monotonic() - t0
                    if dt < self.min_step_s:      # don't hammer on instant-reached skills
                        time.sleep(self.min_step_s - dt)
                    if step >= self.max_steps:
                        break
                if terminate or step >= self.max_steps:
                    break
                # commit-point: adopt the plan computed during execution
                bundle = self._next_bundle(pending, target, step)
                pending = None
            cancelled = self._cancel.is_set()
            if cancelled:
                self._dispatch_stop()
            self.get_logger().info('VLM mission ended after %d steps%s%s' % (
                step, ' (CANCELLED by operator)' if cancelled else '',
                ' (DEGRADED: ran in FLAT fallback)' if self._degrade.degraded else ''))
            self._activity('mission_end', target=target, steps=step,
                           degraded=bool(self._degrade.degraded),
                           cancelled=bool(cancelled))
        finally:
            # Join the in-flight replan BEFORE clearing _busy, so a stale pool worker
            # can never write this mission's circuit-breaker / degrade-latch / notes
            # after the NEXT mission has reinitialised them (cancel() is best-effort:
            # an already-running future ignores it, so we must wait it out).
            if pending is not None:
                pending.cancel()
                try:
                    pending.result(timeout=self.detect_timeout_s + self.vlm_timeout_s + 2.0)
                except Exception:
                    pass
            self._busy = False

    # ---- dispatch one atomic action to the matching FLAT skill ----
    @staticmethod
    def _action_brief(a):
        """Compact human label incl. the numeric argument the VLM chose (for logs)."""
        if a.kind == TURN:
            return 'TURN %+.2frad' % a.turn_yaw_rad
        if a.kind == DRIVE_FORWARD:
            return 'DRIVE_FORWARD %+.2fm' % a.forward_dist_m
        if a.kind == DRIVE_TO_VISIBLE:
            return 'DRIVE_TO_VISIBLE mark=%d' % a.mark_id
        return a.name

    def _forward_clearance_m(self):
        """Forward clearance from the latest /scan via the pure geometry helper;
        None when there is no scan (topic disabled / sensor dark)."""
        scan = self._scan
        if scan is None:
            return None
        return orch.forward_clearance(scan.ranges, scan.angle_min,
                                      scan.angle_increment,
                                      self.forward_corridor_half_width_m)

    def _dispatch(self, action, cand_pixels):
        # Причину отказа гасим на входе: её выставляет _send_and_wait, а часть
        # ветвей до исполнителя не доходит вовсе (нечем ехать, метка пропала). С
        # прошлой причиной в поле такой шаг получил бы в заметки чужое
        # объяснение — худший вид неправды, потому что выглядит осмысленно.
        self._skill_reason = ''
        if action.kind in (TURN, DRIVE_FORWARD):
            pose = self._motion_pose()
            if pose is None:
                self.get_logger().warn(
                    'no TF for relative motion (%s->%s or %s->%s); skip motion'
                    % (self.map_frame, self.robot_frame,
                       self.motion_fallback_frame, self.robot_frame))
                self._exec_note = ('blocked: the robot does not know where it is '
                                   '(no localisation)')
                return False
            if action.kind == DRIVE_FORWARD:
                ok = self._drive_forward_clamped(action, pose)
            else:
                gx, gy, gyaw = orch.relative_goal(pose[0], pose[1], pose[2], action)
                ok = self._send_goto(gx, gy, gyaw, frame_id=pose[3])
            self._append_motion_result(action, pose, ok)
            return ok
        if action.kind == DRIVE_TO_VISIBLE:
            ok = self._send_approach_mark(action.mark_id, action.arg_label, cand_pixels)
            if not ok and self._skill_reason and not self._exec_note:
                self._exec_note = self._skill_reason
            return ok
        if action.kind == DETECT_ALL:
            return self._do_detect_all()
        return False

    def _append_motion_result(self, action, pose_before, ok):
        """Дописать в заметку ИЗМЕРЕННЫЙ результат движения и причину отказа.

        Это ответ на самую дорогую из наблюдавшихся поломок: одометрия врёт,
        когда робот буксует носом в стену. Nav2 считает цель достигнутой,
        исполнитель рапортует «ok», а робот не сдвинулся — и модель получала
        подтверждение, что поворот выполнен, глядя на ту же самую стену. Теперь
        в заметку идёт разница поз ДО и ПОСЛЕ: «rotated +0.03 rad» при заказе
        1.57 — наблюдение, на которое модели есть чем ответить.

        Мерим в том же кадре, в котором ставили цель (map или odom): смешивать
        их нельзя, map->odom правится SLAM прямо во время движения.
        """
        after = self._lookup_robot_pose(pose_before[3], timeout_s=0.5)
        moved = ''
        if after is not None:
            dx, dy = after[0] - pose_before[0], after[1] - pose_before[1]
            dist = math.hypot(dx, dy)
            dyaw = orch.wrap_angle(after[2] - pose_before[2])
            moved = ('rotated %+.2f rad' % dyaw if action.kind == TURN
                     else 'moved %.2f m' % dist)
        parts = [p for p in (self._exec_note, moved,
                             '' if ok else self._skill_reason) if p]
        self._exec_note = '; '.join(parts) if parts else None

    def _drive_forward_clamped(self, action, pose):
        """Исполнить DRIVE_FORWARD и честно отчитаться в _exec_note.

        Ветви в порядке предпочтения:
          0. запрос отрицательный -> задний ход; короче минимального шага -> отказ;
          1. по курсу свободно на всю длину -> едем прямо;
          2. впереди препятствие -> ищем по карте свободную точку примерно по
             курсу и отдаём её Nav2; маршрут в обход прокладывает он (см.
             _detour_goal). Берём, только если она дальше обрезанного хода;
          3. объезда нет -> едем прямо, насколько пустил лидар;
          4. и на минимальный шаг места нет -> честный отказ с причиной.

        Обрезка лидаром — слой безопасности исполнителя, а не правка плана: Nav2
        принимает цель внутри стены, допуск NavFn сдвигает её к границе раздутия,
        движение «удаётся», и пять повторов доводят робота до срабатывания защиты
        (замер: s7 закончился на минимуме скана 0.154 м). Расстояния в заметках
        округлены до 0.1 м, чтобы повторный упор давал ТУ ЖЕ запись и буфер
        заметок дедуплицировал её, а не переполнялся.
        """
        asked = float(action.forward_dist_m)
        # ЗАДНИЙ ХОД — отдельная ветка, и без неё он не работал ВООБЩЕ. Клэмп
        # ниже считает свободное место передним лидаром и обрезает им запрос;
        # для отрицательного запроса min() всегда возвращал сам запрос, а затем
        # проверка порога («меньше минимального шага») отбрасывала его как
        # заведомо слишком короткий. В журнале это выглядело как
        # «DRIVE_FORWARD -0.40m -> failed | blocked», и робот, упёршийся носом,
        # не мог отъехать — ни одна из двух ветвей движения ему не оставалась.
        # Назад едем вслепую: заднего датчика на роботе нет, поэтому шаг жёстко
        # ограничен REVERSE_MAX_M — этого хватает, чтобы освободить место для
        # разворота, и мало, чтобы въехать во что-то позади.
        if asked < 0.0:
            drive = max(asked, -REVERSE_MAX_M)
            if abs(drive) < self.min_drive_m:
                self._exec_note = ('blocked: a reverse step of %.2f m is shorter '
                                   'than the %.2f m minimum, the robot will not '
                                   'move' % (abs(drive), self.min_drive_m))
                return False
            x, y, yaw, frame = pose
            ok = self._send_goto(x + drive * math.cos(yaw), y + drive * math.sin(yaw),
                                 yaw, frame_id=frame)
            # «tried to», а не «backed up»: рядом в заметке встанет измеренное
            # перемещение, и утверждать успех до него — значит спорить с ним же.
            self._exec_note = ('tried to back up %.2f m blind (there is no rear '
                               'sensor)' % abs(drive))
            return ok
        # Слишком короткий запрос отбивается СРАЗУ, до всякой геометрии: цель
        # попадает внутрь допуска Nav2 (xy_goal_tolerance 0.20), controller_server
        # мгновенно рапортует «Reached the goal», и модель получает «ok» на
        # движение, которого не было. Наблюдалось у оператора ровно так: сорок
        # шагов DRIVE_FORWARD +0.18m подряд, робот стоит, кадр не меняется.
        if asked < self.min_drive_m:
            self._exec_note = ('blocked: a step of %+.2f m is below the %.2f m '
                               'minimum, the robot will not move at all'
                               % (asked, self.min_drive_m))
            return False
        clearance = self._forward_clearance_m()   # scan-frame (camera, robot front)
        if clearance is None:                     # no scan info -> old behaviour
            drive = asked
        else:
            # standoff already contains the camera->footprint-front offset margin
            drive = min(asked, max(0.0, clearance - self.forward_standoff_m))
        x, y, yaw, frame = pose
        # 1) Прямо свободно на всю запрошенную длину — едем прямо, без затей.
        if drive >= asked - 1e-6:
            return self._send_goto(x + drive * math.cos(yaw),
                                   y + drive * math.sin(yaw), yaw, frame_id=frame)
        # 2) Впереди мешает препятствие. Раньше здесь обрезали шаг лидаром, а при
        # совсем коротком остатке отказывали — то есть возлагали на модель работу,
        # которая ей не по силам и не её: понять по картинке, что мешает, и
        # придумать объезд. Ровно для этого в стеке есть Nav2: он знает
        # построенную карту и умеет прокладывать маршрут. Спрашиваем карту о
        # свободной точке примерно по курсу и отдаём её Nav2 — как добраться
        # (обогнуть стул, выйти через дверь) решает он. Берём объезд только если
        # он ДАЛЬШЕ обрезанного хода: иначе прямой шаг честнее и короче.
        detour = (self._detour_goal(pose, asked, drive) if self.nav_detour
                  else None)
        if detour is not None and detour[3] > drive + 1e-6:
            gx, gy, bearing, dist = detour
            ok = self._send_goto(gx, gy, orch.wrap_angle(yaw + bearing),
                                 frame_id=frame)
            self._exec_note = (
                'an obstacle is ~%.1f m straight ahead, so the nav stack was sent '
                'around it, to free floor %.2f m away and %+.0f deg off your '
                'heading' % (clearance if clearance is not None else -1.0, dist,
                             math.degrees(bearing)))
            self.get_logger().info(
                'DRIVE_FORWARD %.2fm: прямо %.2fm, объезд через (%.2f, %.2f), '
                '%+.0f°' % (asked, drive, gx, gy, math.degrees(bearing)))
            return ok
        # 3) Объезда нет. Едем прямо, насколько пустил лидар...
        if drive >= self.min_drive_m:
            ok = self._send_goto(x + drive * math.cos(yaw),
                                 y + drive * math.sin(yaw), yaw, frame_id=frame)
            self._exec_note = ('asked %+.2f m but an obstacle is ~%.1f m ahead, so '
                               'only %+.2f m was attempted'
                               % (asked, clearance, drive))
            self.get_logger().info('DRIVE_FORWARD clamped %.2f -> %.2fm (clearance %.2fm)'
                                   % (asked, drive, clearance))
            return ok
        # 4) ...а если и на минимальный шаг места нет — отказываем ЧЕСТНО. Иначе
        # Nav2 принимает цель внутри своего допуска, мгновенно объявляет её
        # достигнутой, и модель получает «ok» на движение, которого не было; она
        # повторяет ту же команду, пока не кончатся шаги.
        # Формулировка зависит от того, БЫЛА ли карта: сказать «карта не
        # показывает свободного места», не получив ни одной сетки, — врать
        # модели о причине, а причина здесь единственное, на что она опирается.
        self._exec_note = (
            'blocked: no route forward -- a wall is ~%.1f m ahead and %s; turn or '
            'back off'
            % (clearance if clearance is not None else -1.0,
               'the map shows no free spot ahead to route to'
               if self._map is not None else 'no map is available to route by'))
        self.get_logger().warn(
            'DRIVE_FORWARD %.2fm refused: свободно %.2fm < min_drive %.2fm '
            '(clearance %s), объезда не нашлось'
            % (asked, drive, self.min_drive_m,
               ('%.2fm' % clearance) if clearance is not None else 'н/д'))
        return False

    def _detour_goal(self, pose, asked, free_straight_m):
        """Ближайшая СВОБОДНАЯ по карте точка примерно по курсу: (x, y, азимут,
        расстояние) или None.

        Проверяется только сама цель (и круг GOAL_CLEAR_RADIUS_M вокруг неё), а
        не путь до неё: путь — работа Nav2, в том и смысл. Кадр цели тот же, в
        котором пришла поза; если это не кадр карты, объезд не строим — карту
        читать не в чем.

        free_straight_m — измеренный лидаром свободный ход по курсу. Нужен, чтобы
        карта не отменяла датчик: сетка SLAM обновляется реже скана и вполне может
        показывать свободной клетку, перед которой лидар прямо сейчас видит стену
        в 0.4 м. Поэтому цель отбрасывается, если она попадает В КОРИДОР, который
        робот выметает телом (та же геометрия, что в forward_clearance), и при
        этом лежит дальше измеренного свободного хода. Цели вбок от коридора
        лидар не видит вовсе — там авторитет у карты.
        """
        grid = self._map
        if grid is None or not _HAVE_CV or pose[3] != self.map_frame:
            return None
        w, h = int(grid.info.width), int(grid.info.height)
        res = float(grid.info.resolution)
        if w <= 0 or h <= 0 or res <= 0.0:
            return None
        try:
            data = np.asarray(grid.data, dtype=np.int16).reshape(h, w)
        except (ValueError, TypeError):
            return None
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        rad_cells = max(1, int(math.ceil(GOAL_CLEAR_RADIUS_M / res)))
        x, y, yaw = pose[0], pose[1], pose[2]
        free_straight = max(0.0, float(free_straight_m))
        for bearing, dist in orch.drive_goal_candidates(
                asked, self.min_drive_m, self.detour_max_bearing_rad):
            # Объезд имеет смысл только ДАЛЬШЕ обрезанного прямого хода — ровно
            # это проверяет вызывающий. Без такой же проверки ЗДЕСЬ поиск
            # заканчивался на первой же близкой цели по курсу (её пропускает
            # фильтр коридора ниже, ведь она внутри измеренного свободного
            # места), вызывающий её отбрасывал как не лучшую, и веер отклонений
            # не рассматривался ВООБЩЕ. Ветка объезда была почти мертва: при
            # запросе 1.2 м и стене в 1.1 м возвращалась цель прямо по курсу на
            # 0.70 м, то есть тот же подкат к стене, что и до правки.
            if dist <= free_straight + 1e-6:
                continue
            # Внутри выметаемого коридора верим лидару, а не карте (см. docstring).
            if (abs(dist * math.sin(bearing)) <= self.forward_corridor_half_width_m
                    and dist * math.cos(bearing) > free_straight):
                continue
            gx = x + dist * math.cos(yaw + bearing)
            gy = y + dist * math.sin(yaw + bearing)
            cx = int((gx - ox) / res)
            cy = int((gy - oy) / res)
            if not (rad_cells <= cx < w - rad_cells
                    and rad_cells <= cy < h - rad_cells):
                continue
            patch = data[cy - rad_cells:cy + rad_cells + 1,
                         cx - rad_cells:cx + rad_cells + 1]
            # Свободно = всё известно и ничто не занято. Неизвестные клетки не
            # берём в цель намеренно: Nav2 в них не проложит маршрут, и «объезд»
            # выродился бы в отказ на шаг позже.
            if patch.min() >= 0 and patch.max() < 50:
                return gx, gy, bearing, dist
        return None

    def _goal_id(self):
        return uuid.uuid4().hex

    def _send_goto(self, x, y, yaw, frame_id=None):
        g = GoToPose.Goal()
        g.request_id = self._goal_id()
        g.mission_epoch = self._epoch
        ps = PoseStamped()
        ps.header.frame_id = frame_id or self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        qx, qy, qz, qw = _yaw_to_quat(yaw)
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        g.target_pose = ps
        g.xy_tolerance = 0.25
        g.yaw_tolerance = 0.5
        return self._send_and_wait(orch.SKILL_GO_TO_POSE, g)

    def _send_approach(self, label):
        g = ApproachDetection.Goal()
        g.request_id = self._goal_id()
        g.mission_epoch = self._epoch
        g.target_label = label or ''
        g.approach_offset = self.approach_offset
        g.max_pixel_age_s = 1.5
        return self._send_and_wait(orch.SKILL_APPROACH, g)

    def _send_approach_mark(self, mark_id, label, cand_pixels):
        """DRIVE_TO_VISIBLE(mark_id): inject the chosen candidate's pixel onto
        /target_pixel (kept fresh by a background republisher so ApproachDetection's
        freshness gate stays satisfied through the whole drive), then approach.
        Resolves against the plan's pixel snapshot, not live state (a concurrent
        replan may already be overwriting self._cand_pixels)."""
        pt = (cand_pixels or {}).get(int(mark_id))
        if pt is None:
            self.get_logger().warn('DRIVE_TO_VISIBLE: no pixel for mark %s' % mark_id)
            self._exec_note = ('blocked: mark %s is not in the current detection '
                               'list any more' % mark_id)
            return False
        if not distance_is_known(pt.z):
            self.get_logger().warn(
                'DRIVE_TO_VISIBLE: mark %s has unknown depth; refusing ApproachDetection'
                % mark_id)
            self._exec_note = ('blocked: the depth camera has no range for mark %s, '
                               'so the robot cannot drive to it' % mark_id)
            return False
        stop = threading.Event()

        def _republish():
            while not stop.is_set():
                px = PointStamped()
                px.header.frame_id = self.camera_frame
                px.header.stamp = self.get_clock().now().to_msg()
                px.point.x, px.point.y, px.point.z = float(pt.x), float(pt.y), float(pt.z)
                self._pixel_pub.publish(px)
                time.sleep(0.1)
        pub_thread = threading.Thread(target=_republish, daemon=True)
        pub_thread.start()
        try:
            return self._send_approach(label)
        finally:
            stop.set()

    def _call_action(self, ac, goal, timeout_s):
        """Send a goal and block (worker thread) for its result message; None on
        non-accept / timeout. Mirrors _send_and_wait but returns the result."""
        gh_box, gh_evt = {}, threading.Event()

        def _gh_cb(fut):
            gh_box['gh'] = fut.result()
            gh_evt.set()
        ac.send_goal_async(goal).add_done_callback(_gh_cb)
        if not gh_evt.wait(timeout_s) or gh_box.get('gh') is None or not gh_box['gh'].accepted:
            return None
        res_box, res_evt = {}, threading.Event()

        def _res_cb(fut):
            res_box['res'] = fut.result()
            res_evt.set()
        gh_box['gh'].get_result_async().add_done_callback(_res_cb)
        if not res_evt.wait(timeout_s):
            return None
        return getattr(res_box.get('res'), 'result', None)

    def _do_detect_all(self):
        """DETECT_ALL: run the detector over a broad object vocabulary (empty query =>
        detect-all on the server) and record what is in view -- objects + their
        classes -- into the notes the VLM reads next replan. Perception only; the
        robot does not move. Returns True if anything was detected."""
        if not self._detect.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('DETECT_ALL: detector server unavailable')
            return False
        g = DetectTarget.Goal()
        g.request_id = self._goal_id()
        g.mission_epoch = self._epoch
        g.query = ''                      # empty query => broad-vocabulary detection
        g.render_setofmark = True
        g.conf_threshold = self.detect_all_conf
        res = self._call_action(self._detect, g, self.detect_timeout_s)
        cands = getattr(res, 'candidates', None) if res is not None else None
        if not cands:
            self.notes.add_fact('DETECT_ALL: nothing detected in view')
            self._activity('detect_all', objects=[])
            return False
        if getattr(res, 'annotated', None) is not None and res.annotated.data:
            self._publish_view(self._setofmark_pub, bytes(res.annotated.data))
        self._remember_objects(cands)
        seen = ', '.join('%s(%.2f)' % (c.label, c.confidence) for c in cands)
        self.notes.add_fact('objects in view: ' + seen)
        self.get_logger().info('DETECT_ALL: %d object(s): %s' % (len(cands), seen))
        self._activity('detect_all',
                       objects=[{'label': c.label,
                                 'score': round(float(c.confidence), 2)}
                                for c in cands])
        return True

    def _dispatch_stop(self):
        g = Stop.Goal()
        g.request_id = self._goal_id()
        g.mission_epoch = self._epoch
        g.mode = Stop.Goal.SOFT_STOP
        return self._send_and_wait(orch.SKILL_STOP, g)

    def _send_and_wait(self, skill, goal):
        """Send a skill goal and block (in the worker thread) for the result,
        using events set by the executor-thread done-callbacks (loopback-safe).

        Побочно заполняет self._skill_reason — ПОЧЕМУ не получилось. Раньше
        отказ исполнителя схлопывался в один бит: модель видела «failed» и не
        могла отличить «Nav2 не построил маршрут» от «сервер навыка не
        отвечает», а значит не имела повода сменить действие и повторяла его до
        конца миссии. Причина уходит в заметки и попадает в следующий промпт.
        """
        self._skill_reason = ''
        ac = self._ac[skill]
        if not ac.wait_for_server(timeout_sec=self.skill_wait_s):
            self.get_logger().warn('skill %s server unavailable' % skill)
            self._skill_reason = ('the %s skill server is not responding '
                                  '(robot software problem, not your choice)' % skill)
            return False
        gh_box = {}
        gh_evt = threading.Event()

        def _gh_cb(fut):
            gh_box['gh'] = fut.result()
            gh_evt.set()
        ac.send_goal_async(goal).add_done_callback(_gh_cb)
        if not gh_evt.wait(self.skill_wait_s) or gh_box.get('gh') is None or not gh_box['gh'].accepted:
            self.get_logger().warn('skill %s goal not accepted' % skill)
            self._skill_reason = 'the %s skill refused the goal' % skill
            return False
        res_box = {}
        res_evt = threading.Event()

        def _res_cb(fut):
            res_box['res'] = fut.result()
            res_evt.set()
        gh_box['gh'].get_result_async().add_done_callback(_res_cb)
        if not res_evt.wait(self.result_timeout_s):
            self.get_logger().warn('skill %s result timeout' % skill)
            self._skill_reason = ('the robot was still driving after %.0f s and the '
                                  'move was given up' % self.result_timeout_s)
            return False
        res = res_box.get('res')
        outcome = getattr(getattr(res, 'result', None), 'outcome', None)
        if outcome == 0:                      # 0 == SUCCEEDED across the skill results
            return True
        # 1 = ABORTED, 2 = PREEMPTED в GoToPose/ApproachDetection/Stop.
        self._skill_reason = (
            'the nav stack could not get there: no route on the map, or the drive '
            'was aborted on the way' if outcome == 1 else
            'the move was interrupted (%s)' % ('preempted' if outcome == 2
                                               else 'outcome %s' % outcome))
        return False

    def _publish_notes(self, target):
        m = Notes()
        m.header.stamp = self.get_clock().now().to_msg()
        m.mission_epoch = self._epoch
        m.summary = self.notes.summary()
        m.facts = self.notes.facts
        m.token_estimate = self.notes.token_estimate()
        self.notes_pub.publish(m)
        # Mirror into the activity trace so the dashboard reads the VLM's memory
        # without a dependency on the Notes message type.
        self._activity('notes', target=target, summary=m.summary,
                       facts=list(m.facts), token_estimate=int(m.token_estimate))


def main():
    rclpy.init()
    from rclpy.executors import MultiThreadedExecutor
    node = PlannerOrchestrator()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Пулы гасим ДО destroy_node: рабочий поток холостой детекции может в этот
        # момент ждать ответа action-сервера, а обращение к уничтоженному узлу из
        # него роняет процесс уже на выходе — и маскирует настоящую причину останова.
        node.shutdown_pools()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
