"""Pure orchestration helpers (Phase 4). ROS-free so the atomic-action -> skill
dispatch mapping and the relative-motion geometry are unit-testable.

Each atomic action maps to an executive skill (or is handled by the orchestrator):
  TURN / DRIVE_FORWARD -> GoToPose at a pose computed RELATIVE to the robot's real
                          pose (the relative motion is the action's arg, not a
                          VLM-emitted coordinate)
  DRIVE_TO_VISIBLE     -> ApproachDetection
  DETECT_ALL           -> orchestrator-local detector call (no Pi skill)
  DONE                 -> terminal (no skill)
The executive owns motion + safety; the VLM/orchestrator never touches cmd_vel.
A safe Stop is kept as an executive-owned fallback (see SKILL_STOP) -- it is NOT a
VLM action, so the model can never command a stop, only DONE.
"""
from __future__ import annotations

import math
from typing import Tuple

from planner_orchestrator.planner_logic import (
    Action, TURN, DRIVE_FORWARD, DRIVE_TO_VISIBLE, DETECT_ALL, DONE,
)

SKILL_GO_TO_POSE = 'go_to_pose'
SKILL_APPROACH = 'approach_detection'
SKILL_STOP = 'stop'           # executive-owned safety fallback, not a VLM action
SKILL_NONE = ''

_SKILL_MAP = {
    TURN: SKILL_GO_TO_POSE,
    DRIVE_FORWARD: SKILL_GO_TO_POSE,
    DRIVE_TO_VISIBLE: SKILL_APPROACH,
    DETECT_ALL: SKILL_NONE,       # handled directly by the orchestrator
    DONE: SKILL_NONE,
}


def skill_for_action(kind: int) -> str:
    """Executive skill action name that executes this atomic action ('' = none)."""
    return _SKILL_MAP.get(kind, SKILL_NONE)


def is_terminal(kind: int) -> bool:
    """Only DONE ends the VLM mission (STOP is no longer a VLM action)."""
    return kind == DONE


def should_launch_lead_replan(action_index: int, batch_len: int,
                              async_enabled: bool, already_pending: bool) -> bool:
    """Phase 4.6 anytime policy: launch the NEXT replan while the LAST action of the
    current batch is executing (lead-time), exactly once per batch, and only when
    async replanning is on and no replan is already in flight. Adopting the result
    happens later at the commit-point (batch boundary) so an in-flight action is
    never interrupted (consensus-horizon)."""
    return (async_enabled and not already_pending
            and batch_len > 0 and action_index == batch_len - 1)


def describe_occupancy_grid(width: int, height: int, resolution: float,
                            robot_xy: Tuple[float, float],
                            n_free: int, n_occupied: int, n_unknown: int) -> str:
    """Human description of the SLAM occupancy map attached as the 2nd image, so the
    VLM can read it: what kind of map it is, scale, legend, and where the robot is.
    Pure text (the pixels are rendered ROS-side). explored% = mapped / total cells.

    Легенда намеренно объясняет не только ЦВЕТ, но и его смысл для поиска: белое —
    уже осмотренное, серое — куда ни один датчик не заглядывал. Без этой фразы
    карта модели практически ничего не давала: она видела картинку, но не знала,
    что серое — это и есть «где ещё не были», то есть единственное место, где
    ненайденный предмет может находиться. Наблюдалось ровно так — робот кружил по
    уже разведанному белому пятну, имея под боком неисследованные комнаты.
    """
    total = max(1, width * height)
    explored = 100.0 * (n_free + n_occupied) / total
    span_x = width * resolution
    span_y = height * resolution
    rx, ry = robot_xy
    return (
        'Top-down SLAM occupancy map (2nd image), built by the robot itself as it '
        'drives. %dx%d cells at %.3f m/cell (~%.1fm x %.1fm), north-up, metric. '
        'Legend: white=free floor you have ALREADY seen and can drive on; '
        'black=wall/obstacle; gray=unknown, never seen by any sensor. '
        'Red dot=you, red line=the way you face. '
        'Anything you have not found yet is in the GRAY, so explore by heading '
        'toward gray and into rooms you have not entered -- driving around the '
        'white searches ground you have already searched. '
        'Explored %.0f%% of this view. You are at map (%.2f, %.2f) m.'
        % (width, height, resolution, span_x, span_y, explored, rx, ry))


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def rooms_to_map_frame(rooms: dict, ox: float, oy: float, oyaw: float) -> dict:
    """Комнаты из МИРОВЫХ координат в кадр `map`. {имя: (x0, x1, y0, y1)}.

    Комнаты приходят из worlds.yaml, то есть в координатах мира Gazebo, а карта
    SLAM живёт в кадре `map`, начало которого — ПОЗА СТАРТА робота. Пока эту
    разницу не учитывали, подписи ехали ровно на вектор старта: в мире house
    спавн (-7.0, 0.0), и прямоугольники комнат рисовались на семь метров левее
    настоящих стен — на карте это выглядело как аккуратная сетка комнат,
    съехавшая с занятого белым плана здания.

    AABB под поворотом перестаёт быть AABB, поэтому при ненулевом yaw берётся
    описанный прямоугольник по четырём повёрнутым углам: подпись останется в
    своей комнате, рамка станет чуть шире. Во всех текущих мирах курс старта
    нулевой, и путь точный.
    """
    if not rooms:
        return {}
    if abs(ox) < 1e-9 and abs(oy) < 1e-9 and abs(oyaw) < 1e-9:
        return dict(rooms)
    cos_y, sin_y = math.cos(-oyaw), math.sin(-oyaw)
    out = {}
    for name, (x0, x1, y0, y1) in rooms.items():
        xs, ys = [], []
        for wx, wy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            dx, dy = wx - ox, wy - oy
            xs.append(dx * cos_y - dy * sin_y)
            ys.append(dx * sin_y + dy * cos_y)
        out[name] = (min(xs), max(xs), min(ys), max(ys))
    return out


def detection_map_xy(rx: float, ry: float, ryaw: float, depth_m: float,
                     u_px: float, fx: float, cx: float) -> Tuple[float, float]:
    """Куда в кадре `map` попадает детекция: поза робота + пиксель + глубина.

    Глубина RealSense — это z вдоль оптической оси, а не радиус, поэтому
    смещение вбок считается подобием, без тригонометрии: lateral =
    depth * (u - cx) / fx. Дальше поворот в кадр карты, где forward =
    (cos yaw, sin yaw), а right = (sin yaw, -cos yaw): пиксель правее центра
    даёт положительный lateral, то есть смещение ВПРАВО от курса.

    Смещение камеры относительно base_link (единицы сантиметров) не
    учитывается: оно заведомо меньше ошибки самой привязки.
    """
    lateral = depth_m * (u_px - cx) / fx
    return (rx + depth_m * math.cos(ryaw) + lateral * math.sin(ryaw),
            ry + depth_m * math.sin(ryaw) - lateral * math.cos(ryaw))


def drive_goal_candidates(asked_m: float, min_drive_m: float,
                          max_bearing_rad: float = 1.05,
                          bearing_step_rad: float = 0.35,
                          dist_step_m: float = 0.25):
    """Порядок перебора целей для DRIVE_FORWARD, когда прямо ехать некуда:
    (отклонение от курса в радианах, расстояние в метрах), от самого желанного к
    наименее.

    Смысл: команду «вперёд» исполняет Nav2 по ПОСТРОЕННОЙ КАРТЕ, а не слепой
    рывок по курсу. Если точка прямо по курсу лежит в стене, планировщику
    предлагается ближайшая свободная точка примерно в том же направлении, и
    маршрут до неё Nav2 прокладывает сам — в обход препятствия. Поэтому порядок
    ровно такой: сперва минимальное отклонение от курса (это всё ещё «вперёд»),
    и уже внутри него — от дальней точки к ближней, потому что далеко полезнее.

    Веер ограничен ±max_bearing (по умолчанию 60°): дальше это уже не «вперёд», а
    поворот, и решение поворачивать принимает модель, а не исполнитель.
    """
    asked = max(float(asked_m), float(min_drive_m))
    bearings = [0.0]
    b = float(bearing_step_rad)
    while b <= float(max_bearing_rad) + 1e-9:
        bearings += [b, -b]
        b += float(bearing_step_rad)
    out = []
    for bearing in bearings:
        d = asked
        while d >= float(min_drive_m) - 1e-9:
            out.append((bearing, round(d, 3)))
            d -= float(dist_step_m)
    return out


def relative_goal(x: float, y: float, yaw: float, action: Action) -> Tuple[float, float, float]:
    """Absolute (gx, gy, gyaw) for a relative TURN/DRIVE_FORWARD from the robot's
    real pose. TURN rotates in place; DRIVE_FORWARD advances along the heading.
    Any other action returns the current pose (caller won't use it)."""
    if action.kind == TURN:
        return (x, y, wrap_angle(yaw + action.turn_yaw_rad))
    if action.kind == DRIVE_FORWARD:
        d = action.forward_dist_m
        return (x + d * math.cos(yaw), y + d * math.sin(yaw), yaw)
    return (x, y, yaw)


def forward_clearance(ranges, angle_min: float, angle_increment: float,
                      corridor_half_width_m: float):
    """Distance the robot can advance along its heading before its swept corridor
    hits the nearest scan return; None when the scan says nothing about that
    corridor (no rays, or all NaN/inf).

    Geometry, not a cone: a ray at bearing theta with range r only blocks forward
    motion if its hit point lies inside the strip the robot's body sweeps, i.e.
    |r*sin(theta)| <= corridor_half_width_m; the blocking depth is then
    r*cos(theta) (rear hits, cos<=0, never block). A fixed angular cone would
    get this wrong on both ends: at 0.4 m it misses a wall edge 0.2 m off-axis
    (outside the cone, inside the body sweep), and at 4 m it "blocks" on a
    doorframe 0.8 m off-axis that the robot clears with half a metre to spare.

    Why this exists at all: Nav2 accepts a goal pressed against (or inside) a
    wall -- NavFn's tolerance just shifts it to the nearest reachable cell, the
    drive "succeeds" flush with the inflation boundary, and repeating the same
    forward command walks the robot into the collision guard. Measured in the
    house world (s7): five DRIVE_FORWARD into a blank partition ended at scan
    min 0.154 m. The executive owns safety, so the clamp lives here, executive-
    side; the planner's choice is never edited, only truncated by physics, and
    the truncation is reported back honestly in the step note.
    """
    if not ranges or angle_increment == 0.0:
        return None
    best = None
    for i, r in enumerate(ranges):
        if r is None or not math.isfinite(r) or r <= 0.0:
            continue
        theta = angle_min + i * angle_increment
        depth = r * math.cos(theta)
        if depth <= 0.0:
            continue
        if abs(r * math.sin(theta)) > corridor_half_width_m:
            continue
        if best is None or depth < best:
            best = depth
    return best
