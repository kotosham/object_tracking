#!/usr/bin/env python3
"""Planner Orchestrator (Phase 4): VLM-mode planner over the FLAT executive.

Replans every N ATOMIC steps. Each step: build an Observation from the latest
detections + notes (+ the camera frame for the VLM), ask the client (mock or
OpenAI-compatible) for up to N atomic actions, and dispatch each:
  TURN / DRIVE_FORWARD -> GoToPose at a pose RELATIVE to the robot's real pose
  DRIVE_TO_VISIBLE     -> ApproachDetection (drive to a detected object via Nav)
  DETECT_ALL           -> broad-vocab detector call -> objects + classes into notes
  DONE                 -> finish
The vocabulary is deliberately small (raw motion + perception) so the VLM does its
own navigation reasoning -- a fair comparison against the FLAT policy. The VLM is
never on the reactive path; the executive owns motion + safety. A
per-call timeout + circuit-breaker degrade VLM->FLAT on loss. Mock-first: with
use_mock (or no credentials anywhere) the whole loop runs in sim/CI with no API
key. Trigger a mission by publishing the target on /vlm_mission (std_msgs/String).

Real-VLM credentials: set the ROS params vlm_base_url / vlm_api_key / vlm_model,
OR (preferred for secrets) export the environment variables VLM_BASE_URL /
VLM_API_KEY / VLM_MODEL -- env fills in any param left blank, so keys never need
to live in a launch file. A base_url from either source auto-engages the real
OpenAI-compatible client unless use_mock:=true is set explicitly.
"""
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
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import (Buffer, ConnectivityException, ExtrapolationException,
                     LookupException, TransformListener)

from ar_project_msgs.action import ApproachDetection, GoToPose, Stop
from object_tracking_msgs.action import DetectTarget
from object_tracking_msgs.msg import Notes

from fleet_comms.qos import detection_stream_nodeadline

from fleet_comms.heartbeat import HeartbeatPublisher
from planner_orchestrator import orchestration as orch
from planner_orchestrator.planner_logic import (
    Candidate, CircuitBreaker, DegradationLatch, NotesBuffer, Observation,
    DETECT_ALL, DRIVE_FORWARD, DRIVE_TO_VISIBLE, TURN,
)
from planner_orchestrator.vlm_client import make_client

try:
    import cv2
    import numpy as np
    from cv_bridge import CvBridge
    _HAVE_CV = True
except Exception:                       # mock mode needs no image pipeline
    _HAVE_CV = False


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
        self.declare_parameter('replan_every_n', 3)
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
        # Confidence floor for detections the planner acts on. 0.0 keeps the detector's
        # own default (0.25). Raise it (e.g. 0.5) to ignore weak/edge-of-frame matches --
        # a natural-language query like "ride to bus" scores ~0.45 vs ~0.66 for "bus",
        # so a higher floor with a bare label avoids acting on a marginal glimpse.
        self.declare_parameter('detect_conf', 0.0)
        self.declare_parameter('camera_frame', 'camera_link_optical')
        # Attach the top-down SLAM occupancy map as a 2nd image to the VLM. map_max_px
        # bounds the rendered map's longest side (kept small to limit tokens/latency).
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('send_map', True)
        self.declare_parameter('map_max_px', 384)
        g = lambda n: self.get_parameter(n).value
        self.replan_n = max(1, int(g('replan_every_n')))
        self.turn_step = float(g('turn_step_rad'))
        self.fwd_step = float(g('forward_step_m'))
        self.approach_offset = float(g('approach_offset'))
        self.max_steps = int(g('max_steps'))
        self.map_frame = g('map_frame')
        self.robot_frame = g('robot_frame')
        self.skill_wait_s = float(g('skill_wait_s'))
        self.result_timeout_s = float(g('result_timeout_s'))
        self.min_step_s = float(g('min_step_s'))
        self.detect_timeout_s = float(g('detect_timeout_s'))
        self.detect_conf = float(g('detect_conf'))
        self.vlm_timeout_s = float(g('vlm_timeout_s'))
        self.camera_frame = g('camera_frame')
        self.send_map = bool(g('send_map')) and _HAVE_CV
        self.map_max_px = int(g('map_max_px'))
        self.async_replan = bool(g('async_replan'))
        self._planner_pool = ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix='replan')

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
        self._jpeg = None
        self._map = None                 # latest SLAM OccupancyGrid (for the VLM map)
        # guards the consistency of the (camera jpeg, /target_pixel) snapshot vs the
        # ROS executor threads that write them (_on_image / _on_pixel)
        self._lock = threading.Lock()
        self._bridge = CvBridge() if _HAVE_CV else None
        sub = ReentrantCallbackGroup()
        if self.send_map:
            self.create_subscription(OccupancyGrid, g('map_topic'), self._on_map,
                                     _map_latched_qos(), callback_group=sub)
        # BEST_EFFORT/no-deadline to match the detector/tracker's offered QoS (a
        # RELIABLE sub would receive nothing from a BEST_EFFORT publisher)
        self.create_subscription(PointStamped, '/target_pixel', self._on_pixel,
                                 detection_stream_nodeadline(), callback_group=sub)
        if _HAVE_CV:
            self.create_subscription(Image, '/camera/camera/color/image_raw',
                                     self._on_image, 1, callback_group=sub)
        self.create_subscription(String, '/vlm_mission', self._on_mission, 1,
                                 callback_group=sub)
        self.notes_pub = self.create_publisher(Notes, '/planner/notes', 1)

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

    def _on_pixel(self, msg):
        with self._lock:
            self._pixel = msg

    def _on_image(self, msg):
        try:
            cv = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            ok, buf = cv2.imencode('.jpg', cv)
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()
        except Exception as e:
            self.get_logger().warn('image encode failed: %s' % e, throttle_duration_sec=5.0)

    def _on_mission(self, msg):
        target = (msg.data or '').strip()
        if not target or self._busy:
            return
        self._busy = True
        threading.Thread(target=self._run_mission, args=(target,), daemon=True).start()

    # ---- observation ----
    def _camera_jpeg(self):
        with self._lock:
            return self._jpeg

    def _observation(self, target, step_index):
        """Pull candidates + the matching VLM image together, then build the
        Observation. Returns (obs, pixels, jpeg, map_jpeg) so the plan uses a
        CONSISTENT (candidate ids, camera image) pair even while the camera/replan
        threads run; the top-down SLAM map is rendered alongside (or None)."""
        cands, pixels, jpeg = self._refresh_candidates(target)
        map_jpeg, map_text = self._render_map()
        obs = Observation(target=target, candidates=cands,
                          notes_facts=self.notes.facts, step_index=step_index,
                          map_text=map_text)
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
            g.conf_threshold = self.detect_conf
            res = self._call_action(self._detect, g, self.detect_timeout_s)
            if res is not None and getattr(res, 'candidates', None):
                cands, pix = [], {}
                for c in res.candidates:
                    cands.append(Candidate(mark_id=int(c.mark_id), label=c.label,
                                           score=float(c.confidence),
                                           distance_m=float(c.pixel.z)))  # z = depth_m
                    pix[int(c.mark_id)] = c.pixel    # Point: x=u, y=v, z=depth_m
                jpeg = bytes(res.annotated.data) if res.annotated.data else self._camera_jpeg()
                return cands, pix, jpeg
            # Detector answered but found NOTHING -> report honestly empty. Must NOT fall
            # back to /target_pixel here: during DRIVE_TO_VISIBLE the orchestrator keeps
            # republishing the chosen pixel on /target_pixel, which would otherwise leak
            # back as a PHANTOM stale detection (target still "1.7 m away" after we drove
            # right up to it and YOLOE lost it at close range).
            return [], {}, self._camera_jpeg()
        # Detector server absent -> last-resort single /target_pixel candidate (lets the
        # orchestrator also run against the continuous rgb_tracker instead of the service).
        return self._fallback_candidates(target)

    def _fallback_candidates(self, target):
        with self._lock:
            px = self._pixel
            jpeg = self._jpeg
        if px is not None:
            return ([Candidate(mark_id=1, label=target, score=1.0,
                               distance_m=float(px.point.z))], {1: px.point}, jpeg)
        return [], {}, jpeg

    def _robot_pose(self):
        try:
            tf = self._tf.lookup_transform(self.map_frame, self.robot_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.x, t.y, yaw)

    def _render_map(self):
        """Render the latest SLAM OccupancyGrid to a compact top-down JPEG with the
        robot drawn on it (white=free, black=obstacle, gray=unknown; red dot+line =
        robot pose+heading), plus a text description. North-up, metric. Returns
        (jpeg_bytes | None, description | '')."""
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
        ok, buf = cv2.imencode('.jpg', img)
        if not ok:
            return None, ''
        return buf.tobytes(), orch.describe_occupancy_grid(
            w, h, res, robot_xy, n_free, n_occ, n_unknown)

    # ---- anytime/async mission loop (Phase 4.6): replan overlaps execution ----
    def _compute_plan(self, target, step):
        """Build an observation and ask the planner for up to N atomic actions.
        Runs either inline (bootstrap) or on the planner pool concurrently with
        execution. Returns a _PlanBundle (actions + the candidate pixel snapshot
        the VLM chose from); empty actions on VLM failure (circuit-breaker fed)."""
        # (obs, pixels, jpeg, map_jpeg) captured together -> the plan's DRIVE_TO_VISIBLE
        # pixels and the VLM images are a consistent set (race-free vs camera/replan).
        obs, pixels, jpeg, map_jpeg = self._observation(target, step)
        # Phase 5.1: pick VLM or the latched FLAT fallback (once the breaker opens).
        client = self._degrade.select(self.client, self._fallback, self.cb.is_open)
        if self._degrade.just_degraded():
            self.get_logger().error('circuit-breaker OPEN -> degrade VLM->FLAT (mission '
                                     'continues as FLAT, DEGRADED)')
            self.notes.add_fact('DEGRADED: VLM lost -> continuing in FLAT fallback')
        best = max(obs.candidates, key=lambda c: c.score, default=None)
        det = ('' if best is None
               else " best='%s' conf=%.2f @%.2fm" % (best.label, best.score, best.distance_m))
        self.get_logger().info(
            'observe@step %d: %d detection(s)%s, notes=%d, map=%s -> asking %s'
            % (step, len(obs.candidates), det, len(obs.notes_facts),
               'yes' if map_jpeg else 'no', type(client).__name__))
        try:
            actions = list(client.plan_sequence(obs, jpeg, map_jpeg, n=self.replan_n))
            self.cb.record_success() if actions else self.cb.record_failure()
            self.get_logger().info('plan@step %d: VLM returned %d action(s): %s'
                                   % (step, len(actions),
                                      ', '.join(self._action_brief(a) for a in actions) or '-'))
        except Exception as e:
            self.cb.record_failure()
            self.get_logger().warn('plan failed (%s); cb_open=%s' % (e, self.cb.is_open))
            actions = []
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
        self.notes = NotesBuffer()
        self.cb = CircuitBreaker()
        self._degrade = DegradationLatch()   # fresh mission retries the VLM
        step = 0
        pending = None
        try:
            bundle = self._compute_plan(target, step)   # bootstrap (the only idle point)
            while rclpy.ok() and step < self.max_steps:
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
                    self.get_logger().info('step %d: %s -- %s'
                                           % (step, self._action_brief(action),
                                              action.rationale or ''))
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
                    ok = self._dispatch(action, bundle.pixels)
                    self.notes.add_fact('%s%s -> %s' % (
                        action.name,
                        (' ' + action.rationale) if action.rationale else '',
                        'ok' if ok else 'failed'))
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
            self.get_logger().info('VLM mission ended after %d steps%s' % (
                step, ' (DEGRADED: ran in FLAT fallback)' if self._degrade.degraded else ''))
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

    def _dispatch(self, action, cand_pixels):
        if action.kind in (TURN, DRIVE_FORWARD):
            pose = self._robot_pose()
            if pose is None:
                self.get_logger().warn('no %s->%s TF; skip motion' % (self.map_frame, self.robot_frame))
                return False
            gx, gy, gyaw = orch.relative_goal(pose[0], pose[1], pose[2], action)
            return self._send_goto(gx, gy, gyaw)
        if action.kind == DRIVE_TO_VISIBLE:
            return self._send_approach_mark(action.mark_id, action.arg_label, cand_pixels)
        if action.kind == DETECT_ALL:
            return self._do_detect_all()
        return False

    def _goal_id(self):
        return uuid.uuid4().hex

    def _send_goto(self, x, y, yaw):
        g = GoToPose.Goal()
        g.request_id = self._goal_id()
        g.mission_epoch = self._epoch
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
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
        g.conf_threshold = self.detect_conf
        res = self._call_action(self._detect, g, self.detect_timeout_s)
        cands = getattr(res, 'candidates', None) if res is not None else None
        if not cands:
            self.notes.add_fact('DETECT_ALL: nothing detected in view')
            return False
        seen = ', '.join('%s(%.2f)' % (c.label, c.confidence) for c in cands)
        self.notes.add_fact('objects in view: ' + seen)
        self.get_logger().info('DETECT_ALL: %d object(s): %s' % (len(cands), seen))
        return True

    def _dispatch_stop(self):
        g = Stop.Goal()
        g.request_id = self._goal_id()
        g.mission_epoch = self._epoch
        g.mode = Stop.Goal.SOFT_STOP
        return self._send_and_wait(orch.SKILL_STOP, g)

    def _send_and_wait(self, skill, goal):
        """Send a skill goal and block (in the worker thread) for the result,
        using events set by the executor-thread done-callbacks (loopback-safe)."""
        ac = self._ac[skill]
        if not ac.wait_for_server(timeout_sec=self.skill_wait_s):
            self.get_logger().warn('skill %s server unavailable' % skill)
            return False
        gh_box = {}
        gh_evt = threading.Event()

        def _gh_cb(fut):
            gh_box['gh'] = fut.result()
            gh_evt.set()
        ac.send_goal_async(goal).add_done_callback(_gh_cb)
        if not gh_evt.wait(self.skill_wait_s) or gh_box.get('gh') is None or not gh_box['gh'].accepted:
            self.get_logger().warn('skill %s goal not accepted' % skill)
            return False
        res_box = {}
        res_evt = threading.Event()

        def _res_cb(fut):
            res_box['res'] = fut.result()
            res_evt.set()
        gh_box['gh'].get_result_async().add_done_callback(_res_cb)
        if not res_evt.wait(self.result_timeout_s):
            self.get_logger().warn('skill %s result timeout' % skill)
            return False
        res = res_box.get('res')
        outcome = getattr(getattr(res, 'result', None), 'outcome', None)
        return outcome == 0   # 0 == SUCCEEDED across the skill results

    def _publish_notes(self, target):
        m = Notes()
        m.header.stamp = self.get_clock().now().to_msg()
        m.mission_epoch = self._epoch
        m.summary = self.notes.summary()
        m.facts = self.notes.facts
        m.token_estimate = self.notes.token_estimate()
        self.notes_pub.publish(m)


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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
