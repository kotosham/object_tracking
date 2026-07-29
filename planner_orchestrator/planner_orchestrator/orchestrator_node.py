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
_PlanBundle = namedtuple('_PlanBundle', 'actions pixels obs')

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
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
    Action, Candidate, CircuitBreaker, ContextMark, DegradationLatch,
    NotesBuffer, Observation, DETECT_ALL, DRIVE_FORWARD, DRIVE_TO_VISIBLE,
    TURN, centered_forward_blocker, context_mark_promotable_to_target,
    context_relevance_for, distance_for_options, distance_is_known,
    format_distance, image_side,
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
        self.declare_parameter('vlm_timeout_s', 30.0)
        self.declare_parameter('turn_step_rad', 0.6)
        self.declare_parameter('forward_step_m', 0.5)
        self.declare_parameter('approach_offset', 0.58)
        # Mirrors the Pi-side ApproachDetection default. If the visual target is
        # farther than offset + this step, a successful DRIVE_TO_VISIBLE means an
        # intermediate bounded approach, not final target arrival.
        self.declare_parameter('approach_max_goal_step_m', 1.2)
        # Once a final ApproachDetection reports SUCCEEDED, stop the VLM mission
        # instead of replanning on a close-range frame where depth often becomes
        # unknown and the target may overflow the camera. Long-range bounded
        # approaches keep the mission alive.
        self.declare_parameter('finish_on_approach_success', True)
        # A strict target seen farther than this is not point-blank enough to close
        # the mission solely on Nav2 success; re-observe once to confirm proximity.
        self.declare_parameter('approach_final_observe_start_dist_m', 0.9)
        self.declare_parameter('max_steps', 40)
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
        self.declare_parameter('async_replan', False)
        # Phase 3 binding: pull real Set-of-Mark candidates from the edge detector
        # and feed the chosen mark's pixel to ApproachDetection on DRIVE_TO_VISIBLE.
        self.declare_parameter('detect_action_name', 'detect_target')
        self.declare_parameter('detect_timeout_s', 6.0)
        # Legacy override: if >0, applies one confidence floor to both target
        # detection and DETECT_ALL. Prefer the split thresholds below: they mirror
        # the diploma tracker (DINO strict for target, YOLOE permissive for overview).
        self.declare_parameter('detect_conf', 0.0)
        self.declare_parameter('target_detect_conf', 0.50)
        self.declare_parameter('detect_all_conf', 0.08)
        # DINO occasionally emits one-frame semantic spikes ("chair" on a handle,
        # ball, table edge). Require the target-like object to be seen twice on
        # consecutive detector calls before the VLM may drive to it.
        self.declare_parameter('target_confirm_observations', 2)
        self.declare_parameter('context_target_confirm_observations', 2)
        self.declare_parameter('target_confirm_interval_s', 0.20)
        self.declare_parameter('target_confirm_pixel_tolerance_px', 90.0)
        self.declare_parameter('target_confirm_depth_tolerance_m', 0.80)
        # When the target detector returns no candidates, run a broad-vocab context
        # pass so the VLM can reason "office furniture is on the left" instead of
        # falling straight into a blind scan.
        self.declare_parameter('auto_context_when_target_absent', True)
        self.declare_parameter('context_detect_conf', 0.35)
        self.declare_parameter('context_target_promote_conf', 0.35)
        self.declare_parameter('semantic_turn_antioscillation', True)
        self.declare_parameter('semantic_turn_max_streak', 2)
        self.declare_parameter('semantic_probe_forward_m', 0.45)
        self.declare_parameter(
            'office_context_query',
            'desk | table | drawer cabinet | cabinet | file cabinet | bookshelf | shelf | '
            'monitor | keyboard | laptop | printer')
        self.declare_parameter('camera_image_width', 640)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('subscribe_camera_image', True)
        self.declare_parameter('camera_image_topic', '/camera_edge/color/image_raw')
        self.declare_parameter('camera_use_compressed_input', False)
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
        self.approach_max_goal_step_m = float(g('approach_max_goal_step_m'))
        self.approach_final_observe_start_dist_m = float(
            g('approach_final_observe_start_dist_m'))
        self.finish_on_approach_success = bool(g('finish_on_approach_success'))
        self._last_approach_result = None
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
        self.target_confirm_observations = max(1, int(g('target_confirm_observations')))
        self.context_target_confirm_observations = max(
            1, int(g('context_target_confirm_observations')))
        self.target_confirm_interval_s = max(0.0, float(g('target_confirm_interval_s')))
        self.target_confirm_pixel_tolerance_px = max(
            0.0, float(g('target_confirm_pixel_tolerance_px')))
        self.target_confirm_depth_tolerance_m = max(
            0.0, float(g('target_confirm_depth_tolerance_m')))
        self.auto_context_when_target_absent = bool(g('auto_context_when_target_absent'))
        self.context_detect_conf = float(g('context_detect_conf'))
        self.context_target_promote_conf = float(g('context_target_promote_conf'))
        self.semantic_turn_antioscillation = bool(g('semantic_turn_antioscillation'))
        self.semantic_turn_max_streak = max(1, int(g('semantic_turn_max_streak')))
        self.semantic_probe_forward_m = float(g('semantic_probe_forward_m'))
        self._semantic_turn_side = ''
        self._semantic_turn_streak = 0
        self.office_context_query = str(g('office_context_query'))
        self.camera_image_width = max(1, int(g('camera_image_width')))
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
        if self.subscribe_camera_image:
            image_type = CompressedImage if self.camera_use_compressed_input else Image
            self.create_subscription(image_type, self.camera_image_topic,
                                     self._on_image, media_besteffort(), callback_group=sub)
            self.get_logger().info(
                'planner_orchestrator camera input: topic=%s compressed=%s'
                % (self.camera_image_topic, self.camera_use_compressed_input))
        self.create_subscription(String, '/vlm_mission', self._on_mission, 1,
                                 callback_group=sub)
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
            if self.camera_use_compressed_input:
                arr = np.frombuffer(msg.data, np.uint8)
                cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                cv = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            if cv is None:
                return
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
        with self._lock:
            return self._jpeg

    def _observation(self, target, step_index):
        """Pull candidates + the matching VLM image together, then build the
        Observation. Returns (obs, pixels, jpeg, map_jpeg) so the plan uses a
        CONSISTENT (candidate ids, camera image) pair even while the camera/replan
        threads run; the top-down SLAM map is rendered alongside (or None)."""
        cands, pixels, jpeg = self._refresh_candidates(target)
        context_marks = []
        if not cands and self.auto_context_when_target_absent:
            context_marks, context_jpeg, promoted_cands, promoted_pixels = self._detect_context(target)
            if promoted_cands:
                cands = promoted_cands
                pixels = promoted_pixels
            if context_jpeg:
                jpeg = context_jpeg
        map_jpeg, map_text = self._render_map()
        obs = Observation(target=target, candidates=cands,
                          context_marks=context_marks,
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
            res = self._call_detect_target(target, self.target_detect_conf, True)
            if res is not None and getattr(res, 'candidates', None):
                cands, pix = self._candidates_and_pixels_from_detect_candidates(
                    res.candidates)
                cands, pix = self._confirm_target_candidates(
                    target, cands, pix, self.target_detect_conf,
                    self.target_confirm_observations, source='target')
                jpeg = bytes(res.annotated.data) if res.annotated.data else self._camera_jpeg()
                if not cands:
                    self.get_logger().info(
                        'target_confirm: raw target detection(s) rejected as unconfirmed')
                    self._activity('target_confirm',
                                   target=target, raw=len(getattr(res, 'candidates', []) or []),
                                   confirmed=0)
                    return [], {}, self._camera_jpeg()
                # Dashboard view: what the robot sees + the mark ids offered to the VLM.
                self._publish_view(self._setofmark_pub, jpeg)
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

    def _call_detect_target(self, query, conf_threshold, render_setofmark):
        g = DetectTarget.Goal()
        g.request_id = self._goal_id()
        g.mission_epoch = self._epoch
        g.query = query
        g.render_setofmark = bool(render_setofmark)
        g.conf_threshold = float(conf_threshold)
        return self._call_action(self._detect, g, self.detect_timeout_s)

    def _candidates_and_pixels_from_detect_candidates(self, candidates):
        cands, pix = [], {}
        for c in candidates or []:
            bbox = getattr(c, 'bbox', None)
            if bbox is not None and int(getattr(bbox, 'width', 0) or 0) > 0:
                center_x = float(bbox.x_offset) + float(bbox.width) / 2.0
            else:
                center_x = float(getattr(getattr(c, 'pixel', None), 'x',
                                         self.camera_image_width / 2.0))
            center_x_norm = max(0.0, min(1.0, center_x / float(self.camera_image_width)))
            pixel = getattr(c, 'pixel', None)
            distance_m = float(getattr(pixel, 'z', 0.0) or 0.0)
            mark_id = int(getattr(c, 'mark_id', 0) or 0)
            cands.append(Candidate(mark_id=mark_id,
                                   label=str(getattr(c, 'label', '') or ''),
                                   score=float(getattr(c, 'confidence', 0.0) or 0.0),
                                   distance_m=distance_m,
                                   side=image_side(center_x_norm),
                                   center_x_norm=center_x_norm))
            if pixel is not None:
                pix[mark_id] = pixel    # Point: x=u, y=v, z=depth_m
        return cands, pix

    @staticmethod
    def _label_compatible(target, a, b):
        """Loose label check for two detector outputs describing the same target."""
        t = (target or '').strip().lower()
        la = (a or '').strip().lower()
        lb = (b or '').strip().lower()
        if not la or not lb:
            return False
        if la in lb or lb in la:
            return True
        if t and ((t in la and t in lb)
                  or (set(t.split()) & set(la.split()) & set(lb.split()))):
            return True
        return bool(set(la.split()) & set(lb.split()))

    def _candidate_confirmed_by(self, target, cand, pixel, other_cands, other_pixels):
        """Match a candidate against a second detector pass by label + image/depth."""
        for other in other_cands or []:
            if not self._label_compatible(target, cand.label, other.label):
                continue
            other_pixel = (other_pixels or {}).get(int(other.mark_id))
            if pixel is not None and other_pixel is not None:
                dx = abs(float(pixel.x) - float(other_pixel.x))
                dy = abs(float(pixel.y) - float(other_pixel.y))
                if dx > self.target_confirm_pixel_tolerance_px:
                    continue
                if dy > self.target_confirm_pixel_tolerance_px:
                    continue
            else:
                tol_norm = self.target_confirm_pixel_tolerance_px / float(
                    max(1, self.camera_image_width))
                if abs(float(cand.center_x_norm) - float(other.center_x_norm)) > tol_norm:
                    continue
            if (distance_is_known(cand.distance_m)
                    and distance_is_known(other.distance_m)):
                dd = abs(float(cand.distance_m) - float(other.distance_m))
                allowed = max(self.target_confirm_depth_tolerance_m,
                              0.35 * min(float(cand.distance_m),
                                         float(other.distance_m)))
                if dd > allowed:
                    continue
            return True
        return False

    def _confirm_target_candidates(self, target, cands, pix, conf_threshold,
                                   required_observations, source, query=None):
        if required_observations <= 1 or not cands:
            return cands, pix
        if self.target_confirm_interval_s > 0.0:
            time.sleep(self.target_confirm_interval_s)
        res2 = self._call_detect_target(query or target, conf_threshold, False)
        if res2 is None or not getattr(res2, 'candidates', None):
            self.get_logger().info(
                'target_confirm[%s]: %d raw -> 0 confirmed (second pass empty)'
                % (source, len(cands)))
            return [], {}
        cands2, pix2 = self._candidates_and_pixels_from_detect_candidates(res2.candidates)
        confirmed, confirmed_pix = [], {}
        for cand in cands:
            pixel = (pix or {}).get(int(cand.mark_id))
            if self._candidate_confirmed_by(target, cand, pixel, cands2, pix2):
                confirmed.append(cand)
                if pixel is not None:
                    confirmed_pix[int(cand.mark_id)] = pixel
        self.get_logger().info(
            'target_confirm[%s]: %d raw -> %d confirmed over %d observation(s)'
            % (source, len(cands), len(confirmed), required_observations))
        self._activity('target_confirm', target=target, source=source,
                       raw=len(cands), confirmed=len(confirmed),
                       required_observations=required_observations)
        return confirmed, confirmed_pix

    def _fallback_candidates(self, target):
        with self._lock:
            px = self._pixel
            jpeg = self._jpeg
        if px is not None:
            return ([Candidate(mark_id=1, label=target, score=1.0,
                               distance_m=float(px.point.z),
                               source='fallback')], {1: px.point}, jpeg)
        return [], {}, jpeg

    def _context_marks_from_candidates(self, target, candidates):
        """Convert broad-vocab detector results into scene-context marks for the
        VLM. These marks are not valid DRIVE_TO_VISIBLE targets; they only explain
        what kind of area is visible and on which side of the camera frame."""
        out = []
        for c in candidates or []:
            bbox = getattr(c, 'bbox', None)
            if bbox is not None and int(getattr(bbox, 'width', 0) or 0) > 0:
                center_x = float(bbox.x_offset) + float(bbox.width) / 2.0
            else:
                center_x = float(getattr(getattr(c, 'pixel', None), 'x', self.camera_image_width / 2.0))
            center_x_norm = max(0.0, min(1.0, center_x / float(self.camera_image_width)))
            distance_m = float(getattr(getattr(c, 'pixel', None), 'z', 0.0) or 0.0)
            label = str(getattr(c, 'label', '') or '')
            out.append(ContextMark(
                mark_id=int(getattr(c, 'mark_id', 0) or 0),
                label=label,
                score=float(getattr(c, 'confidence', 0.0) or 0.0),
                distance_m=distance_m,
                side=image_side(center_x_norm),
                center_x_norm=center_x_norm,
                relevance=context_relevance_for(target, label),
            ))
        return out

    def _context_brief(self, mark):
        return '%d:%s(%.2f@%s,%s,%s)' % (
            mark.mark_id, mark.label, mark.score,
            format_distance(mark.distance_m), mark.side, mark.relevance)

    @staticmethod
    def _use_office_context(target):
        t = (target or '').strip().lower()
        return any(term in t for term in (
            'office', 'chair', 'desk', 'table', 'cabinet', 'drawer', 'shelf',
            'bookcase', 'monitor', 'keyboard', 'laptop', 'printer'))

    def _context_query_for_target(self, target):
        if self._use_office_context(target):
            return (self.office_context_query or '').strip()
        return ''

    def _detect_context(self, target):
        """Automatic context pass used only when the final target is absent. It
        gives the VLM semantic search cues such as 'desk on the left' without
        pretending those context objects are final approach targets."""
        if not self._detect.wait_for_server(timeout_sec=1.0):
            return [], None, [], {}
        query = self._context_query_for_target(target)
        conf_threshold = self.context_detect_conf if query else self.detect_all_conf
        res = self._call_detect_target(query, conf_threshold, True)
        cands = getattr(res, 'candidates', None) if res is not None else None
        jpeg = None
        if getattr(res, 'annotated', None) is not None and res.annotated.data:
            jpeg = bytes(res.annotated.data)
            self._publish_view(self._setofmark_pub, jpeg)
        marks = self._context_marks_from_candidates(target, cands)
        det_cands, det_pixels = self._candidates_and_pixels_from_detect_candidates(cands)
        target_like_ids = {
            int(m.mark_id) for m in marks
            if (m.relevance or '').lower() == 'target_like'
        }
        confirmed_target_like_ids = set(target_like_ids)
        if target_like_ids and self.context_target_confirm_observations > 1:
            target_like_cands = [
                c for c in det_cands if int(c.mark_id) in target_like_ids
            ]
            confirmed_like, _ = self._confirm_target_candidates(
                target, target_like_cands, det_pixels, conf_threshold,
                self.context_target_confirm_observations,
                source='context_target_like', query=query)
            confirmed_target_like_ids = {int(c.mark_id) for c in confirmed_like}
            hidden = len(target_like_ids - confirmed_target_like_ids)
            if hidden:
                self.get_logger().info(
                    'context_confirm: hiding %d unconfirmed target-like mark(s)'
                    % hidden)
        if confirmed_target_like_ids != target_like_ids:
            marks = [
                m for m in marks
                if ((m.relevance or '').lower() != 'target_like'
                    or int(m.mark_id) in confirmed_target_like_ids)
            ]
        promotable_ids = {
            int(m.mark_id) for m in marks
            if context_mark_promotable_to_target(
                target, m, self.context_target_promote_conf)
        }
        promoted = [
            Candidate(mark_id=c.mark_id, label=c.label, score=c.score,
                      distance_m=c.distance_m, side=c.side,
                      center_x_norm=c.center_x_norm,
                      source='context_promoted')
            for c in det_cands if int(c.mark_id) in promotable_ids
        ]
        promoted_pixels = {
            int(mark_id): point for mark_id, point in det_pixels.items()
            if int(mark_id) in promotable_ids
        }
        if marks:
            backend = 'dino_office' if query else 'yoloe_all'
            self.get_logger().info('context_detect[%s]: %d object(s): %s' % (
                backend, len(marks), ', '.join(self._context_brief(m) for m in marks)))
        if promoted:
            self.get_logger().info(
                'context_promote: %d target-like context object(s) promoted to candidates: %s'
                % (len(promoted), ', '.join(
                    '%d:%s(%.2f@%s,%s)' % (
                        c.mark_id, c.label, c.score,
                        format_distance(c.distance_m), c.side)
                    for c in promoted)))
        self._activity(
            'context_detect',
            backend='dino_office' if query else 'yoloe_all',
            objects=[{'mark_id': m.mark_id, 'label': m.label,
                      'score': round(float(m.score), 2),
                      'distance_m': distance_for_options(m.distance_m),
                      'side': m.side, 'relevance': m.relevance}
                     for m in marks],
            promoted=[{'mark_id': c.mark_id, 'label': c.label,
                       'score': round(float(c.score), 2),
                       'distance_m': distance_for_options(c.distance_m),
                       'side': c.side}
                      for c in promoted])
        return marks, jpeg, promoted, promoted_pixels

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
        best_context = max(obs.context_marks, key=lambda c: c.score, default=None)
        det = ('' if best is None else " best='%s' conf=%.2f @%s"
               % (best.label, best.score, format_distance(best.distance_m)))
        ctx = ('' if best_context is None else " context=%d best_context='%s' %s %.2f @%s"
               % (len(obs.context_marks), best_context.label, best_context.side,
                  best_context.score, format_distance(best_context.distance_m)))
        self.get_logger().info(
            'observe@step %d: %d target detection(s)%s%s, notes=%d, map=%s -> asking %s'
            % (step, len(obs.candidates), det, ctx, len(obs.notes_facts),
               'yes' if map_jpeg else 'no', type(client).__name__))
        self._activity(
            'observe', step=step, n_detections=len(obs.candidates),
            detections=[{'mark_id': c.mark_id, 'label': c.label,
                         'score': round(float(c.score), 3),
                         'distance_m': distance_for_options(c.distance_m)}
                        for c in obs.candidates],
            context_marks=[{'mark_id': c.mark_id, 'label': c.label,
                            'score': round(float(c.score), 3),
                            'distance_m': distance_for_options(c.distance_m),
                            'side': c.side, 'relevance': c.relevance}
                           for c in obs.context_marks],
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
                          'role': self._action_role(a, obs),
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
        return _PlanBundle(actions, pixels, obs)

    def _next_bundle(self, pending, target, step):
        """Adopt the concurrently-computed plan at the commit-point (no idle if it
        finished during execution), or compute inline when async is off."""
        if pending is not None:
            try:
                return pending.result()
            except Exception:
                return _PlanBundle([], {}, None)
        return self._compute_plan(target, step)

    def _run_mission(self, target):
        self.get_logger().info('VLM mission start: target="%s"' % target)
        self._activity('mission_start', target=target,
                       client=type(self.client).__name__, creds=self._cred_src,
                       replan_every_n=self.replan_n, max_steps=self.max_steps)
        self.notes = NotesBuffer()
        self.cb = CircuitBreaker()
        self._degrade = DegradationLatch()   # fresh mission retries the VLM
        self._semantic_turn_side = ''
        self._semantic_turn_streak = 0
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
                    role = self._action_role(action, bundle.obs)
                    action = self._semantic_explore_antioscillation(
                        action, bundle.obs, role)
                    role = self._action_role(action, bundle.obs)
                    self.get_logger().info('step %d [%s]: %s -- %s'
                                           % (step, role, self._action_brief(action),
                                              action.rationale or ''))
                    self._activity('step_start', step=step,
                                   role=role,
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
                    ok = self._dispatch(action, bundle.pixels, target)
                    self.notes.add_fact('%s%s -> %s' % (
                        action.name,
                        (' ' + action.rationale) if action.rationale else '',
                        'ok' if ok else 'failed'))
                    self._activity('step_result', step=step,
                                   action=self._action_brief(action),
                                   result='ok' if ok else 'failed',
                                   duration_s=round(time.monotonic() - t0, 2))
                    self._remember_semantic_motion(action, bundle.obs, role, ok)
                    self._publish_notes(target)
                    step += 1
                    if ok and action.kind == DRIVE_TO_VISIBLE and self._approach_can_auto_finish(
                            action, bundle.obs):
                        self.get_logger().info(
                            'target approach succeeded -> finishing mission at last confirmed '
                            'approach pose')
                        self._activity(
                            'auto_done', step=step, target=target,
                            reason='DRIVE_TO_VISIBLE succeeded; final approach pose reached')
                        terminate = True
                        break
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
            self._activity('mission_end', target=target, steps=step,
                           degraded=bool(self._degrade.degraded))
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

    @staticmethod
    def _action_role(action, obs):
        """Human-readable intent class for logs/dashboard."""
        if action.kind == DRIVE_TO_VISIBLE:
            return 'target_approach'
        if action.kind in (TURN, DRIVE_FORWARD):
            rationale = (action.rationale or '').lower()
            useful_context = bool(
                obs and any((m.relevance or '').lower() in
                            ('target_like', 'office_context', 'ambiguous')
                            for m in obs.context_marks))
            if 'semantic_explore' in rationale or useful_context:
                return 'semantic_explore'
            return 'blind_scan'
        if action.kind == DETECT_ALL:
            return 'blind_scan'
        if orch.is_terminal(action.kind):
            return 'done'
        return 'other'

    @staticmethod
    def _turn_side(action):
        if action.kind != TURN or abs(float(action.turn_yaw_rad)) < 0.2:
            return ''
        return 'left' if float(action.turn_yaw_rad) > 0.0 else 'right'

    def _semantic_explore_antioscillation(self, action, obs, role):
        if not self.semantic_turn_antioscillation:
            return action
        if role != 'semantic_explore' or obs is None:
            return action
        if obs.candidates:
            return action
        if action.kind != TURN:
            return action
        side = self._turn_side(action)
        if not side:
            return action

        blocker = centered_forward_blocker(obs)
        last_side = self._semantic_turn_side
        reverse_turn = bool(last_side and side != last_side)
        too_many_turns = (
            last_side == side and self._semantic_turn_streak >= self.semantic_turn_max_streak)
        if not reverse_turn and not too_many_turns:
            return action

        reason = []
        if reverse_turn:
            reason.append('blocked reverse semantic turn %s->%s' % (last_side, side))
        if too_many_turns:
            reason.append('semantic turn streak %d on %s' % (
                self._semantic_turn_streak, side))
        base = ('semantic_explore: anti_oscillation: %s'
                % '; '.join(reason))
        if action.rationale:
            base += '; original rationale: ' + action.rationale

        if blocker is None:
            return Action(DRIVE_FORWARD,
                          forward_dist_m=max(0.05, self.semantic_probe_forward_m),
                          arg_label=action.arg_label,
                          rationale=base + '; probe forward after inspecting context')

        keep_side = last_side or side
        yaw = self.turn_step * (1.0 if keep_side == 'left' else -1.0) * 0.5
        return Action(TURN, turn_yaw_rad=yaw, arg_label=action.arg_label,
                      rationale=(base + '; forward probe blocked by centered "%s" at %.2fm, '
                                 'keep inspecting %s instead of oscillating'
                                 % (blocker.label, float(blocker.distance_m), keep_side)))

    def _remember_semantic_motion(self, action, obs, role, ok):
        if role != 'semantic_explore' or not ok:
            if action.kind == DRIVE_TO_VISIBLE:
                self._semantic_turn_side = ''
                self._semantic_turn_streak = 0
            return
        if action.kind == TURN:
            side = self._turn_side(action)
            if not side:
                return
            if side == self._semantic_turn_side:
                self._semantic_turn_streak += 1
            else:
                self._semantic_turn_side = side
                self._semantic_turn_streak = 1
            return
        if action.kind == DRIVE_FORWARD:
            self._semantic_turn_side = ''
            self._semantic_turn_streak = 0

    def _approach_can_auto_finish(self, action, obs):
        """Only final, not bounded, ApproachDetection success should end a mission.

        The Pi clamps far visual targets to short Nav2 segments so online SLAM and
        costmaps can grow. Reaching that segment is progress, not arrival.
        """
        if not self.finish_on_approach_success:
            return False
        res = self._last_approach_result
        bounded_step = bool(getattr(res, 'bounded_step', False))
        final_distance = getattr(res, 'final_distance_m', float('nan'))
        cand = None
        if obs is not None:
            cand = next((c for c in obs.candidates
                         if int(c.mark_id) == int(action.mark_id)), None)
        if cand is not None and (cand.source or '') == 'context_promoted':
            self.get_logger().info(
                'context-promoted target approach succeeded -> continuing mission '
                'until strict target detector confirms the goal')
            self._activity(
                'step_progress', action=self._action_brief(action),
                result='context_promoted_probe',
                candidate_label=cand.label,
                candidate_score=round(float(cand.score), 2),
                final_distance_m=(round(float(final_distance), 2)
                                  if distance_is_known(final_distance) else None))
            return False
        if bounded_step:
            final_threshold = self.approach_offset + 0.35
            if distance_is_known(final_distance):
                self.get_logger().info(
                    'bounded target approach succeeded with %.2fm still expected '
                    'to target (final threshold %.2fm) -> continuing mission'
                    % (float(final_distance), final_threshold))
                self._activity(
                    'step_progress', action=self._action_brief(action),
                    result='bounded_approach',
                    final_distance_m=round(float(final_distance), 2),
                    final_threshold_m=round(final_threshold, 2))
            else:
                self.get_logger().info(
                    'bounded target approach succeeded (remaining distance unknown) '
                    '-> continuing mission')
                self._activity(
                    'step_progress', action=self._action_brief(action),
                    result='bounded_approach',
                    final_distance_m=None)
            return False
        if distance_is_known(final_distance):
            final_threshold = self.approach_offset + 0.35
            if float(final_distance) <= final_threshold:
                return True
            self.get_logger().info(
                'intermediate target approach succeeded with %.2fm still expected '
                'to target (final threshold %.2fm) -> continuing mission'
                % (float(final_distance), final_threshold))
            self._activity(
                'step_progress', action=self._action_brief(action),
                result='intermediate_approach',
                final_distance_m=round(float(final_distance), 2),
                final_threshold_m=round(final_threshold, 2))
            return False
        if obs is None:
            return True
        if cand is None or not distance_is_known(cand.distance_m):
            return True
        confirm_threshold = max(0.0, self.approach_final_observe_start_dist_m)
        if confirm_threshold > 0.0 and float(cand.distance_m) > confirm_threshold:
            self.get_logger().info(
                'strict target approach succeeded from %.2fm start distance; '
                're-observing before DONE (confirm threshold %.2fm)'
                % (float(cand.distance_m), confirm_threshold))
            self._activity(
                'step_progress', action=self._action_brief(action),
                result='strict_reobserve_before_done',
                start_distance_m=round(float(cand.distance_m), 2),
                confirm_threshold_m=round(confirm_threshold, 2))
            return False
        auto_finish_threshold = (
            self.approach_offset + max(0.0, self.approach_max_goal_step_m) + 0.05)
        if float(cand.distance_m) <= auto_finish_threshold:
            return True
        self.get_logger().info(
            'intermediate target approach succeeded at start distance %.2fm '
            '(auto-finish threshold %.2fm) -> continuing mission'
            % (float(cand.distance_m), auto_finish_threshold))
        self._activity(
            'step_progress', action=self._action_brief(action),
            result='intermediate_approach',
            start_distance_m=round(float(cand.distance_m), 2),
            auto_finish_threshold_m=round(auto_finish_threshold, 2))
        return False

    def _dispatch(self, action, cand_pixels, target=''):
        if action.kind in (TURN, DRIVE_FORWARD):
            pose = self._motion_pose()
            if pose is None:
                self.get_logger().warn(
                    'no TF for relative motion (%s->%s or %s->%s); skip motion'
                    % (self.map_frame, self.robot_frame,
                       self.motion_fallback_frame, self.robot_frame))
                return False
            gx, gy, gyaw = orch.relative_goal(pose[0], pose[1], pose[2], action)
            return self._send_goto(gx, gy, gyaw, frame_id=pose[3])
        if action.kind == DRIVE_TO_VISIBLE:
            return self._send_approach_mark(action.mark_id, action.arg_label, cand_pixels)
        if action.kind == DETECT_ALL:
            return self._do_detect_all(target)
        return False

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
        self._last_approach_result = self._send_and_wait_result(orch.SKILL_APPROACH, g)
        return getattr(self._last_approach_result, 'outcome', None) == 0

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
        if not distance_is_known(pt.z):
            self.get_logger().warn(
                'DRIVE_TO_VISIBLE: mark %s has unknown depth; refusing ApproachDetection'
                % mark_id)
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

    def _do_detect_all(self, target=''):
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
        marks = self._context_marks_from_candidates(target, cands)
        seen = ', '.join('%s(%.2f,%s,%s)' % (
            m.label, m.score, m.side, m.relevance) for m in marks)
        self.notes.add_fact('objects in view: ' + seen)
        self.get_logger().info('DETECT_ALL: %d object(s): %s' % (len(cands), seen))
        self._activity('detect_all',
                       objects=[{'mark_id': m.mark_id, 'label': m.label,
                                 'score': round(float(m.score), 2),
                                 'distance_m': distance_for_options(m.distance_m),
                                 'side': m.side, 'relevance': m.relevance}
                                for m in marks])
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
        res = self._send_and_wait_result(skill, goal)
        outcome = getattr(res, 'outcome', None)
        return outcome == 0   # 0 == SUCCEEDED across the skill results

    def _send_and_wait_result(self, skill, goal):
        """Like _send_and_wait(), but returns the action result object."""
        ac = self._ac[skill]
        if not ac.wait_for_server(timeout_sec=self.skill_wait_s):
            self.get_logger().warn('skill %s server unavailable' % skill)
            return None
        gh_box = {}
        gh_evt = threading.Event()

        def _gh_cb(fut):
            gh_box['gh'] = fut.result()
            gh_evt.set()
        ac.send_goal_async(goal).add_done_callback(_gh_cb)
        if not gh_evt.wait(self.skill_wait_s) or gh_box.get('gh') is None or not gh_box['gh'].accepted:
            self.get_logger().warn('skill %s goal not accepted' % skill)
            return None
        res_box = {}
        res_evt = threading.Event()

        def _res_cb(fut):
            res_box['res'] = fut.result()
            res_evt.set()
        gh_box['gh'].get_result_async().add_done_callback(_res_cb)
        if not res_evt.wait(self.result_timeout_s):
            self.get_logger().warn('skill %s result timeout' % skill)
            return None
        res = res_box.get('res')
        return getattr(res, 'result', None)

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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
