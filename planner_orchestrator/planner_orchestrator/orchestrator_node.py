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
    TURN, context_relevance_for, distance_for_options, distance_is_known,
    format_distance, image_side, lost_target_lock_recovery_action,
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
        self.declare_parameter('min_effective_turn_rad', 0.6)
        self.declare_parameter('initial_scan_when_target_absent', True)
        self.declare_parameter('initial_scan_left_rad', 3.14)
        self.declare_parameter('initial_scan_right_rad', 1.57)
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
        # After a TURN action, wait before the next detector/VLM observation so
        # the RealSense image is not captured while the robot is still settling.
        self.declare_parameter('turn_settle_s', 2.0)
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
        self.declare_parameter('target_detect_conf', 0.60)
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
        self.declare_parameter('context_detect_conf', 0.30)
        # Legacy no-op kept so old runbook commands with this parameter still start.
        # Context detections are no longer promoted to visible target candidates.
        self.declare_parameter('context_target_promote_conf', 0.35)
        self.declare_parameter('semantic_turn_antioscillation', True)
        self.declare_parameter('semantic_turn_max_streak', 1)
        self.declare_parameter('semantic_probe_forward_m', 0.45)
        # Keep a short-lived memory of a strict target after bounded approaches.
        # If it drops out for a frame, recover toward the last confirmed side
        # before falling back to generic context search.
        self.declare_parameter('target_lock_recovery_steps', 3)
        self.declare_parameter('target_lock_recovery_turn_rad', 0.6)
        self.declare_parameter('target_lock_recovery_forward_m', 0.45)
        # After a strict target was confidently approached once, keep a navigation
        # lock on its map point. The detector may lose the object at close range
        # because it overflows the camera; that should not drop us back to context
        # search while a saved target coordinate is still actionable.
        self.declare_parameter('locked_target_approach_max_attempts', 8)
        # If the target is confidently localized but ApproachDetection cannot find
        # a safe final or bounded approach pose yet, do not keep retrying the same
        # blocked target. Move a little to expand the map/reposition, then retry.
        self.declare_parameter('target_approach_blocked_recovery_steps', 2)
        self.declare_parameter('target_approach_blocked_forward_m', 0.55)
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
        self.min_effective_turn_rad = max(0.0, float(g('min_effective_turn_rad')))
        self.initial_scan_when_target_absent = bool(g('initial_scan_when_target_absent'))
        self.initial_scan_left_rad = abs(float(g('initial_scan_left_rad')))
        self.initial_scan_right_rad = abs(float(g('initial_scan_right_rad')))
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
        self.turn_settle_s = max(0.0, float(g('turn_settle_s')))
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
        self.context_target_promote_conf = float(g('context_target_promote_conf'))  # legacy no-op
        self.semantic_turn_antioscillation = bool(g('semantic_turn_antioscillation'))
        self.semantic_turn_max_streak = max(1, int(g('semantic_turn_max_streak')))
        self.semantic_probe_forward_m = float(g('semantic_probe_forward_m'))
        self._semantic_turn_side = ''
        self._semantic_turn_streak = 0
        self.target_lock_recovery_steps = max(0, int(g('target_lock_recovery_steps')))
        self.target_lock_recovery_turn_rad = float(g('target_lock_recovery_turn_rad'))
        self.target_lock_recovery_forward_m = float(g('target_lock_recovery_forward_m'))
        self._target_lock = None
        self.locked_target_approach_max_attempts = max(
            0, int(g('locked_target_approach_max_attempts')))
        self._target_nav_lock = None
        self.target_approach_blocked_recovery_steps = max(
            0, int(g('target_approach_blocked_recovery_steps')))
        self.target_approach_blocked_forward_m = max(
            0.05, float(g('target_approach_blocked_forward_m')))
        self._target_approach_blocked = None
        self._corridor_scan = {}
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
        if cands:
            self._remember_target_lock(target, cands, step_index)
        context_marks = []
        skip_context_for_nav_lock = self._active_target_nav_lock(target) is not None
        if not cands and self.auto_context_when_target_absent and not skip_context_for_nav_lock:
            context_marks, context_jpeg = self._detect_context(target)
            if context_jpeg:
                jpeg = context_jpeg
        map_jpeg, map_text = self._render_map()
        self._record_corridor_scan(target, step_index, cands, context_marks)
        notes = self.notes.facts
        lock_note = self._target_lock_note(target, step_index, have_target=bool(cands))
        if lock_note:
            notes = notes + [lock_note]
        obs = Observation(target=target, candidates=cands,
                          context_marks=context_marks,
                          notes_facts=notes, step_index=step_index,
                          map_text=map_text,
                          corridor_scan=self._corridor_scan_options())
        return obs, pixels, jpeg, map_jpeg

    def _corridor_scan_view(self, step_index):
        """Which initial-scan viewpoint this observation represents.

        step 0 is the starting forward view. After the first initial-scan TURN,
        step 1 observes the right-side corridor. The step-1 plan contains two
        signed left turns, so the next observation is step 3: the left-side view.
        """
        if not self.initial_scan_when_target_absent:
            return ''
        step = int(step_index)
        if step == 0:
            return 'forward'
        if step == 1:
            return 'right'
        if step >= 3 and 'left' not in self._corridor_scan:
            return 'left'
        return ''

    @staticmethod
    def _corridor_object_dict(mark):
        return {
            'label': mark.label,
            'score': round(float(mark.score), 2),
            'distance_m': distance_for_options(mark.distance_m),
            'side': mark.side,
            'relevance': mark.relevance,
        }

    def _record_corridor_scan(self, target, step_index, cands, context_marks):
        view = self._corridor_scan_view(step_index)
        if not view:
            return
        strict_target = any((c.source or '') == 'target' for c in cands or [])
        if strict_target:
            return
        useful = [
            m for m in context_marks or []
            if (m.relevance or '').lower() in ('target_like', 'office_context', 'ambiguous')
        ]
        useful = sorted(
            useful,
            key=lambda m: (
                2 if (m.relevance or '').lower() == 'target_like'
                else 1 if (m.relevance or '').lower() == 'office_context'
                else 0,
                float(m.score),
            ),
            reverse=True)[:6]
        objects = [self._corridor_object_dict(m) for m in useful]
        if objects:
            object_text = ', '.join(
                '%s(%.2f@%s,%s,%s)' % (
                    o['label'], o['score'],
                    'unknown' if o['distance_m'] is None else '%.2fm' % o['distance_m'],
                    o['side'], o['relevance'])
                for o in objects)
        else:
            object_text = 'no useful context objects'
        entry = {
            'view': view,
            'step': int(step_index),
            'objects': objects,
            'summary': (
                '%s corridor/view: %s. Use these objects only as semantic cues '
                'for choosing a free corridor on the SLAM map; do not approach '
                'the context objects themselves.' % (view, object_text)
            ),
        }
        self._corridor_scan[view] = entry
        fact = 'CORRIDOR_SCAN[%s]: %s' % (view, object_text)
        self.notes.add_fact(fact)
        self.get_logger().info(fact)
        self._activity('corridor_scan', step=step_index, view=view,
                       objects=objects, target=target)

    def _corridor_scan_options(self):
        return [
            self._corridor_scan[v]
            for v in ('forward', 'right', 'left')
            if v in self._corridor_scan
        ]

    def _initial_scan_actions(self, obs, step_index):
        """Start a search mission with a short panoramic scan when no strict
        target is visible yet. The forward view has already been checked by the
        current observation. If it is empty, look right first, then sweep left in
        two signed 90-degree turns. Splitting the 180-degree sweep avoids the Nav2
        ambiguity where the controller can choose either physical direction for a
        pi-radian yaw change. Context hints do not cancel the scan: they are useful
        for later corridor choice, but too noisy to skip the initial map-building
        sweep.
        """
        if not self.initial_scan_when_target_absent or obs is None:
            return []
        strict_target = any((c.source or '') == 'target' for c in obs.candidates)
        if strict_target:
            return []
        step = int(step_index)
        if step == 0:
            return [Action(
                TURN, turn_yaw_rad=-self.initial_scan_right_rad,
                rationale=('initial_scan: strict target is not visible in the forward '
                           'view; rotate right ~90deg to check the right-side corridor '
                           'before choosing an exploration direction'))]
        if step == 1:
            left_step = max(0.0, self.initial_scan_left_rad / 2.0)
            if left_step <= 0.0:
                return []
            return [
                Action(
                    TURN, turn_yaw_rad=left_step,
                    rationale=('initial_scan: strict target is still not visible after '
                               'checking right; rotate left ~90deg back through the '
                               'starting heading')),
                Action(
                    TURN, turn_yaw_rad=left_step,
                    rationale=('initial_scan: continue left another ~90deg to inspect '
                               'the left-side corridor before active corridor '
                               'exploration')),
            ]
        return []

    def _initial_scan_action(self, obs, step_index):
        """Compatibility helper for tests/older callers that expect one action."""
        actions = self._initial_scan_actions(obs, step_index)
        return actions[0] if actions else None

    def _remember_target_lock(self, target, cands, step_index):
        strict = [
            c for c in cands or []
            if (c.source or '') == 'target' and distance_is_known(c.distance_m)
        ]
        if not strict:
            return
        best = max(strict, key=lambda c: (float(c.score), -float(c.distance_m)))
        self._target_lock = {
            'target': target,
            'label': best.label,
            'score': float(best.score),
            'distance_m': float(best.distance_m),
            'side': best.side,
            'step': int(step_index),
            'recoveries': 0,
        }
        self.get_logger().info(
            'target_lock: remembered "%s" conf=%.2f @%s on %s at step %d'
            % (best.label, best.score, format_distance(best.distance_m),
               best.side, step_index))

    def _active_target_lock(self, target, step_index):
        lock = self._target_lock
        if not lock or lock.get('target') != target:
            return None
        age_steps = max(0, int(step_index) - int(lock.get('step', 0)))
        recoveries = int(lock.get('recoveries', 0))
        if age_steps <= 0:
            return None
        if self.target_lock_recovery_steps <= 0:
            return None
        if age_steps > self.target_lock_recovery_steps:
            return None
        if recoveries >= self.target_lock_recovery_steps:
            return None
        return lock

    def _target_lock_note(self, target, step_index, have_target=False):
        if have_target:
            return ''
        lock = self._active_target_lock(target, step_index)
        if lock is None:
            return ''
        age_steps = max(0, int(step_index) - int(lock.get('step', 0)))
        return ('TARGET_LOCK: strict target "%s" was last confirmed %.2fm on %s '
                '%d step(s) ago; try to reacquire it before generic context search'
                % (lock.get('label', target), float(lock.get('distance_m', 0.0)),
                   lock.get('side', 'center'), age_steps))

    def _apply_target_lock_recovery(self, actions, obs, target, step_index):
        if obs is None or obs.candidates:
            return actions
        lock = self._active_target_lock(target, step_index)
        if lock is None:
            return actions
        age_steps = max(0, int(step_index) - int(lock.get('step', 0)))
        action = lost_target_lock_recovery_action(
            obs,
            label=str(lock.get('label', target) or target),
            distance_m=float(lock.get('distance_m', 0.0) or 0.0),
            side=str(lock.get('side', 'center') or 'center'),
            age_steps=age_steps,
            turn_step_rad=self.target_lock_recovery_turn_rad,
            forward_dist_m=self.target_lock_recovery_forward_m)
        if action is None:
            return actions
        proposed = ', '.join(self._action_brief(a) for a in actions) or '-'
        lock['recoveries'] = int(lock.get('recoveries', 0)) + 1
        self.get_logger().info(
            'target_lock: recovering last confirmed target; overriding plan %s -> %s '
            '(recovery %d/%d)'
            % (proposed, self._action_brief(action), lock['recoveries'],
               self.target_lock_recovery_steps))
        self._activity(
            'target_lock_recovery', step=step_index,
            action=self._action_brief(action),
            label=lock.get('label', target),
            distance_m=round(float(lock.get('distance_m', 0.0) or 0.0), 2),
            side=lock.get('side', 'center'),
            recovery=lock['recoveries'],
            max_recovery=self.target_lock_recovery_steps)
        return [action]

    @staticmethod
    def _pose_is_valid(ps):
        if ps is None:
            return False
        try:
            return bool(ps.header.frame_id) and math.isfinite(ps.pose.position.x) \
                and math.isfinite(ps.pose.position.y)
        except AttributeError:
            return False

    @staticmethod
    def _point_is_valid(pt):
        if pt is None:
            return False
        try:
            return bool(pt.header.frame_id) and math.isfinite(pt.point.x) \
                and math.isfinite(pt.point.y)
        except AttributeError:
            return False

    def _remember_target_nav_lock(self, target, label, result, step_index,
                                  allow_blocked=False):
        """Latch the map target from a successful strict visual approach.

        This is the missing contract between perception and navigation: after one
        confident object detection, losing the object from the camera should not
        erase the saved map coordinate. Later steps may update this lock from a
        fresh strict detection, or continue toward the locked point directly.
        """
        outcome = getattr(result, 'outcome', None)
        blocked = outcome == ApproachDetection.Result.ABORTED
        if result is None or (outcome != ApproachDetection.Result.SUCCEEDED
                              and not (allow_blocked and blocked)):
            return
        target_point = getattr(result, 'target_point', None)
        final_goal_pose = getattr(result, 'final_goal_pose', None)
        if not self._point_is_valid(target_point) or not self._pose_is_valid(final_goal_pose):
            return
        previous = self._target_nav_lock or {}
        self._target_nav_lock = {
            'target': target,
            'label': label or target,
            'target_point': target_point,
            'final_goal_pose': final_goal_pose,
            'final_distance_m': float(getattr(result, 'final_distance_m', 0.0) or 0.0),
            'bounded_step': bool(getattr(result, 'bounded_step', False)),
            'step': int(step_index),
            'attempts': int(previous.get('attempts', 0)),
            'blocked': bool(blocked),
        }
        self.get_logger().info(
            'target_nav_lock: remembered "%s" target=(%.2f,%.2f) final_goal=(%.2f,%.2f) '
            'bounded=%s blocked=%s final_distance=%.2fm'
            % (self._target_nav_lock['label'],
               target_point.point.x, target_point.point.y,
               final_goal_pose.pose.position.x, final_goal_pose.pose.position.y,
               self._target_nav_lock['bounded_step'],
               self._target_nav_lock['blocked'],
               self._target_nav_lock['final_distance_m']))
        self._activity(
            'target_nav_lock',
            step=step_index,
            label=self._target_nav_lock['label'],
            target_x=round(float(target_point.point.x), 2),
            target_y=round(float(target_point.point.y), 2),
            final_goal_x=round(float(final_goal_pose.pose.position.x), 2),
            final_goal_y=round(float(final_goal_pose.pose.position.y), 2),
            bounded=bool(self._target_nav_lock['bounded_step']),
            blocked=bool(self._target_nav_lock['blocked']),
            final_distance_m=round(float(self._target_nav_lock['final_distance_m']), 2))

    def _active_target_nav_lock(self, target):
        lock = self._target_nav_lock
        if not lock or lock.get('target') != target:
            return None
        if self.locked_target_approach_max_attempts <= 0:
            return None
        if int(lock.get('attempts', 0)) >= self.locked_target_approach_max_attempts:
            return None
        if not self._point_is_valid(lock.get('target_point')):
            return None
        return lock

    def _locked_target_action(self, obs, target, step_index):
        lock = self._active_target_nav_lock(target)
        if lock is None:
            return None
        strict = [
            c for c in (obs.candidates if obs is not None else [])
            if (c.source or '') == 'target' and distance_is_known(c.distance_m)
        ]
        if strict:
            best = max(strict, key=lambda c: (float(c.score), -float(c.distance_m)))
            return Action(
                DRIVE_TO_VISIBLE, mark_id=best.mark_id, arg_label=best.label,
                rationale=('target_nav_lock: strict target is visible again; update the '
                           'locked map point from mark %d instead of asking VLM to rethink '
                           'the scene' % best.mark_id))
        return Action(
            DRIVE_TO_VISIBLE, mark_id=0, arg_label='__locked_target__',
            rationale=('target_nav_lock: target was already confidently localized at '
                       '(%.2f, %.2f) in %s; continue toward that saved point even though '
                       'the object is not currently visible in the frame')
            % (lock['target_point'].point.x, lock['target_point'].point.y,
               lock['target_point'].header.frame_id))

    def _active_target_approach_blocked(self, target):
        block = self._target_approach_blocked
        if not block or block.get('target') != target:
            return None
        if self.target_approach_blocked_recovery_steps <= 0:
            return None
        if int(block.get('recoveries', 0)) >= self.target_approach_blocked_recovery_steps:
            return None
        return block

    def _target_approach_blocked_action(self, obs, target, step_index):
        block = self._active_target_approach_blocked(target)
        if block is None:
            return None
        remaining = self.target_approach_blocked_recovery_steps - int(
            block.get('recoveries', 0))
        label = str(block.get('label') or target)
        reason = str(block.get('reason') or 'approach aborted')
        distance = block.get('distance_m', None)
        dist_text = ('unknown' if not distance_is_known(distance)
                     else '%.2fm' % float(distance))
        return Action(
            DRIVE_FORWARD,
            forward_dist_m=self.target_approach_blocked_forward_m,
            arg_label=label,
            rationale=(
                'target_approach_blocked: confirmed target "%s" is localized but '
                'ApproachDetection found no safe bounded approach (%s, target_range=%s); '
                'advance %.2fm through the current free corridor to expand/reposition, '
                'then retry the saved target (%d recovery step(s) left)'
                % (label, reason, dist_text, self.target_approach_blocked_forward_m,
                   remaining)))

    @staticmethod
    def _is_target_approach_blocked_recovery(action):
        return bool(
            action is not None
            and action.kind in (TURN, DRIVE_FORWARD)
            and 'target_approach_blocked:' in (action.rationale or '').lower())

    def _remember_target_approach_blocked_motion(self, action, ok):
        if not self._is_target_approach_blocked_recovery(action):
            return
        block = self._target_approach_blocked
        if not block:
            return
        block['recoveries'] = int(block.get('recoveries', 0)) + 1
        self._activity(
            'target_approach_blocked_recovery',
            label=block.get('label', ''),
            recovery=block['recoveries'],
            max_recovery=self.target_approach_blocked_recovery_steps,
            motion_ok=bool(ok))

    def _remember_target_approach_blocked(self, action, obs, target, step_index, ok):
        if action.kind != DRIVE_TO_VISIBLE:
            return False
        if ok:
            if self._target_approach_blocked is not None:
                self.get_logger().info(
                    'target_approach_blocked: cleared after successful target approach')
            self._target_approach_blocked = None
            return False
        res = self._last_approach_result
        if getattr(res, 'outcome', None) != ApproachDetection.Result.ABORTED:
            return False
        cand = None
        if obs is not None and int(action.mark_id) != 0:
            cand = next((c for c in obs.candidates
                         if int(c.mark_id) == int(action.mark_id)), None)
        lock = self._active_target_nav_lock(target)
        label = (
            getattr(cand, 'label', None)
            or action.arg_label
            or (lock.get('label') if lock else None)
            or target)
        distance = (
            getattr(cand, 'distance_m', None)
            if cand is not None else
            (lock.get('final_distance_m') if lock else None))
        self._remember_target_nav_lock(
            target, label, res, step_index, allow_blocked=True)
        self._target_approach_blocked = {
            'target': target,
            'label': label,
            'distance_m': distance,
            'step': int(step_index),
            'recoveries': 0,
            'reason': 'no safe bounded approach',
        }
        self.get_logger().warn(
            'target_approach_blocked: "%s" approach aborted; will do %d recovery '
            'motion(s) before retrying target'
            % (label, self.target_approach_blocked_recovery_steps))
        self._activity(
            'target_approach_blocked',
            step=step_index,
            label=label,
            distance_m=distance_for_options(distance),
            recovery_steps=self.target_approach_blocked_recovery_steps,
            action=self._action_brief(action))
        return True

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
            if pixel is not None:
                pixel_x_norm = max(0.0, min(
                    1.0, float(getattr(pixel, 'x', center_x)) / float(self.camera_image_width)))
                # We currently only use x for edge guards; y is recorded for
                # future diagnostics without adding another camera-height param.
                pixel_y_norm = 0.5
            else:
                pixel_x_norm = center_x_norm
                pixel_y_norm = 0.5
            mark_id = int(getattr(c, 'mark_id', 0) or 0)
            cands.append(Candidate(mark_id=mark_id,
                                   label=str(getattr(c, 'label', '') or ''),
                                   score=float(getattr(c, 'confidence', 0.0) or 0.0),
                                   distance_m=distance_m,
                                   side=image_side(center_x_norm),
                                   center_x_norm=center_x_norm,
                                   pixel_x_norm=pixel_x_norm,
                                   pixel_y_norm=pixel_y_norm))
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
            return [], None
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
        if marks:
            backend = 'dino_office' if query else 'yoloe_all'
            self.get_logger().info('context_detect[%s]: %d object(s): %s' % (
                backend, len(marks), ', '.join(self._context_brief(m) for m in marks)))
        self._activity(
            'context_detect',
            backend='dino_office' if query else 'yoloe_all',
            objects=[{'mark_id': m.mark_id, 'label': m.label,
                      'score': round(float(m.score), 2),
                      'distance_m': distance_for_options(m.distance_m),
                      'side': m.side, 'relevance': m.relevance}
                     for m in marks])
        return marks, jpeg

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
        blocked_action = self._target_approach_blocked_action(obs, target, step)
        locked_action = None if blocked_action is not None else self._locked_target_action(
            obs, target, step)
        initial_scan_actions = [] if (blocked_action is not None or locked_action is not None) \
            else self._initial_scan_actions(obs, step)
        if blocked_action is not None:
            planner_name = 'target_approach_blocked'
        elif locked_action is not None:
            planner_name = 'target_nav_lock'
        elif initial_scan_actions:
            planner_name = 'initial_scan'
        else:
            planner_name = type(client).__name__
        self.get_logger().info(
            'observe@step %d: %d target detection(s)%s%s, notes=%d, map=%s -> planner=%s'
            % (step, len(obs.candidates), det, ctx, len(obs.notes_facts),
               'yes' if map_jpeg else 'no', planner_name))
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
        if initial_scan_actions:
            actions = list(initial_scan_actions)
            self.get_logger().info('plan@step %d: initial scan action(s): %s'
                                   % (step, ', '.join(self._action_brief(a)
                                                      for a in actions)))
            self._activity(
                'plan', step=step, latency_ms=0.0, source='initial_scan',
                actions=[{'action': self._action_brief(a),
                          'role': self._action_role(a, obs),
                          'rationale': a.rationale or ''} for a in actions])
            self.heartbeat.set_latency_ms(0.0)
            self.heartbeat.set_status(
                Heartbeat.DEGRADED if (self.cb.is_open or self._degrade.degraded)
                else Heartbeat.OK)
            return _PlanBundle(actions, pixels, obs)
        if blocked_action is not None:
            actions = [blocked_action]
            self.get_logger().info('plan@step %d: target blocked recovery action: %s'
                                   % (step, self._action_brief(blocked_action)))
            self._activity(
                'plan', step=step, latency_ms=0.0, source='target_approach_blocked',
                actions=[{'action': self._action_brief(blocked_action),
                          'role': self._action_role(blocked_action, obs),
                          'rationale': blocked_action.rationale or ''}])
            self.heartbeat.set_latency_ms(0.0)
            self.heartbeat.set_status(
                Heartbeat.DEGRADED if (self.cb.is_open or self._degrade.degraded)
                else Heartbeat.OK)
            return _PlanBundle(actions, pixels, obs)
        if locked_action is not None:
            actions = [locked_action]
            self.get_logger().info('plan@step %d: target nav-lock action: %s'
                                   % (step, self._action_brief(locked_action)))
            self._activity(
                'plan', step=step, latency_ms=0.0, source='target_nav_lock',
                actions=[{'action': self._action_brief(locked_action),
                          'role': self._action_role(locked_action, obs),
                          'rationale': locked_action.rationale or ''}])
            self.heartbeat.set_latency_ms(0.0)
            self.heartbeat.set_status(
                Heartbeat.DEGRADED if (self.cb.is_open or self._degrade.degraded)
                else Heartbeat.OK)
            return _PlanBundle(actions, pixels, obs)
        try:
            actions = list(client.plan_sequence(obs, jpeg, map_jpeg, n=self.replan_n))
            self.cb.record_success() if actions else self.cb.record_failure()
            actions = self._apply_target_lock_recovery(actions, obs, target, step)
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
            actions = self._apply_target_lock_recovery(actions, obs, target, step)
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

    def _should_launch_lead_replan(self, action, action_index, batch_len, already_pending):
        """Do not observe/replan while a turn is physically in progress.

        TURN actions are where motion blur hurt us most: Nav2 may report success
        before the camera image has visually settled. Keep async replan for other
        motions, but force a fresh post-settle observation after every TURN.
        """
        if action.kind == TURN and self.turn_settle_s > 0.0:
            return False
        return orch.should_launch_lead_replan(
            action_index, batch_len, self.async_replan, already_pending)

    def _settle_after_turn(self, action, ok):
        if not ok or action.kind != TURN or self.turn_settle_s <= 0.0:
            return
        delay = float(self.turn_settle_s)
        self.get_logger().info(
            'turn settle: waiting %.2fs before next observation to avoid motion-blurred detections'
            % delay)
        self._activity(
            'perception_settle',
            action=self._action_brief(action),
            duration_s=round(delay, 2),
            reason='post_turn_image_stabilization')
        time.sleep(delay)

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
        self._target_lock = None
        self._target_nav_lock = None
        self._target_approach_blocked = None
        self._corridor_scan = {}
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
                    action = self._normalize_turn_action(action, role)
                    role = self._action_role(action, bundle.obs)
                    action = self._semantic_explore_antioscillation(
                        action, bundle.obs, role)
                    # Repairs above may synthesize a new TURN. Clamp once more at
                    # the final execution boundary so VLM-level turns are never
                    # swallowed by Nav2 yaw tolerance as no-ops.
                    action = self._normalize_turn_action(action, role)
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
                    if self._should_launch_lead_replan(
                            action, i, len(bundle.actions), pending is not None):
                        pending = self._planner_pool.submit(self._compute_plan, target, step + 1)
                    t0 = time.monotonic()
                    ok = self._dispatch(action, bundle.pixels, target, step)
                    self.notes.add_fact('%s%s -> %s' % (
                        action.name,
                        (' ' + action.rationale) if action.rationale else '',
                        'ok' if ok else 'failed'))
                    self._activity('step_result', step=step,
                                   action=self._action_brief(action),
                                   result='ok' if ok else 'failed',
                                   duration_s=round(time.monotonic() - t0, 2))
                    blocked = self._remember_target_approach_blocked(
                        action, bundle.obs, target, step, ok)
                    self._remember_target_approach_blocked_motion(action, ok)
                    if blocked and pending is not None:
                        # A precomputed async plan may still contain the now-known
                        # blocked DRIVE_TO_VISIBLE retry. Drop it so the next plan
                        # sees target_approach_blocked and actively repositions.
                        pending.cancel()
                        pending = None
                    self._remember_semantic_motion(action, bundle.obs, role, ok)
                    self._publish_notes(target)
                    self._settle_after_turn(action, ok)
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
            if a.mark_id == 0 and a.arg_label == '__locked_target__':
                return 'DRIVE_TO_LOCKED_TARGET'
            return 'DRIVE_TO_VISIBLE mark=%d' % a.mark_id
        return a.name

    @staticmethod
    def _action_role(action, obs):
        """Human-readable intent class for logs/dashboard."""
        if action.kind == DRIVE_TO_VISIBLE:
            return 'target_approach'
        if action.kind in (TURN, DRIVE_FORWARD):
            rationale = (action.rationale or '').lower()
            if 'target_approach_blocked:' in rationale:
                return 'target_approach_blocked'
            if 'initial_scan:' in rationale:
                return 'initial_scan'
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

    def _normalize_turn_action(self, action, role):
        """Avoid no-op TURN actions that are smaller than Nav2's yaw tolerance."""
        if action.kind != TURN:
            return action
        yaw = float(action.turn_yaw_rad)
        if not math.isfinite(yaw):
            yaw = 0.0
        min_yaw = float(self.min_effective_turn_rad)
        if min_yaw <= 0.0 or abs(yaw) >= min_yaw:
            return action
        sign = 1.0 if yaw >= 0.0 else -1.0
        normalized = sign * min_yaw
        base = ('turn_guard: requested %.2frad is below the effective turn %.2frad; '
                'normalizing to %.2frad so Nav2 cannot treat it as already reached'
                % (yaw, min_yaw, normalized))
        if action.rationale:
            base += '; original rationale: ' + action.rationale
        return Action(TURN, turn_yaw_rad=normalized,
                      arg_label=action.arg_label, rationale=base)

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

        return Action(DRIVE_FORWARD,
                      forward_dist_m=max(0.05, self.semantic_probe_forward_m),
                      arg_label=action.arg_label,
                      rationale=(base + '; probe forward after inspecting context; '
                                 'navigation costmaps decide whether the short motion is safe'))

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

    def _dispatch(self, action, cand_pixels, target='', step_index=0):
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
            self._last_approach_result = None
            if action.mark_id == 0 and action.arg_label == '__locked_target__':
                return self._send_locked_target_approach(target, step_index)
            return self._send_approach_mark(
                action.mark_id, action.arg_label, cand_pixels, target, step_index)
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

    def _send_approach(self, label, target='', step_index=0, locked_target_point=None):
        g = ApproachDetection.Goal()
        g.request_id = self._goal_id()
        g.mission_epoch = self._epoch
        g.target_label = label or ''
        g.approach_offset = self.approach_offset
        g.max_pixel_age_s = 1.5
        if locked_target_point is not None:
            g.use_locked_target = True
            g.locked_target_point = locked_target_point
        self._last_approach_result = self._send_and_wait_result(orch.SKILL_APPROACH, g)
        self._remember_target_nav_lock(
            target, label, self._last_approach_result, step_index)
        return getattr(self._last_approach_result, 'outcome', None) == 0

    def _send_approach_mark(self, mark_id, label, cand_pixels, target='', step_index=0):
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
            return self._send_approach(label, target, step_index)
        finally:
            stop.set()

    def _send_locked_target_approach(self, target='', step_index=0):
        lock = self._active_target_nav_lock(target)
        if lock is None:
            self.get_logger().warn('DRIVE_TO_LOCKED_TARGET: no active target nav-lock')
            return False
        lock['attempts'] = int(lock.get('attempts', 0)) + 1
        self.get_logger().info(
            'target_nav_lock: continuing saved target "%s" attempt %d/%d'
            % (lock.get('label', target), lock['attempts'],
               self.locked_target_approach_max_attempts))
        return self._send_approach(
            str(lock.get('label', target) or target),
            target=target,
            step_index=step_index,
            locked_target_point=lock.get('target_point'))

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
