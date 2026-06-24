#!/usr/bin/env python3
"""Planner Orchestrator (Phase 4): VLM-mode planner over the FLAT executive.

Replans every N ATOMIC steps. Each step: build an Observation from the latest
frontiers + detections + notes (+ the camera frame for the VLM), ask the client
(mock or OpenAI-compatible) for up to N atomic actions, and dispatch each to an
EXISTING FLAT skill on the Pi executive:
  TURN / DRIVE_FORWARD -> GoToPose at a pose RELATIVE to the robot's real pose
  DRIVE_TO_VISIBLE     -> ApproachDetection
  GO_TO_FRONTIER       -> ExploreFrontier(id)
  GET_OBSERVATION      -> GetObservation ; STOP -> Stop ; DONE -> finish
The VLM is never on the reactive path; the executive owns motion + safety. A
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

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import (Buffer, ConnectivityException, ExtrapolationException,
                     LookupException, TransformListener)

from ar_project_msgs.action import (ApproachDetection, ExploreFrontier,
                                    GetObservation, GoToPose, Stop)
from ar_project_msgs.msg import FrontierArray
from object_tracking_msgs.msg import Notes

from fleet_comms.heartbeat import HeartbeatPublisher
from planner_orchestrator import orchestration as orch
from planner_orchestrator.planner_logic import (
    Candidate, CircuitBreaker, FrontierOpt, NotesBuffer, Observation,
    DRIVE_FORWARD, DRIVE_TO_VISIBLE, GET_OBSERVATION, GO_TO_FRONTIER, STOP, TURN,
)
from planner_orchestrator.vlm_client import make_client

try:
    import cv2
    from cv_bridge import CvBridge
    _HAVE_CV = True
except Exception:                       # mock mode needs no image pipeline
    _HAVE_CV = False


def _yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


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
        self.declare_parameter('max_travel_m', 2.5)
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
        g = lambda n: self.get_parameter(n).value
        self.replan_n = max(1, int(g('replan_every_n')))
        self.turn_step = float(g('turn_step_rad'))
        self.fwd_step = float(g('forward_step_m'))
        self.max_travel = float(g('max_travel_m'))
        self.approach_offset = float(g('approach_offset'))
        self.max_steps = int(g('max_steps'))
        self.map_frame = g('map_frame')
        self.robot_frame = g('robot_frame')
        self.skill_wait_s = float(g('skill_wait_s'))
        self.result_timeout_s = float(g('result_timeout_s'))
        self.min_step_s = float(g('min_step_s'))

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
        self.notes = NotesBuffer()
        self._epoch = int(g('mission_epoch'))

        # ---- inputs ----
        self._frontiers = None
        self._pixel = None
        self._jpeg = None
        self._bridge = CvBridge() if _HAVE_CV else None
        sub = ReentrantCallbackGroup()
        self.create_subscription(FrontierArray, '/frontiers', self._on_frontiers, 1,
                                 callback_group=sub)
        self.create_subscription(PointStamped, '/target_pixel', self._on_pixel, 1,
                                 callback_group=sub)
        if _HAVE_CV:
            self.create_subscription(Image, '/camera/camera/color/image_raw',
                                     self._on_image, 1, callback_group=sub)
        self.create_subscription(String, '/vlm_mission', self._on_mission, 1,
                                 callback_group=sub)
        self.notes_pub = self.create_publisher(Notes, '/planner/notes', 1)

        # ---- executive skill clients (loopback-style poll on a reentrant group) ----
        cg = ReentrantCallbackGroup()
        self._ac = {
            orch.SKILL_EXPLORE: ActionClient(self, ExploreFrontier, 'explore_frontier', callback_group=cg),
            orch.SKILL_GO_TO_POSE: ActionClient(self, GoToPose, 'go_to_pose', callback_group=cg),
            orch.SKILL_APPROACH: ActionClient(self, ApproachDetection, 'approach_detection', callback_group=cg),
            orch.SKILL_GET_OBS: ActionClient(self, GetObservation, 'get_observation', callback_group=cg),
            orch.SKILL_STOP: ActionClient(self, Stop, 'stop', callback_group=cg),
        }
        self._tf = Buffer()
        self._tfl = TransformListener(self._tf, self)
        self._busy = False
        self.get_logger().info(
            'planner_orchestrator up (Phase 4 VLM mode): client=%s creds=%s replan_every_n=%d. '
            'Publish target on /vlm_mission to start.'
            % (type(self.client).__name__, self._cred_src, self.replan_n))

    # ---- input callbacks ----
    def _on_frontiers(self, msg):
        self._frontiers = msg

    def _on_pixel(self, msg):
        self._pixel = msg

    def _on_image(self, msg):
        try:
            cv = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            ok, buf = cv2.imencode('.jpg', cv)
            if ok:
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
    def _observation(self, target, step_index):
        cands = []
        if self._pixel is not None:      # one Set-of-Mark candidate from the live detection
            cands.append(Candidate(mark_id=0, label=target, score=1.0))
        fr = []
        if self._frontiers is not None:
            for f in self._frontiers.frontiers:
                fr.append(FrontierOpt(id=int(f.id), distance_m=float(f.distance_m),
                                      score=float(f.score)))
        return Observation(target=target, candidates=cands, frontiers=fr,
                           notes_facts=self.notes.facts, step_index=step_index)

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

    # ---- mission loop: replan every N atomic steps ----
    def _run_mission(self, target):
        self.get_logger().info('VLM mission start: target="%s"' % target)
        self.notes = NotesBuffer()
        self.cb = CircuitBreaker()
        step = 0
        try:
            while rclpy.ok() and step < self.max_steps:
                obs = self._observation(target, step)
                try:
                    plan = self.client.plan_sequence(obs, self._jpeg, n=self.replan_n)
                    self.cb.record_success()
                except Exception as e:
                    self.cb.record_failure()
                    self.get_logger().warn('VLM plan failed (%s); cb_open=%s'
                                           % (e, self.cb.is_open))
                    if self.cb.is_open:
                        self.get_logger().error('circuit-breaker OPEN -> degrade VLM->FLAT; stopping')
                        self._dispatch_stop()
                        break
                    continue
                # execute this plan (up to N atomic actions), then replan
                replan = False
                for action in plan:
                    self.get_logger().info('step %d: %s (%s)'
                                           % (step, action.name, action.rationale or ''))
                    if orch.is_terminal(action.kind):
                        self.get_logger().info('VLM mission finished: %s' % action.name)
                        if action.kind == STOP:
                            self._dispatch_stop()
                        self._publish_notes(target)
                        return
                    t0 = time.monotonic()
                    ok = self._dispatch(action)
                    self.notes.add_fact('%s%s -> %s' % (
                        action.name,
                        (' ' + action.rationale) if action.rationale else '',
                        'ok' if ok else 'failed'))
                    self._publish_notes(target)
                    step += 1
                    replan = True
                    dt = time.monotonic() - t0
                    if dt < self.min_step_s:      # don't hammer on instant-reached skills
                        time.sleep(self.min_step_s - dt)
                    if step >= self.max_steps:
                        break
                if not replan:
                    break
            self.get_logger().info('VLM mission ended after %d steps' % step)
        finally:
            self._busy = False

    # ---- dispatch one atomic action to the matching FLAT skill ----
    def _dispatch(self, action):
        skill = orch.skill_for_action(action.kind)
        if action.kind in (TURN, DRIVE_FORWARD):
            pose = self._robot_pose()
            if pose is None:
                self.get_logger().warn('no %s->%s TF; skip motion' % (self.map_frame, self.robot_frame))
                return False
            gx, gy, gyaw = orch.relative_goal(pose[0], pose[1], pose[2], action)
            return self._send_goto(gx, gy, gyaw)
        if action.kind == GO_TO_FRONTIER:
            return self._send_explore(action.frontier_id)
        if action.kind == DRIVE_TO_VISIBLE:
            return self._send_approach(action.arg_label)
        if action.kind == GET_OBSERVATION:
            return self._send_getobs()
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

    def _send_explore(self, frontier_id):
        g = ExploreFrontier.Goal()
        g.request_id = self._goal_id()
        g.mission_epoch = self._epoch
        g.frontier_id = int(frontier_id)
        g.max_travel_m = self.max_travel
        return self._send_and_wait(orch.SKILL_EXPLORE, g)

    def _send_approach(self, label):
        g = ApproachDetection.Goal()
        g.request_id = self._goal_id()
        g.mission_epoch = self._epoch
        g.target_label = label or ''
        g.approach_offset = self.approach_offset
        g.max_pixel_age_s = 1.5
        return self._send_and_wait(orch.SKILL_APPROACH, g)

    def _send_getobs(self):
        g = GetObservation.Goal()
        g.request_id = self._goal_id()
        g.mission_epoch = self._epoch
        g.with_setofmark = True
        return self._send_and_wait(orch.SKILL_GET_OBS, g)

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
