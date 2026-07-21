#!/usr/bin/env python3
"""DetectTarget action server (Phase 3.2): on-demand open-vocab detection + Set-of-Mark.

The service-mode counterpart to the continuous rgb_tracker_node: instead of streaming
/target_pixel, it answers a DetectTarget goal (open-vocab ``query`` + ``conf_threshold``)
with a numbered Candidate[] and an optional annotated Set-of-Mark frame, so the VLM can
pick a target by ``mark_id`` (DRIVE_TO_VISIBLE). YOLOE is the default backend; the heavy
torch/ultralytics import is deferred to construction so this module imports without a GPU
(node returns ABORTED if the backend can't load). The VLM/executive owns all motion --
this node only perceives.
"""
import threading
import time
from collections import deque

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage, Image, RegionOfInterest

from object_tracking_msgs.action import DetectTarget
from object_tracking_msgs.msg import Candidate

from object_tracking.setofmark import assign_marks, render_setofmark

try:
    import cv2
    import numpy as np
    from cv_bridge import CvBridge
    _HAVE_CV = True
except Exception:                       # pragma: no cover
    _HAVE_CV = False

try:                                    # fleet-wide health bus (ar_project ws)
    from ar_project_msgs.msg import Heartbeat
    from fleet_comms.heartbeat import HeartbeatPublisher
    _HAVE_HEARTBEAT = True
except Exception:                       # standalone/sim without fleet_comms
    _HAVE_HEARTBEAT = False


class DetectTargetServer(Node):
    def __init__(self):
        super().__init__('detect_target_server')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('use_compressed_input', False)
        self.declare_parameter('input_reliability', 'best_effort')
        self.declare_parameter('model_mode', 'yoloe')
        self.declare_parameter('conf_default', 0.25)
        self.declare_parameter('min_mask_area', 200)
        self.declare_parameter('max_marks', 9)           # Set-of-Mark legibility cap
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('use_depth', True)
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 8.0)
        self.declare_parameter('depth_window', 2)        # +/- px median window
        self.declare_parameter('depth_match_tolerance_s', 0.2)
        self.declare_parameter('depth_buffer_size', 30)

        g = lambda n: self.get_parameter(n).value
        self.image_topic = g('image_topic')
        self.use_compressed = bool(g('use_compressed_input'))
        self.model_mode = str(g('model_mode')).strip().lower()
        self.conf_default = float(g('conf_default'))
        self.min_mask_area = int(g('min_mask_area'))
        self.max_marks = int(g('max_marks'))
        self.jpeg_quality = int(g('jpeg_quality'))
        self.use_depth = bool(g('use_depth'))
        self.min_depth_m = float(g('min_depth_m'))
        self.max_depth_m = float(g('max_depth_m'))
        self.depth_window = int(g('depth_window'))
        self.depth_match_tolerance_s = float(g('depth_match_tolerance_s'))
        self.depth_buffer_size = max(1, int(g('depth_buffer_size')))

        self._bridge = CvBridge() if _HAVE_CV else None
        self._frame = None               # latest BGR frame
        self._frame_header = None
        self._depth_frames = deque(maxlen=self.depth_buffer_size)
        self._lock = threading.Lock()    # single-in-flight detection

        self.segmentor = self._load_segmentor()

        sub_group = ReentrantCallbackGroup()
        q = QoSProfile(depth=1)
        q.reliability = (ReliabilityPolicy.BEST_EFFORT
                         if str(g('input_reliability')).strip().lower() == 'best_effort'
                         else ReliabilityPolicy.RELIABLE)
        q.durability = DurabilityPolicy.VOLATILE
        msg_type = CompressedImage if self.use_compressed else Image
        # image_transport publishes CompressedImage on '<base>/compressed', not on
        # the base topic. Subscribing CompressedImage to the bare base topic used
        # to match NO publisher, so the detector silently received zero frames.
        image_sub_topic = self.image_topic
        if self.use_compressed and not image_sub_topic.endswith('/compressed'):
            image_sub_topic = image_sub_topic + '/compressed'
            self.get_logger().info(
                'use_compressed_input: subscribing to "%s" (appended /compressed '
                'to the base image topic)' % image_sub_topic)
        self.create_subscription(msg_type, image_sub_topic, self._on_image, q,
                                 callback_group=sub_group)
        if self.use_depth:
            self.create_subscription(Image, str(g('depth_topic')), self._on_depth, q,
                                     callback_group=sub_group)

        self._srv = ActionServer(
            self, DetectTarget, 'detect_target',
            execute_callback=self._execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _c: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup())

        # Fleet heartbeat ('detector' row on the mission dashboard): DOWN while
        # the backend failed to load, DEGRADED until camera frames flow (and when
        # they go stale), OK while frames are fresh. Inference latency is fed
        # from _execute. No-op when fleet_comms is not on the path (bare sim).
        self._last_frame_mono = 0.0
        self._hb = None
        if _HAVE_HEARTBEAT:
            self._hb = HeartbeatPublisher(self, 'detector', period_s=0.5)
            self._hb.set_status(Heartbeat.DOWN if self.segmentor is None
                                else Heartbeat.DEGRADED)
            self.create_timer(1.0, self._update_health)

        self.get_logger().info(
            'detect_target_server up (Phase 3.2): backend=%s topic=%s compressed=%s '
            'max_marks=%d depth_sync=%.3fs%s'
            % (self.model_mode, self.image_topic, self.use_compressed, self.max_marks,
               self.depth_match_tolerance_s,
               '' if self.segmentor is not None else ' [BACKEND FAILED -> goals ABORT]'))

    def _update_health(self):
        """Roll the heartbeat status from backend + camera-frame freshness."""
        if self._hb is None:
            return
        if self.segmentor is None:
            self._hb.set_status(Heartbeat.DOWN)
        elif (time.monotonic() - self._last_frame_mono) > 2.0:
            self._hb.set_status(Heartbeat.DEGRADED)   # no/stale camera frames
        else:
            self._hb.set_status(Heartbeat.OK)

    # ---- detector backend (heavy import deferred here) ----
    def _load_segmentor(self):
        if not _HAVE_CV:
            self.get_logger().error('cv2/cv_bridge unavailable; detector disabled')
            return None
        try:
            if self.model_mode == 'yoloe':
                from object_tracking.yoloe_image_segmentation import YOLOESegmentor
                seg = YOLOESegmentor()
                self.get_logger().info(seg.runtime_info())
                return seg
            self.get_logger().error('model_mode "%s" not supported by DetectTarget yet'
                                     % self.model_mode)
            return None
        except Exception as exc:                       # torch/weights missing, etc.
            self.get_logger().error('detector backend load failed: %r' % (exc,))
            return None

    # ---- camera ----
    def _on_image(self, msg):
        if not _HAVE_CV:
            return
        try:
            if self.use_compressed:
                arr = np.frombuffer(msg.data, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn('frame decode failed: %r' % (exc,))
            return
        if frame is not None:
            with self._lock:
                self._frame = frame
                self._frame_header = msg.header
            self._last_frame_mono = time.monotonic()

    def _on_depth(self, msg):
        if not _HAVE_CV:
            return
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().warn('depth decode failed: %r' % (exc,))
            return
        depth = np.asarray(depth)
        if np.issubdtype(depth.dtype, np.integer):    # 16UC1 in mm -> meters
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
        with self._lock:
            self._depth_frames.append((self._stamp_to_ns(msg.header.stamp), depth))

    @staticmethod
    def _stamp_to_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _match_depth_locked(self, image_header):
        """Return the closest depth frame to the RGB stamp, or None if too far."""
        if not self._depth_frames:
            return None
        if image_header is None:
            return self._depth_frames[-1][1]
        target_ns = self._stamp_to_ns(image_header.stamp)
        if target_ns <= 0:
            return self._depth_frames[-1][1]
        best_ns, best_depth = min(
            self._depth_frames,
            key=lambda item: abs(item[0] - target_ns),
        )
        delta_s = abs(best_ns - target_ns) / 1e9
        if delta_s > self.depth_match_tolerance_s:
            self.get_logger().warn(
                'detect_target: no synchronized depth for RGB stamp; closest delta '
                '%.3fs > %.3fs' % (delta_s, self.depth_match_tolerance_s),
                throttle_duration_sec=2.0,
            )
            return None
        return best_depth

    def _sample_depth(self, depth, cx, cy):
        """Median valid depth (m) in a small window around (cx,cy); 0.0 if none."""
        if depth is None:
            return 0.0
        h, w = depth.shape[:2]
        if not (0 <= cy < h and 0 <= cx < w):
            return 0.0
        r = max(0, self.depth_window)
        patch = depth[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].ravel()
        valid = patch[(patch >= self.min_depth_m) & (patch <= self.max_depth_m)
                      & np.isfinite(patch)]
        return float(np.median(valid)) if valid.size else 0.0

    # ---- DetectTarget goal ----
    def _execute(self, goal_handle):
        req = goal_handle.request
        result = DetectTarget.Result()
        if self.segmentor is None:
            goal_handle.abort()
            result.outcome = DetectTarget.Result.ABORTED
            return result

        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            header = self._frame_header
            depth = self._match_depth_locked(header) if self.use_depth else None
        if frame is None:
            self.get_logger().warn('detect_target: no camera frame yet')
            goal_handle.abort()
            result.outcome = DetectTarget.Result.ABORTED
            return result

        # The query is passed straight through (open-vocab object class). An EMPTY
        # query means DETECT_ALL: detect every object in a broad built-in vocabulary
        # and report each with its OWN predicted class, rather than one named target.
        query = (req.query or '').strip()
        conf = req.conf_threshold if req.conf_threshold > 0.0 else self.conf_default
        fb = DetectTarget.Feedback()
        seg_t0 = time.monotonic()
        try:
            if query:
                dets = self.segmentor.segment_all(frame, query, conf=conf,
                                                  min_mask_area=self.min_mask_area)
            else:
                dets = self.segmentor.segment_vocab(frame, conf=conf,
                                                    min_mask_area=self.min_mask_area)
        except Exception as exc:
            self.get_logger().error('segment failed: %r' % (exc,))
            goal_handle.abort()
            result.outcome = DetectTarget.Result.ABORTED
            return result
        if self._hb is not None:
            self._hb.set_latency_ms((time.monotonic() - seg_t0) * 1e3)

        marked = assign_marks(dets, conf_threshold=conf, max_marks=self.max_marks)
        if self.use_depth:
            for d in marked:                       # fill metric depth at each center
                d.depth_m = self._sample_depth(depth, d.cx, d.cy)
        fb.frames_processed = 1
        fb.best_confidence = float(marked[0].confidence) if marked else 0.0
        goal_handle.publish_feedback(fb)

        frame_id = header.frame_id if header is not None else ''
        stamp = header.stamp if header is not None else self.get_clock().now().to_msg()
        result.candidates = [self._to_candidate(d, frame_id, stamp) for d in marked]
        if req.render_setofmark and marked:
            result.annotated = self._encode_setofmark(frame, marked, stamp, frame_id)
        result.outcome = (DetectTarget.Result.FOUND if marked
                          else DetectTarget.Result.NOT_FOUND)
        self.get_logger().info('detect_target "%s": %d candidate(s) (conf>=%.2f)'
                               % (query or '<all>', len(marked), conf))
        goal_handle.succeed()
        return result

    def _to_candidate(self, d, frame_id, stamp):
        c = Candidate()
        c.mark_id = int(d.mark_id)
        c.label = d.label
        c.confidence = float(d.confidence)
        c.pixel = Point(x=float(d.cx), y=float(d.cy), z=float(d.depth_m))
        c.source_frame_id = frame_id
        c.stamp = stamp
        x1, y1, x2, y2 = d.bbox
        c.bbox = RegionOfInterest(x_offset=max(0, int(x1)), y_offset=max(0, int(y1)),
                                  width=max(0, int(x2 - x1)), height=max(0, int(y2 - y1)),
                                  do_rectify=False)
        return c

    def _encode_setofmark(self, frame, marked, stamp, frame_id):
        annotated = render_setofmark(frame, marked)
        ok, buf = cv2.imencode('.jpg', annotated,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.format = 'jpeg'
        if ok:
            msg.data = buf.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = DetectTargetServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
