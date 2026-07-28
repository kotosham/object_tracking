#!/usr/bin/env python3
"""DetectTarget action server (Phase 3.2): on-demand open-vocab detection + Set-of-Mark.

The service-mode counterpart to the continuous rgb_tracker_node: instead of streaming
/target_pixel, it answers a DetectTarget goal (open-vocab ``query`` + ``conf_threshold``)
with a numbered Candidate[] and an optional annotated Set-of-Mark frame, so the VLM can
pick a target by ``mark_id`` (DRIVE_TO_VISIBLE). In hybrid mode, concrete target
queries use GroundingDINO+MobileSAM while DETECT_ALL stays on YOLOE's broad
vocabulary. The heavy torch/ultralytics imports are deferred to construction so
this module imports without a GPU (node returns ABORTED if a required backend
can't load). The VLM/executive owns all motion -- this node only perceives.
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
        self.declare_parameter('conf_default', -1.0)  # legacy override for both paths
        self.declare_parameter('target_conf_default', 0.50)
        self.declare_parameter('vocab_conf_default', 0.08)
        self.declare_parameter('min_mask_area', 200)
        self.declare_parameter('max_marks', 9)           # Set-of-Mark legibility cap
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('use_depth', True)
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 8.0)
        self.declare_parameter('depth_window', 2)        # +/- px median window
        self.declare_parameter('depth_point_strategy', 'nearest_mask')
        self.declare_parameter('nearest_depth_percentile', 2.0)
        self.declare_parameter('depth_match_tolerance_s', 0.2)
        self.declare_parameter('depth_buffer_size', 30)

        g = lambda n: self.get_parameter(n).value
        self.image_topic = g('image_topic')
        self.use_compressed = bool(g('use_compressed_input'))
        self.model_mode = str(g('model_mode')).strip().lower()
        legacy_conf_default = float(g('conf_default'))
        self.target_conf_default = float(g('target_conf_default'))
        self.vocab_conf_default = float(g('vocab_conf_default'))
        if legacy_conf_default > 0.0:
            self.target_conf_default = legacy_conf_default
            self.vocab_conf_default = legacy_conf_default
        self.min_mask_area = int(g('min_mask_area'))
        self.max_marks = int(g('max_marks'))
        self.jpeg_quality = int(g('jpeg_quality'))
        self.use_depth = bool(g('use_depth'))
        self.min_depth_m = float(g('min_depth_m'))
        self.max_depth_m = float(g('max_depth_m'))
        self.depth_window = int(g('depth_window'))
        self.depth_point_strategy = str(g('depth_point_strategy')).strip().lower()
        self.nearest_depth_percentile = float(g('nearest_depth_percentile'))
        self.depth_match_tolerance_s = float(g('depth_match_tolerance_s'))
        self.depth_buffer_size = max(1, int(g('depth_buffer_size')))

        self._bridge = CvBridge() if _HAVE_CV else None
        self._frame = None               # latest BGR frame
        self._frame_header = None
        self._depth_frames = deque(maxlen=self.depth_buffer_size)
        self._lock = threading.Lock()    # single-in-flight detection

        self.target_segmentor, self.vocab_segmentor = self._load_segmentors()
        # Backward-compatible alias for older tests/tools that only check that a
        # backend exists. Query routing below uses target_segmentor/vocab_segmentor.
        self.segmentor = self.target_segmentor or self.vocab_segmentor

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
            self._hb.set_status(Heartbeat.DOWN if not self._backends_ready()
                                else Heartbeat.DEGRADED)
            self.create_timer(1.0, self._update_health)

        self.get_logger().info(
            'detect_target_server up (Phase 3.2): mode=%s target_backend=%s '
            'vocab_backend=%s topic=%s compressed=%s '
            'max_marks=%d target_conf=%.2f vocab_conf=%.2f depth_sync=%.3fs%s'
            % (
               self.model_mode,
               self._backend_name(self.target_segmentor),
               self._backend_name(self.vocab_segmentor),
               self.image_topic,
               self.use_compressed,
               self.max_marks,
               self.target_conf_default,
               self.vocab_conf_default,
               self.depth_match_tolerance_s,
               '' if self._backends_ready() else ' [BACKEND FAILED -> some goals ABORT]',
            ))

    def _update_health(self):
        """Roll the heartbeat status from backend + camera-frame freshness."""
        if self._hb is None:
            return
        if not self._backends_ready():
            self._hb.set_status(Heartbeat.DOWN)
        elif (time.monotonic() - self._last_frame_mono) > 2.0:
            self._hb.set_status(Heartbeat.DEGRADED)   # no/stale camera frames
        else:
            self._hb.set_status(Heartbeat.OK)

    # ---- detector backend (heavy import deferred here) ----
    def _load_segmentors(self):
        if not _HAVE_CV:
            self.get_logger().error('cv2/cv_bridge unavailable; detector disabled')
            return None, None

        mode = self.model_mode
        if mode in ('hybrid', 'hybrid_dino_yoloe', 'dino_yoloe'):
            return self._load_backend('dino_mobilesam'), self._load_backend('yoloe')
        if mode == 'dino_mobilesam':
            # Concrete target mode only. DETECT_ALL is unavailable in this mode.
            return self._load_backend('dino_mobilesam'), None
        if mode == 'yoloe':
            yoloe = self._load_backend('yoloe')
            return yoloe, yoloe

        self.get_logger().error('model_mode "%s" not supported by DetectTarget' % mode)
        return None, None

    def _load_backend(self, backend_name):
        try:
            if backend_name == 'yoloe':
                from object_tracking.yoloe_image_segmentation import YOLOESegmentor
                seg = YOLOESegmentor()
            elif backend_name == 'dino_mobilesam':
                from object_tracking.dino_mobilesam_image_segmentation import (
                    GroundingDINOMobileSAMSegmentor,
                )
                seg = GroundingDINOMobileSAMSegmentor()
            else:
                self.get_logger().error('unknown detector backend "%s"' % backend_name)
                return None
            if hasattr(seg, 'runtime_info'):
                self.get_logger().info('%s: %s' % (backend_name, seg.runtime_info()))
            return seg
        except Exception as exc:                       # torch/weights missing, etc.
            self.get_logger().error('detector backend "%s" load failed: %r'
                                    % (backend_name, exc))
            return None

    @staticmethod
    def _backend_name(segmentor):
        if segmentor is None:
            return 'none'
        return type(segmentor).__name__

    def _backends_ready(self):
        if self.model_mode in ('hybrid', 'hybrid_dino_yoloe', 'dino_yoloe'):
            return self.target_segmentor is not None and self.vocab_segmentor is not None
        if self.model_mode == 'dino_mobilesam':
            return self.target_segmentor is not None
        return self.target_segmentor is not None and self.vocab_segmentor is not None

    def _conf_for_query(self, query, request_conf):
        if request_conf > 0.0:
            return float(request_conf)
        return self.target_conf_default if query else self.vocab_conf_default

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

    @staticmethod
    def _unknown_depth():
        return float('nan')

    def _valid_depth_values(self, values):
        values = np.asarray(values).ravel()
        return values[(values >= self.min_depth_m) & (values <= self.max_depth_m)
                      & np.isfinite(values)]

    def _scale_rgb_point_to_depth(self, cx, cy, rgb_shape, depth_shape):
        """Map RGB pixel coordinates into the aligned-depth image grid.

        RealSense aligned_depth_to_color can still be published at a lower
        resolution than RGB (e.g. RGB 640x480, depth 424x240). The detector runs
        on RGB pixels, while depth sampling indexes the depth image, so coordinates
        must be scaled before lookup.
        """
        dh, dw = depth_shape[:2]
        if rgb_shape is None:
            return int(round(cx)), int(round(cy))
        rh, rw = rgb_shape[:2]
        if rw <= 0 or rh <= 0:
            return int(round(cx)), int(round(cy))
        dx = int(round(float(cx) * float(dw) / float(rw)))
        dy = int(round(float(cy) * float(dh) / float(rh)))
        return dx, dy

    def _scale_rgb_bbox_to_depth(self, bbox, rgb_shape, depth_shape):
        if bbox is None or rgb_shape is None:
            return None
        dh, dw = depth_shape[:2]
        rh, rw = rgb_shape[:2]
        if rw <= 0 or rh <= 0:
            return None
        x1, y1, x2, y2 = bbox
        sx = float(dw) / float(rw)
        sy = float(dh) / float(rh)
        return (
            max(0, min(dw, int(round(float(x1) * sx)))),
            max(0, min(dh, int(round(float(y1) * sy)))),
            max(0, min(dw, int(round(float(x2) * sx)))),
            max(0, min(dh, int(round(float(y2) * sy)))),
        )

    def _scale_depth_point_to_rgb(self, dx, dy, rgb_shape, depth_shape):
        dh, dw = depth_shape[:2]
        if rgb_shape is None:
            return int(dx), int(dy)
        rh, rw = rgb_shape[:2]
        if rw <= 0 or rh <= 0 or dw <= 0 or dh <= 0:
            return int(dx), int(dy)
        # Use pixel-center mapping so 424x240 depth maps back into 640x480 RGB.
        rx = int(round((float(dx) + 0.5) * float(rw) / float(dw) - 0.5))
        ry = int(round((float(dy) + 0.5) * float(rh) / float(dh) - 0.5))
        return max(0, min(rw - 1, rx)), max(0, min(rh - 1, ry))

    def _scale_rgb_mask_to_depth(self, mask, rgb_shape, depth_shape):
        if mask is None or rgb_shape is None:
            return None
        mask = np.asarray(mask).astype(np.uint8)
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        rh, rw = rgb_shape[:2]
        dh, dw = depth_shape[:2]
        if rw <= 0 or rh <= 0 or dw <= 0 or dh <= 0:
            return None
        if mask.shape[:2] != (rh, rw):
            mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)

    def _bbox_mask_in_depth(self, bbox, rgb_shape, depth_shape):
        db = self._scale_rgb_bbox_to_depth(bbox, rgb_shape, depth_shape)
        if db is None:
            return None
        x1, y1, x2, y2 = db
        if x2 <= x1 or y2 <= y1:
            return None
        mask = np.zeros(depth_shape[:2], dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask

    def _front_surface_point_in_depth(self, depth, mask):
        """Pick a robust nearest valid depth point inside a depth-resolution mask."""
        if depth is None or mask is None:
            return None
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[:2] != depth.shape[:2]:
            return None
        valid_mask = mask & np.isfinite(depth) & (depth >= self.min_depth_m) & (depth <= self.max_depth_m)
        if not np.any(valid_mask):
            return None

        valid_depths = depth[valid_mask]
        pct = max(0.0, min(100.0, float(getattr(self, 'nearest_depth_percentile', 2.0))))
        cutoff = float(np.percentile(valid_depths, pct))
        front_mask = valid_mask & (depth <= cutoff)
        if not np.any(front_mask):
            min_depth = float(np.min(valid_depths))
            front_mask = valid_mask & (depth <= min_depth)

        ys, xs = np.where(front_mask)
        if xs.size == 0:
            return None

        # If multiple pixels are equally near, choose a representative point on that
        # front surface instead of a random edge pixel.
        mx = float(np.median(xs))
        my = float(np.median(ys))
        idx = int(np.argmin((xs.astype(np.float32) - mx) ** 2 + (ys.astype(np.float32) - my) ** 2))
        dx, dy = int(xs[idx]), int(ys[idx])
        return dx, dy, float(depth[dy, dx])

    def _sample_depth(self, depth, cx, cy, rgb_shape=None, bbox=None):
        """Metric depth near an RGB detection center; NaN if unknown.

        Primary sample: median valid depth in a small window around the center,
        after scaling RGB coordinates into the depth image grid. If the center is
        a hole (common on chairs/mesh/reflective surfaces), fall back to a lower
        percentile over the scaled bbox so we prefer the object surface over the
        farther background.
        """
        if depth is None:
            return self._unknown_depth()
        h, w = depth.shape[:2]
        dx, dy = self._scale_rgb_point_to_depth(cx, cy, rgb_shape, depth.shape)
        if not (0 <= dy < h and 0 <= dx < w):
            return self._unknown_depth()
        r = max(0, self.depth_window)
        patch = depth[max(0, dy - r):dy + r + 1, max(0, dx - r):dx + r + 1]
        valid = self._valid_depth_values(patch)
        if valid.size:
            return float(np.median(valid))

        db = self._scale_rgb_bbox_to_depth(bbox, rgb_shape, depth.shape)
        if db is None:
            return self._unknown_depth()
        x1, y1, x2, y2 = db
        if x2 <= x1 or y2 <= y1:
            return self._unknown_depth()
        valid = self._valid_depth_values(depth[y1:y2, x1:x2])
        if not valid.size:
            return self._unknown_depth()
        return float(np.percentile(valid, 20))

    def _sample_depth_point(self, depth, cx, cy, rgb_shape=None, bbox=None, mask=None):
        """Return (rgb_x, rgb_y, depth_m) for navigation.

        In nearest_mask mode we prefer the nearest valid depth point inside the
        segmentation mask (or bbox fallback). This makes DRIVE_TO_VISIBLE aim at the
        visible front surface of the object rather than the mask's geometric center.
        """
        if depth is None:
            return int(round(cx)), int(round(cy)), self._unknown_depth()

        if self.depth_point_strategy in ('nearest', 'nearest_mask', 'front_surface'):
            depth_mask = self._scale_rgb_mask_to_depth(mask, rgb_shape, depth.shape)
            if depth_mask is None:
                depth_mask = self._bbox_mask_in_depth(bbox, rgb_shape, depth.shape)
            front = self._front_surface_point_in_depth(depth, depth_mask)
            if front is not None:
                dx, dy, z = front
                rx, ry = self._scale_depth_point_to_rgb(dx, dy, rgb_shape, depth.shape)
                return rx, ry, z

        return int(round(cx)), int(round(cy)), self._sample_depth(
            depth, cx, cy, rgb_shape=rgb_shape, bbox=bbox)

    # ---- DetectTarget goal ----
    def _execute(self, goal_handle):
        req = goal_handle.request
        result = DetectTarget.Result()
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
        fb = DetectTarget.Feedback()
        seg_t0 = time.monotonic()
        try:
            if query:
                if self.target_segmentor is None:
                    raise RuntimeError('target detector backend is not available')
                segmentor = self.target_segmentor
                conf = self._conf_for_query(query, float(req.conf_threshold))
                dets = segmentor.segment_all(frame, query, conf=conf,
                                             min_mask_area=self.min_mask_area)
            else:
                if self.vocab_segmentor is None:
                    raise RuntimeError('DETECT_ALL backend is not available')
                segmentor = self.vocab_segmentor
                conf = self._conf_for_query(query, float(req.conf_threshold))
                dets = segmentor.segment_vocab(frame, conf=conf,
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
            for d in marked:                       # fill metric depth and navigation pixel
                d.cx, d.cy, d.depth_m = self._sample_depth_point(
                    depth, d.cx, d.cy, frame.shape, d.bbox, getattr(d, 'mask', None))
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
        self.get_logger().info('detect_target "%s": %d candidate(s) via %s (conf>=%.2f)'
                               % (query or '<all>', len(marked),
                                  self._backend_name(segmentor), conf))
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
