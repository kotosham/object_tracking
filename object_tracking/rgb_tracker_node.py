import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32, String, UInt32


class RGBTrackerNode(Node):
    def __init__(self):
        super().__init__('rgb_tracker_node')
        self.get_logger().info('RGB tracker node initialized')

        self.declare_parameter('use_sam', False)
        self.declare_parameter('model_mode', 'auto')
        self.declare_parameter('search_angular_speed', 0.5)
        self.declare_parameter('use_compressed_input', True)
        self.declare_parameter('input_reliability', 'best_effort')
        self.declare_parameter('goal_locked_topic', '/target_goal_locked')
        self.declare_parameter('enable_search_rotation', False)
        self.declare_parameter('burst_quiet_period', 2.0)
        self.declare_parameter('burst_complete_topic', '/tracker/burst_complete')
        self.declare_parameter('tracking_mode', 'burst')
        self.declare_parameter('use_depth_input', False)
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('depth_match_tolerance', 0.2)
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 6.0)
        self.declare_parameter('nearest_depth_band_m', 0.02)
        self.declare_parameter('nearest_depth_percentile', 5.0)
        self.declare_parameter('nearest_depth_min_pixels', 3)
        self.declare_parameter('target_publish_rate', 3.0)
        self.declare_parameter('continuous_frame_max_age', 2.0)
        self.declare_parameter('continuous_rgb_stamp_max_age', 0.0)
        self.declare_parameter('publish_mask_in_continuous', False)
        self.declare_parameter('target_anchor_timeout_s', 3.0)
        self.declare_parameter('florence2_model_id', 'microsoft/Florence-2-base-ft')
        self.declare_parameter('florence2_task_prompt', '<REFERRING_EXPRESSION_SEGMENTATION>')
        self.declare_parameter('florence2_max_new_tokens', 1024)
        self.declare_parameter('florence2_num_beams', 3)
        self.declare_parameter('clip_threshold', 0.70)
        self.declare_parameter('clip_min_mask_area', 100)
        self.declare_parameter('dino_box_threshold', 0.50)
        self.declare_parameter('dino_mobilesam_min_mask_area', 100)
        self.declare_parameter('florence2_min_mask_area', 100)
        self.declare_parameter('yoloe_conf_threshold', 0.12)
        self.declare_parameter('yoloe_min_mask_area', 100)
        self.declare_parameter('cv_runtime_topic', '/experiment/cv_runtime')
        self.declare_parameter('cv_model_topic', '/experiment/cv_model')

        use_sam = self.get_parameter('use_sam').get_parameter_value().bool_value
        requested_mode = self.get_parameter('model_mode').get_parameter_value().string_value.strip().lower()
        self.search_angular_speed = self.get_parameter('search_angular_speed').get_parameter_value().double_value
        self.use_compressed_input = self.get_parameter('use_compressed_input').get_parameter_value().bool_value
        self.input_reliability = self.get_parameter('input_reliability').get_parameter_value().string_value.strip().lower()
        self.goal_locked_topic = self.get_parameter('goal_locked_topic').get_parameter_value().string_value
        self.enable_search_rotation = self.get_parameter('enable_search_rotation').get_parameter_value().bool_value
        self.burst_quiet_period = self.get_parameter('burst_quiet_period').get_parameter_value().double_value
        self.burst_complete_topic = self.get_parameter('burst_complete_topic').get_parameter_value().string_value
        self.tracking_mode = self.get_parameter('tracking_mode').get_parameter_value().string_value.strip().lower()
        self.use_depth_input = self.get_parameter('use_depth_input').get_parameter_value().bool_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.depth_match_tolerance = self.get_parameter('depth_match_tolerance').get_parameter_value().double_value
        self.min_depth_m = self.get_parameter('min_depth_m').get_parameter_value().double_value
        self.max_depth_m = self.get_parameter('max_depth_m').get_parameter_value().double_value
        self.nearest_depth_band_m = self.get_parameter('nearest_depth_band_m').get_parameter_value().double_value
        self.nearest_depth_percentile = (
            self.get_parameter('nearest_depth_percentile').get_parameter_value().double_value
        )
        self.nearest_depth_min_pixels = max(
            1,
            self.get_parameter('nearest_depth_min_pixels').get_parameter_value().integer_value,
        )
        self.target_publish_rate = self.get_parameter('target_publish_rate').get_parameter_value().double_value
        self.continuous_frame_max_age = self.get_parameter('continuous_frame_max_age').get_parameter_value().double_value
        self.continuous_rgb_stamp_max_age = (
            self.get_parameter('continuous_rgb_stamp_max_age').get_parameter_value().double_value
        )
        self.publish_mask_in_continuous = self.get_parameter('publish_mask_in_continuous').get_parameter_value().bool_value
        self.target_anchor_timeout_s = self.get_parameter('target_anchor_timeout_s').get_parameter_value().double_value
        self.florence2_model_id = self.get_parameter('florence2_model_id').get_parameter_value().string_value
        self.florence2_task_prompt = self.get_parameter('florence2_task_prompt').get_parameter_value().string_value
        self.florence2_max_new_tokens = self.get_parameter('florence2_max_new_tokens').get_parameter_value().integer_value
        self.florence2_num_beams = self.get_parameter('florence2_num_beams').get_parameter_value().integer_value
        self.clip_threshold = self.get_parameter('clip_threshold').get_parameter_value().double_value
        self.clip_min_mask_area = self.get_parameter('clip_min_mask_area').get_parameter_value().integer_value
        self.dino_box_threshold = self.get_parameter('dino_box_threshold').get_parameter_value().double_value
        self.dino_mobilesam_min_mask_area = (
            self.get_parameter('dino_mobilesam_min_mask_area').get_parameter_value().integer_value
        )
        self.florence2_min_mask_area = self.get_parameter('florence2_min_mask_area').get_parameter_value().integer_value
        self.yoloe_conf_threshold = self.get_parameter('yoloe_conf_threshold').get_parameter_value().double_value
        self.yoloe_min_mask_area = self.get_parameter('yoloe_min_mask_area').get_parameter_value().integer_value
        self.cv_runtime_topic = self.get_parameter('cv_runtime_topic').get_parameter_value().string_value
        self.cv_model_topic = self.get_parameter('cv_model_topic').get_parameter_value().string_value

        if self.tracking_mode not in ('burst', 'continuous'):
            self.get_logger().warn(
                f'Unknown tracking_mode "{self.tracking_mode}", falling back to "burst".'
            )
            self.tracking_mode = 'burst'

        self.bridge = CvBridge()
        self.model_mode = self._resolve_model_mode(requested_mode, use_sam)
        self.model_label = self._get_model_label(self.model_mode)
        if self.input_reliability not in ('best_effort', 'reliable'):
            self.get_logger().warn(
                f'Unknown input_reliability "{self.input_reliability}", falling back to best_effort.'
            )
            self.input_reliability = 'best_effort'

        if self.model_mode == 'dino_mobilesam':
            from object_tracking.dino_mobilesam_image_segmentation import GroundingDINOMobileSAMSegmentor
            self.segmentor = GroundingDINOMobileSAMSegmentor()
        elif self.model_mode == 'florence2':
            from object_tracking.florence2_image_segmentation import Florence2Segmentor
            self.segmentor = Florence2Segmentor(
                model_id=self.florence2_model_id,
                task_prompt=self.florence2_task_prompt,
                max_new_tokens=self.florence2_max_new_tokens,
                num_beams=self.florence2_num_beams,
            )
        elif self.model_mode == 'yoloe':
            from object_tracking.yoloe_image_segmentation import YOLOESegmentor
            self.segmentor = YOLOESegmentor()
        else:
            from object_tracking.clip_image_segmentation import CLIPSegmentor
            self.segmentor = CLIPSegmentor()

        self.get_logger().info(f'Using segmentation backend: {self.model_label}')
        if hasattr(self.segmentor, 'runtime_info'):
            self.get_logger().info(f'Inference runtime: {self.segmentor.runtime_info()}')
        self.get_logger().info(
            f'Search rotation is {"enabled" if self.enable_search_rotation else "disabled"} '
            f'(cmd_vel search topic remains available).'
        )
        self.get_logger().info(
            f'Tracking mode: {self.tracking_mode}. '
            f'Depth input is {"enabled" if self.use_depth_input else "disabled"}. '
            f'Input QoS reliability: {self.input_reliability}. '
            f'Continuous RGB stamp age limit: '
            f'{"disabled" if self.continuous_rgb_stamp_max_age <= 0.0 else f"{self.continuous_rgb_stamp_max_age:.2f}s"}.'
        )

        self.current_prompt = None
        self.target_found = False
        self.goal_locked = False
        self.tracking_enabled = False
        self.total_seg_time = 0.0
        self.segmentations = 0
        self.last_tracking_log_time = 0.0
        self.last_logged_center = None
        self.tracking_log_period = 1.0
        self.not_found_log_period = 2.0
        self.last_not_found_log_time = 0.0
        self.burst_active = False
        self.burst_frames_seen = 0
        self.burst_frames_with_detections = 0
        self.last_frame_received_time = None
        self.latest_burst_header = None
        self.best_candidate = None
        self.burst_complete_received = False
        self.burst_expected_frames = 0
        self.depth_buffer = deque(maxlen=30)
        self.latest_continuous_frame = None
        self.last_target_publish_time = 0.0
        self.prompt_started_monotonic = 0.0
        self.last_rgb_frame_received_monotonic = 0.0
        self.frames_received_for_prompt = 0
        self.last_depth_sync_warn_time = 0.0
        self.depth_sync_warn_period = 2.0
        self.last_continuous_age_warn_time = 0.0
        self.last_continuous_stamp_age_warn_time = 0.0
        self.last_no_frame_warn_time = 0.0
        self.target_publish_period = 0.0 if self.target_publish_rate <= 0.0 else 1.0 / self.target_publish_rate
        self.last_published_target_coords = None
        self.last_published_target_time = 0.0
        sensor_qos = QoSProfile(depth=1)
        sensor_qos.reliability = (
            ReliabilityPolicy.RELIABLE
            if self.input_reliability == 'reliable'
            else ReliabilityPolicy.BEST_EFFORT
        )
        sensor_qos.durability = DurabilityPolicy.VOLATILE
        tracking_output_qos = QoSProfile(depth=1)
        tracking_output_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        tracking_output_qos.durability = DurabilityPolicy.VOLATILE
        image_msg_type = CompressedImage if self.use_compressed_input else Image
        image_callback = self.compressed_image_callback if self.use_compressed_input else self.raw_image_callback
        self.image_sub = self.create_subscription(
            image_msg_type,
            '/image_in',
            image_callback,
            sensor_qos,
        )
        self.depth_sub = None
        if self.use_depth_input:
            self.depth_sub = self.create_subscription(
                Image,
                self.depth_topic,
                self.depth_callback,
                sensor_qos,
            )
        self.prompt_sub = self.create_subscription(String, '/target_prompt', self.prompt_callback, 1)
        latched_state_qos = QoSProfile(depth=1)
        latched_state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        latched_state_qos.reliability = ReliabilityPolicy.RELIABLE
        self.goal_locked_sub = self.create_subscription(
            Bool,
            self.goal_locked_topic,
            self.goal_locked_callback,
            latched_state_qos,
        )
        self.burst_complete_sub = self.create_subscription(
            UInt32,
            self.burst_complete_topic,
            self.burst_complete_callback,
            10,
        )

        self.search_cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_pub = self.create_publisher(Image, '/image_out', 1)
        self.pixel_pub = self.create_publisher(PointStamped, '/target_pixel', tracking_output_qos)
        self.mask_pub = self.create_publisher(Image, '/target_mask', tracking_output_qos)
        self.cv_runtime_pub = self.create_publisher(Float32, self.cv_runtime_topic, 10)
        latched_state_qos = QoSProfile(depth=1)
        latched_state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        latched_state_qos.reliability = ReliabilityPolicy.RELIABLE
        self.cv_model_pub = self.create_publisher(String, self.cv_model_topic, latched_state_qos)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self._publish_cv_model()

    def _resolve_model_mode(self, requested_mode, use_sam):
        if requested_mode in ('', 'auto'):
            return 'dino_mobilesam' if use_sam else 'clip'

        valid_modes = {'clip', 'dino_mobilesam', 'florence2', 'yoloe'}
        if requested_mode not in valid_modes:
            self.get_logger().warn(
                f'Unknown model_mode "{requested_mode}", falling back to '
                f'{"dino_mobilesam" if use_sam else "clip"}.'
            )
            return 'dino_mobilesam' if use_sam else 'clip'

        return requested_mode

    def _get_model_label(self, model_mode):
        labels = {
            'clip': 'CLIPSeg',
            'dino_mobilesam': 'GroundingDINO + MobileSAM',
            'florence2': 'Florence-2',
            'yoloe': 'YOLOE',
        }
        return labels.get(model_mode, model_mode)

    def prompt_callback(self, msg):
        if msg.data == self.current_prompt and self.tracking_enabled:
            self.get_logger().info(
                f'Ignoring duplicate prompt "{msg.data}" while the current tracking request is still active.'
            )
            return

        self._publish_cv_model()
        self.current_prompt = msg.data
        self.target_found = False
        self.goal_locked = False
        self.tracking_enabled = bool(self.current_prompt)
        self.total_seg_time = 0.0
        self.segmentations = 0
        self.last_tracking_log_time = 0.0
        self.last_logged_center = None
        self.last_not_found_log_time = 0.0
        self.last_target_publish_time = 0.0
        self.prompt_started_monotonic = time.monotonic()
        self.last_rgb_frame_received_monotonic = 0.0
        self.frames_received_for_prompt = 0
        self.last_no_frame_warn_time = 0.0
        self.last_continuous_stamp_age_warn_time = 0.0
        self.latest_continuous_frame = None
        self.depth_buffer.clear()
        self.last_published_target_coords = None
        self.last_published_target_time = 0.0
        self._reset_burst_state()
        self.get_logger().info(f'New prompt received: "{self.current_prompt}"')

    def _publish_cv_model(self):
        msg = String()
        msg.data = self.model_mode
        self.cv_model_pub.publish(msg)

    def goal_locked_callback(self, msg):
        was_locked = self.goal_locked
        self.goal_locked = bool(msg.data)
        self.tracking_enabled = bool(self.current_prompt) and not self.goal_locked
        if self.goal_locked and not was_locked:
            completed_prompt = self.current_prompt
            self.latest_continuous_frame = None
            self._reset_burst_state()
            self.current_prompt = None
            self.tracking_enabled = False
            self.target_found = False
            self.prompt_started_monotonic = 0.0
            self.last_rgb_frame_received_monotonic = 0.0
            self.frames_received_for_prompt = 0
            self.last_continuous_stamp_age_warn_time = 0.0
            self.depth_buffer.clear()
            self.last_published_target_coords = None
            self.last_published_target_time = 0.0
            self.get_logger().info('Goal lock received from Raspberry Pi bridge. Pausing RGB tracking until the next prompt.')
            if completed_prompt:
                self.get_logger().info(
                    f'Completed prompt "{completed_prompt}" and cleared tracker state. '
                    'Waiting for the next prompt.'
                )

    def burst_complete_callback(self, msg):
        if self.tracking_mode != 'burst':
            return

        if self.current_prompt is None or not self.tracking_enabled:
            return

        self.burst_complete_received = True
        self.burst_expected_frames = int(msg.data)
        self.get_logger().info(
            f'Received burst-complete signal for {self.burst_expected_frames} exported frame(s).'
        )
        self._try_finalize_burst()

    def timer_callback(self):
        if self.current_prompt is None or not self.tracking_enabled:
            if self.tracking_mode == 'burst':
                self._try_finalize_burst()
            return

        if self.tracking_mode == 'burst':
            self._try_finalize_burst()
        else:
            self._process_latest_continuous_frame()
        self._warn_if_no_rgb_frames_arrived()
        if not self.enable_search_rotation or self.target_found:
            return

        msg = Twist()
        msg.angular.z = self.search_angular_speed
        self.search_cmd_pub.publish(msg)

    def _should_log_tracking_update(self, center_coords):
        now = time.time()
        if self.last_logged_center is None:
            self.last_tracking_log_time = now
            self.last_logged_center = center_coords
            return True

        center_changed = (
            abs(center_coords[0] - self.last_logged_center[0]) >= 10
            or abs(center_coords[1] - self.last_logged_center[1]) >= 10
        )
        period_elapsed = now - self.last_tracking_log_time >= self.tracking_log_period

        if center_changed or period_elapsed:
            self.last_tracking_log_time = now
            self.last_logged_center = center_coords
            return True

        return False

    def raw_image_callback(self, msg):
        if self.current_prompt is None or not self.tracking_enabled:
            return

        self._note_rgb_frame_received(msg.header)
        if self.tracking_mode == 'burst':
            self._note_burst_frame_arrival(msg.header)
        if self.tracking_mode == 'continuous' and self._is_continuous_rgb_stamp_stale(msg.header):
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'RGB conversion failed: {exc}')
            return
        if self.tracking_mode == 'continuous':
            self._store_latest_continuous_frame(image, msg.header)
            return
        depth_frame = self._get_matching_depth(msg.header)
        self._process_image(image, msg.header, depth_frame)

    def compressed_image_callback(self, msg):
        if self.current_prompt is None or not self.tracking_enabled:
            return

        self._note_rgb_frame_received(msg.header)
        if self.tracking_mode == 'burst':
            self._note_burst_frame_arrival(msg.header)
        if self.tracking_mode == 'continuous' and self._is_continuous_rgb_stamp_stale(msg.header):
            return
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().error('Failed to decode compressed RGB frame')
            return
        if self.tracking_mode == 'continuous':
            self._store_latest_continuous_frame(image, msg.header)
            return
        depth_frame = self._get_matching_depth(msg.header)
        self._process_image(image, msg.header, depth_frame)

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as exc:
            self.get_logger().error(f'Depth conversion failed: {exc}')
            return

        depth_array = np.asarray(depth_image)
        if msg.encoding == '16UC1':
            depth_array = depth_array.astype(np.float32) / 1000.0
        elif msg.encoding == '32FC1':
            depth_array = depth_array.astype(np.float32)
        else:
            self.get_logger().warn(f'Unsupported depth encoding for continuous tracking: {msg.encoding}')
            return

        self.depth_buffer.append({
            'stamp_ns': self._stamp_to_ns(msg.header.stamp),
            'image': depth_array,
        })

    def _note_rgb_frame_received(self, header):
        now = time.monotonic()
        self.last_rgb_frame_received_monotonic = now
        self.frames_received_for_prompt += 1
        if self.frames_received_for_prompt == 1:
            self.get_logger().info(
                f'Received first RGB frame for prompt "{self.current_prompt}" '
                f'with stamp {header.stamp.sec}.{header.stamp.nanosec:09d}.'
            )

    def _warn_if_no_rgb_frames_arrived(self):
        if self.current_prompt is None or not self.tracking_enabled:
            return
        if self.frames_received_for_prompt > 0:
            return
        if self.prompt_started_monotonic <= 0.0:
            return

        now = time.monotonic()
        wait_s = now - self.prompt_started_monotonic
        if wait_s < 2.0:
            return
        if (now - self.last_no_frame_warn_time) < self.depth_sync_warn_period:
            return

        self.last_no_frame_warn_time = now
        self.get_logger().warn(
            f'No RGB frames have arrived on /image_in for {wait_s:.1f}s after prompt "{self.current_prompt}".'
        )

    def _process_image(self, image, header, depth_frame):
        if self.model_mode == 'dino_mobilesam':
            seg_img, center_coords, _unused_depth, segmentation_time = self.segmentor.segment(
                image,
                self.current_prompt,
                depth_frame,
                box_threshold=float(self.dino_box_threshold),
                min_mask_area=int(self.dino_mobilesam_min_mask_area),
            )
        elif self.model_mode == 'florence2':
            seg_img, center_coords, segmentation_time, _unused_depth = self.segmentor.segment(
                image,
                self.current_prompt,
                depth_frame,
                min_mask_area=int(self.florence2_min_mask_area),
            )
        elif self.model_mode == 'yoloe':
            seg_img, center_coords, segmentation_time, _unused_depth = self.segmentor.segment(
                image,
                self.current_prompt,
                depth_frame,
                conf=float(self.yoloe_conf_threshold),
                min_mask_area=int(self.yoloe_min_mask_area),
            )
        else:
            seg_img, center_coords, segmentation_time, _unused_depth = self.segmentor.segment(
                image,
                self.current_prompt,
                depth_frame,
                threshold=float(self.clip_threshold),
                min_mask_area=int(self.clip_min_mask_area),
            )

        self.total_seg_time += segmentation_time
        self.segmentations += 1
        self.cv_runtime_pub.publish(Float32(data=float(self.total_seg_time / self.segmentations)))

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(seg_img, encoding='bgr8'))

        if center_coords is None:
            if self.tracking_mode == 'continuous':
                self.target_found = False
            if not self.target_found and time.time() - self.last_not_found_log_time >= self.not_found_log_period:
                self.get_logger().warn('Target object not found in RGB frame')
                self.last_not_found_log_time = time.time()
            return

        self.burst_frames_with_detections += 1

        score = getattr(self.segmentor, 'last_detection_score', None)
        mask = getattr(self.segmentor, 'last_mask', None)
        self._update_best_candidate(center_coords, header, score, image.shape, mask)

        if self.tracking_mode == 'continuous':
            self.target_found = True
            published_coords, published_depth = self._publish_continuous_target(
                center_coords, header, mask, depth_frame
            )
            if published_coords is not None and self._should_log_tracking_update(published_coords):
                avg_seg = self.total_seg_time / self.segmentations
                if published_depth is not None:
                    self.get_logger().info(
                        f'{self.model_label} target point: center=({center_coords[0]}, {center_coords[1]}), '
                        f'nearest=({published_coords[0]}, {published_coords[1]}), depth={published_depth:.2f}m, '
                        f'avg_seg={avg_seg:.3f}s'
                    )
                else:
                    self.get_logger().info(
                        f'{self.model_label} target pixel: center=({center_coords[0]}, {center_coords[1]}), '
                        f'avg_seg={avg_seg:.3f}s'
                    )
            return

        if self._should_log_tracking_update(center_coords):
            avg_seg = self.total_seg_time / self.segmentations
            self.get_logger().info(
                f'{self.model_label} target pixel: center=({center_coords[0]}, {center_coords[1]}), '
                f'avg_seg={avg_seg:.3f}s'
            )

    def _store_latest_continuous_frame(self, image, header):
        self.latest_continuous_frame = {
            'image': image,
            'header': header,
            'stored_monotonic': time.monotonic(),
        }

    def _process_latest_continuous_frame(self):
        if self.latest_continuous_frame is None:
            return

        frame = self.latest_continuous_frame
        if self._is_continuous_rgb_stamp_stale(frame['header'], stage='before inference'):
            self.latest_continuous_frame = None
            return

        frame_age = time.monotonic() - frame['stored_monotonic']
        if self.continuous_frame_max_age > 0.0 and frame_age > self.continuous_frame_max_age:
            self.latest_continuous_frame = None
            if (time.monotonic() - self.last_continuous_age_warn_time) >= self.depth_sync_warn_period:
                self.last_continuous_age_warn_time = time.monotonic()
                self.get_logger().warn(
                    f'Skipping stale continuous frame aged {frame_age:.3f}s '
                    f'(limit {self.continuous_frame_max_age:.3f}s).'
                )
            return

        depth_frame = self._get_matching_depth(frame['header'])
        if self.use_depth_input and depth_frame is None:
            return

        self.latest_continuous_frame = None
        self._process_image(frame['image'], frame['header'], depth_frame)

    def _note_burst_frame_arrival(self, header):
        now = time.monotonic()
        if not self.burst_active:
            self.burst_active = True
            self.burst_frames_seen = 0
            self.burst_frames_with_detections = 0
            self.best_candidate = None
            self.get_logger().info('Started processing a new RGB burst.')

        self.burst_frames_seen += 1
        self.last_frame_received_time = now
        self.latest_burst_header = header

    def _update_best_candidate(self, center_coords, header, score, image_shape, mask):
        image_height, image_width = image_shape[:2]
        center_distance = float(
            np.hypot(center_coords[0] - image_width / 2.0, center_coords[1] - image_height / 2.0)
        )
        normalized_score = float(score) if score is not None else 0.0
        candidate = {
            'header': header,
            'center_coords': center_coords,
            'score': normalized_score,
            'center_distance': center_distance,
            'mask': None if mask is None else np.asarray(mask, dtype=np.uint8).copy(),
        }

        if self.best_candidate is None:
            self.best_candidate = candidate
            return

        if normalized_score > self.best_candidate['score']:
            self.best_candidate = candidate
            return

        if (
            abs(normalized_score - self.best_candidate['score']) <= 1e-6
            and center_distance < self.best_candidate['center_distance']
        ):
            self.best_candidate = candidate

    def _try_finalize_burst(self):
        if self.tracking_mode != 'burst':
            return

        if not self.burst_active or self.last_frame_received_time is None:
            return

        now = time.monotonic()
        if self.burst_complete_received:
            if self.burst_frames_seen < self.burst_expected_frames:
                if (now - self.last_frame_received_time) < self.burst_quiet_period:
                    return
                missing_frames = max(0, self.burst_expected_frames - self.burst_frames_seen)
                self.get_logger().warn(
                    f'Burst-complete was received for {self.burst_expected_frames} frame(s), '
                    f'but only {self.burst_frames_seen} frame(s) were processed and no new RGB frame '
                    f'arrived for {self.burst_quiet_period:.1f}s. '
                    f'Finalizing with the available frames (missing {missing_frames}).'
                )
        else:
            if (now - self.last_frame_received_time) < self.burst_quiet_period:
                return
            self.get_logger().warn(
                f'Burst-complete signal was not received; falling back to quiet-period finalization '
                f'after processing {self.burst_frames_seen} frame(s).'
            )

        if self.best_candidate is None:
            self.get_logger().warn(
                f'RGB burst completed with no valid target detections '
                f'({self.burst_frames_seen} frames processed).'
            )
            self._reset_burst_state()
            return

        pixel = PointStamped()
        pixel.header = self.latest_burst_header or self.best_candidate['header']
        pixel.point.x = float(self.best_candidate['center_coords'][0])
        pixel.point.y = float(self.best_candidate['center_coords'][1])
        pixel.point.z = 0.0

        target_mask = self.best_candidate.get('mask')
        if target_mask is not None:
            mask_msg = self.bridge.cv2_to_imgmsg((target_mask * 255).astype(np.uint8), encoding='mono8')
            mask_msg.header = pixel.header
            self.mask_pub.publish(mask_msg)

        self.pixel_pub.publish(pixel)
        self.target_found = True
        self.get_logger().info(
            f'Selected burst candidate center=({self.best_candidate["center_coords"][0]}, '
            f'{self.best_candidate["center_coords"][1]}) '
            f'score={self.best_candidate["score"]:.2f} '
            f'from {self.burst_frames_with_detections}/{self.burst_frames_seen} detection frame(s).'
        )
        self._reset_burst_state()

    def _publish_continuous_target(self, center_coords, header, mask, depth_frame):
        now = time.monotonic()
        if self.target_publish_period > 0.0 and (now - self.last_target_publish_time) < self.target_publish_period:
            return None, None

        target_u = int(center_coords[0])
        target_v = int(center_coords[1])
        target_depth = None
        if self.use_depth_input:
            reference_coords = self._get_continuous_target_anchor(center_coords)
            nearest_target = self._select_nearest_mask_target(reference_coords, mask, depth_frame)
            if nearest_target is None:
                self._warn_depth_sync(
                    'Continuous target update skipped: no valid depth-backed target point was found in the mask.'
                )
                return None, None
            target_u, target_v, target_depth = nearest_target

        pixel = PointStamped()
        pixel.header = header
        pixel.point.x = float(target_u)
        pixel.point.y = float(target_v)
        pixel.point.z = 0.0 if target_depth is None else float(target_depth)

        if mask is not None and self.publish_mask_in_continuous:
            mask_msg = self.bridge.cv2_to_imgmsg((np.asarray(mask, dtype=np.uint8) * 255).astype(np.uint8), encoding='mono8')
            mask_msg.header = header
            self.mask_pub.publish(mask_msg)

        self.pixel_pub.publish(pixel)
        self.last_target_publish_time = now
        self.last_published_target_coords = (int(target_u), int(target_v))
        self.last_published_target_time = now
        return (target_u, target_v), target_depth

    def _get_continuous_target_anchor(self, center_coords):
        if self.last_published_target_coords is None:
            return center_coords

        if self.target_anchor_timeout_s > 0.0:
            age_s = time.monotonic() - self.last_published_target_time
            if age_s > self.target_anchor_timeout_s:
                return center_coords

        return self.last_published_target_coords

    def _select_nearest_mask_target(self, center_coords, mask, depth_frame):
        if mask is None:
            self._warn_depth_sync('Continuous target update skipped: segmentor did not provide a target mask.')
            return None
        if depth_frame is None:
            self._warn_depth_sync('Continuous target update skipped: no matched depth frame is available yet.')
            return None

        depth = np.asarray(depth_frame, dtype=np.float32)
        if depth.ndim != 2:
            self._warn_depth_sync('Continuous target update skipped: depth frame is not a single-channel image.')
            return None

        mask_array = np.asarray(mask, dtype=np.uint8)
        if mask_array.shape != depth.shape:
            mask_array = cv2.resize(
                mask_array,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_bool = mask_array > 0

        valid = (
            mask_bool
            & np.isfinite(depth)
            & (depth >= self.min_depth_m)
            & (depth <= self.max_depth_m)
        )
        if not np.any(valid):
            self._warn_depth_sync('Continuous target update skipped: target mask has no valid depth pixels.')
            return None

        valid_depths = depth[valid].astype(np.float32)
        percentile = min(100.0, max(0.0, float(self.nearest_depth_percentile)))
        nearest_depth = float(np.percentile(valid_depths, percentile))
        front_band = valid & (depth <= (nearest_depth + self.nearest_depth_band_m))
        front_band_count = int(np.count_nonzero(front_band))
        if front_band_count < self.nearest_depth_min_pixels:
            self._warn_depth_sync(
                'Continuous target update skipped: nearest-depth candidate set is too sparse '
                f'({front_band_count} < {self.nearest_depth_min_pixels}).'
            )
            return None
        candidate_pixels = np.argwhere(front_band if np.any(front_band) else valid)
        if candidate_pixels.size == 0:
            self._warn_depth_sync('Continuous target update skipped: nearest-depth candidate set is empty.')
            return None

        center_u = float(center_coords[0])
        center_v = float(center_coords[1])
        dv = candidate_pixels[:, 0].astype(np.float32) - center_v
        du = candidate_pixels[:, 1].astype(np.float32) - center_u
        best_idx = int(np.argmin(du * du + dv * dv))
        v = int(candidate_pixels[best_idx, 0])
        u = int(candidate_pixels[best_idx, 1])
        depth_m = float(depth[v, u])
        return u, v, depth_m

    def _get_matching_depth(self, header):
        if not self.use_depth_input:
            return None

        if not self.depth_buffer:
            self._warn_depth_sync('No depth frames are available yet for RGB/depth matching.')
            return None

        rgb_stamp_ns = self._stamp_to_ns(header.stamp)
        best_match = min(self.depth_buffer, key=lambda item: abs(item['stamp_ns'] - rgb_stamp_ns))
        time_delta_s = abs(best_match['stamp_ns'] - rgb_stamp_ns) / 1e9
        if time_delta_s > self.depth_match_tolerance:
            self._warn_depth_sync(
                f'Closest depth frame is too far from RGB stamp: {time_delta_s:.3f}s > '
                f'{self.depth_match_tolerance:.3f}s'
            )
            return None

        return best_match['image']

    def _warn_depth_sync(self, message):
        now = time.monotonic()
        if now - self.last_depth_sync_warn_time < self.depth_sync_warn_period:
            return

        self.last_depth_sync_warn_time = now
        self.get_logger().warn(message)

    def _is_continuous_rgb_stamp_stale(self, header, stage='on arrival'):
        if self.continuous_rgb_stamp_max_age <= 0.0:
            return False

        stamp_ns = self._stamp_to_ns(header.stamp)
        if stamp_ns <= 0:
            return False

        now_ns = self.get_clock().now().nanoseconds
        age_s = (now_ns - stamp_ns) / 1e9
        if age_s <= self.continuous_rgb_stamp_max_age:
            return False

        now = time.monotonic()
        if (now - self.last_continuous_stamp_age_warn_time) >= self.depth_sync_warn_period:
            self.last_continuous_stamp_age_warn_time = now
            self.get_logger().warn(
                f'Dropping stale continuous RGB frame {stage}: '
                f'stamp age {age_s:.3f}s exceeds limit {self.continuous_rgb_stamp_max_age:.3f}s.'
            )
        return True

    @staticmethod
    def _stamp_to_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _reset_burst_state(self):
        self.burst_active = False
        self.burst_frames_seen = 0
        self.burst_frames_with_detections = 0
        self.last_frame_received_time = None
        self.latest_burst_header = None
        self.best_candidate = None
        self.burst_complete_received = False
        self.burst_expected_frames = 0


def main(args=None):
    rclpy.init(args=args)
    node = RGBTrackerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
