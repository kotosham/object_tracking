import os
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class TargetOverlayCapture(Node):
    def __init__(self):
        super().__init__('target_overlay_capture')

        self.declare_parameter('image_topic', '/tracker/color/image_raw/compressed')
        self.declare_parameter('mask_topic', '/target_mask')
        self.declare_parameter('target_pixel_topic', '/target_pixel')
        self.declare_parameter('output_dir', 'experiment_logs/target_overlays')
        self.declare_parameter('stamp_tolerance_s', 0.20)
        self.declare_parameter('mask_alpha', 0.45)
        self.declare_parameter('point_radius_px', 8)
        self.declare_parameter('save_once', False)
        self.declare_parameter('min_save_period_s', 1.0)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.mask_topic = str(self.get_parameter('mask_topic').value)
        self.target_pixel_topic = str(self.get_parameter('target_pixel_topic').value)
        self.output_dir = os.path.expanduser(str(self.get_parameter('output_dir').value))
        self.stamp_tolerance_s = float(self.get_parameter('stamp_tolerance_s').value)
        self.mask_alpha = float(self.get_parameter('mask_alpha').value)
        self.point_radius_px = int(self.get_parameter('point_radius_px').value)
        self.save_once = bool(self.get_parameter('save_once').value)
        self.min_save_period_s = float(self.get_parameter('min_save_period_s').value)

        os.makedirs(self.output_dir, exist_ok=True)
        self.bridge = CvBridge()
        self.images = deque(maxlen=40)
        self.masks = deque(maxlen=40)
        self.saved_count = 0
        self.last_save_time = 0.0

        sensor_qos = QoSProfile(depth=10)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        self.create_subscription(CompressedImage, self.image_topic, self.image_callback, sensor_qos)
        self.create_subscription(Image, self.mask_topic, self.mask_callback, sensor_qos)
        self.create_subscription(PointStamped, self.target_pixel_topic, self.target_callback, sensor_qos)

        self.get_logger().info(
            f'Saving target overlays to {self.output_dir}; image={self.image_topic}, '
            f'mask={self.mask_topic}, target={self.target_pixel_topic}, '
            f'stamp_tolerance={self.stamp_tolerance_s:.2f}s, save_once={self.save_once}.'
        )

    @staticmethod
    def stamp_to_float(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def stamp_to_name(stamp):
        return f'{stamp.sec}_{stamp.nanosec:09d}'

    def image_callback(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn('Failed to decode compressed RGB image.')
            return
        self.images.append({
            'stamp': self.stamp_to_float(msg.header.stamp),
            'stamp_msg': msg.header.stamp,
            'image': image,
        })

    def mask_callback(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert target mask: {exc}')
            return
        self.masks.append({
            'stamp': self.stamp_to_float(msg.header.stamp),
            'stamp_msg': msg.header.stamp,
            'mask': mask,
        })

    def target_callback(self, msg):
        now = time.monotonic()
        if self.min_save_period_s > 0.0 and (now - self.last_save_time) < self.min_save_period_s:
            return

        target_stamp = self.stamp_to_float(msg.header.stamp)
        image_entry = self.find_closest(self.images, target_stamp)
        mask_entry = self.find_closest(self.masks, target_stamp)
        if image_entry is None or mask_entry is None:
            self.get_logger().warn('Cannot save overlay yet: waiting for matching RGB image and mask.')
            return

        image_delta = abs(image_entry['stamp'] - target_stamp)
        mask_delta = abs(mask_entry['stamp'] - target_stamp)
        if image_delta > self.stamp_tolerance_s or mask_delta > self.stamp_tolerance_s:
            self.get_logger().warn(
                f'Closest RGB/mask are too far from target stamp: '
                f'image_delta={image_delta:.3f}s, mask_delta={mask_delta:.3f}s.'
            )
            return

        overlay = self.make_overlay(image_entry['image'], mask_entry['mask'], msg)
        filename = os.path.join(
            self.output_dir,
            f'target_overlay_{self.stamp_to_name(msg.header.stamp)}_{self.saved_count:03d}.png',
        )
        if not cv2.imwrite(filename, overlay):
            self.get_logger().warn(f'Failed to write overlay image: {filename}')
            return

        self.saved_count += 1
        self.last_save_time = now
        self.get_logger().info(f'Saved target overlay: {filename}')

        if self.save_once:
            self.get_logger().info('save_once=true, shutting down after first saved overlay.')
            rclpy.shutdown()

    @staticmethod
    def find_closest(entries, target_stamp):
        if not entries:
            return None
        return min(entries, key=lambda entry: abs(entry['stamp'] - target_stamp))

    def make_overlay(self, image_bgr, mask, target_msg):
        image = image_bgr.copy()
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask_bool = mask > 0
        green = np.zeros_like(image, dtype=np.uint8)
        green[:, :] = (0, 255, 0)
        image[mask_bool] = cv2.addWeighted(
            image[mask_bool],
            1.0 - self.mask_alpha,
            green[mask_bool],
            self.mask_alpha,
            0.0,
        )

        u = int(round(target_msg.point.x))
        v = int(round(target_msg.point.y))
        radius = max(2, self.point_radius_px)
        cv2.circle(image, (u, v), radius + 2, (255, 255, 255), -1)
        cv2.circle(image, (u, v), radius, (0, 0, 255), -1)
        cv2.drawMarker(
            image,
            (u, v),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=radius * 4,
            thickness=2,
        )

        label = f'target=({u}, {v}), depth={target_msg.point.z:.2f}m'
        cv2.putText(
            image,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return image


def main(args=None):
    rclpy.init(args=args)
    node = TargetOverlayCapture()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
