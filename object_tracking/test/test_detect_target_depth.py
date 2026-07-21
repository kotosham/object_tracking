"""Depth sampling tests for DetectTargetServer helpers; no ROS node is started."""
import math

import numpy as np
import pytest

pytest.importorskip("rclpy")

from object_tracking.detect_target_server import DetectTargetServer


def _server():
    srv = DetectTargetServer.__new__(DetectTargetServer)
    srv.min_depth_m = 0.1
    srv.max_depth_m = 8.0
    srv.depth_window = 0
    return srv


def test_sample_depth_scales_rgb_pixel_to_lower_depth_resolution():
    srv = _server()
    depth = np.full((240, 424), np.nan, dtype=np.float32)
    depth[74, 316] = 2.5

    assert srv._sample_depth(depth, 477, 148, rgb_shape=(480, 640, 3)) == pytest.approx(2.5)


def test_sample_depth_returns_nan_when_depth_is_unknown():
    srv = _server()
    depth = np.full((240, 424), np.nan, dtype=np.float32)

    assert math.isnan(srv._sample_depth(depth, 477, 148, rgb_shape=(480, 640, 3)))


def test_sample_depth_falls_back_to_scaled_bbox_when_center_is_empty():
    srv = _server()
    depth = np.full((240, 424), np.nan, dtype=np.float32)
    depth[30:90, 280:340] = 3.0

    assert srv._sample_depth(
        depth,
        477,
        148,
        rgb_shape=(480, 640, 3),
        bbox=(420, 60, 520, 190),
    ) == pytest.approx(3.0)
