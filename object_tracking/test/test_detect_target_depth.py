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
    srv.depth_point_strategy = "nearest_mask"
    srv.nearest_depth_percentile = 0.0
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


def test_sample_depth_point_prefers_nearest_valid_point_inside_mask():
    srv = _server()
    depth = np.full((240, 424), np.nan, dtype=np.float32)
    depth[90, 265] = 4.0       # mask center, but farther away
    depth[60, 232] = 1.5       # nearest visible object surface
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:260, 300:500] = True

    x, y, z = srv._sample_depth_point(
        depth,
        400,
        180,
        rgb_shape=(480, 640, 3),
        bbox=(300, 100, 500, 260),
        mask=mask,
    )

    assert (x, y) == (350, 120)
    assert z == pytest.approx(1.5)


def test_hybrid_mode_routes_target_to_dino_and_detect_all_to_yoloe(monkeypatch):
    srv = DetectTargetServer.__new__(DetectTargetServer)
    srv.model_mode = "hybrid_dino_yoloe"
    loaded = []

    def fake_load_backend(_self, name):
        loaded.append(name)
        return object()

    monkeypatch.setattr(DetectTargetServer, "_load_backend", fake_load_backend)

    target, vocab = srv._load_segmentors()

    assert loaded == ["dino_mobilesam", "yoloe"]
    assert target is not None
    assert vocab is not None


def test_yoloe_mode_uses_one_backend_for_target_and_detect_all(monkeypatch):
    srv = DetectTargetServer.__new__(DetectTargetServer)
    srv.model_mode = "yoloe"
    sentinel = object()

    monkeypatch.setattr(DetectTargetServer, "_load_backend", lambda _self, name: sentinel)

    target, vocab = srv._load_segmentors()

    assert target is sentinel
    assert vocab is sentinel


def test_detector_uses_split_diploma_confidence_defaults():
    srv = DetectTargetServer.__new__(DetectTargetServer)
    srv.target_conf_default = 0.50
    srv.vocab_conf_default = 0.12

    assert srv._conf_for_query("drawer cabinet", 0.0) == pytest.approx(0.50)
    assert srv._conf_for_query("", 0.0) == pytest.approx(0.12)
    assert srv._conf_for_query("drawer cabinet", 0.35) == pytest.approx(0.35)
