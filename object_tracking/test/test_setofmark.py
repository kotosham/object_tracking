"""Unit tests for Set-of-Mark numbering + rendering (Phase 3.3). cv2 + numpy, no torch."""
import numpy as np
import pytest

from object_tracking.setofmark import Detection, assign_marks, render_setofmark


def _det(conf, cx=10, cy=10, bbox=(5, 5, 20, 20), label='bus'):
    return Detection(label=label, confidence=conf, cx=cx, cy=cy, bbox=bbox)


def test_assign_marks_numbers_best_first():
    marks = assign_marks([_det(0.4), _det(0.9), _det(0.7)])
    assert [m.mark_id for m in marks] == [1, 2, 3]
    assert [round(m.confidence, 2) for m in marks] == [0.9, 0.7, 0.4]   # mark 1 = best


def test_assign_marks_starts_at_one():
    marks = assign_marks([_det(0.5)])
    assert marks[0].mark_id == 1                                        # 1-based, not 0


def test_assign_marks_filters_below_threshold():
    marks = assign_marks([_det(0.2), _det(0.8)], conf_threshold=0.5)
    assert len(marks) == 1 and marks[0].confidence == pytest.approx(0.8)


def test_assign_marks_caps_at_max_marks():
    marks = assign_marks([_det(0.9), _det(0.8), _det(0.7), _det(0.6)], max_marks=2)
    assert [m.mark_id for m in marks] == [1, 2]
    assert [round(m.confidence, 1) for m in marks] == [0.9, 0.8]        # the two best


def test_assign_marks_does_not_mutate_inputs():
    raw = [_det(0.9), _det(0.5)]
    assign_marks(raw)
    assert all(d.mark_id == 0 for d in raw)                            # inputs untouched


def test_assign_marks_empty():
    assert assign_marks([]) == []


def test_render_setofmark_preserves_shape_and_draws():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    marks = assign_marks([_det(0.9, bbox=(10, 10, 40, 40)),
                          _det(0.6, bbox=(60, 60, 90, 90))])
    out = render_setofmark(img, marks)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert int(out.sum()) > 0                                          # something was drawn
    assert int(img.sum()) == 0                                         # original untouched (copy)


def test_render_setofmark_clamps_badge_for_corner_box():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    # box hugging the top-left corner: badge must stay inside the frame (no crash)
    out = render_setofmark(img, assign_marks([_det(0.9, bbox=(0, 0, 5, 5))]))
    assert out.shape == img.shape
