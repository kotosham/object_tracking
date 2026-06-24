"""Set-of-Mark candidate numbering + rendering for the DetectTarget service (Phase 3.3).

ROS-free and torch-free on purpose: pure geometry + cv2 drawing over numpy BGR
frames, so it unit-tests without a GPU or the detector backend. The DetectTarget
server (Phase 3.2) hands it the raw open-vocab detections; this module assigns
stable numbered ``mark_id``s (highest-confidence = mark 1) and draws the annotated
frame the VLM looks at -- each candidate boxed and tagged with its mark_id, so the
VLM can pick a target by number (``DRIVE_TO_VISIBLE mark_id``) instead of pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Sequence, Tuple

try:
    import cv2
    import numpy as np
    _HAVE_CV = True
except Exception:                       # pragma: no cover - cv2 absent in pure CI
    _HAVE_CV = False


@dataclass
class Detection:
    """One raw open-vocab detection (pre-numbering); pixel coords in the source image."""
    label: str
    confidence: float
    cx: int                              # detection center u (px)
    cy: int                              # detection center v (px)
    bbox: Tuple[int, int, int, int]      # x1, y1, x2, y2 (px)
    depth_m: float = 0.0                 # 0.0 if no depth
    mark_id: int = 0                     # 0 until assign_marks() numbers it


def assign_marks(detections: Sequence[Detection], conf_threshold: float = 0.0,
                 max_marks: int = 0) -> List[Detection]:
    """Filter by confidence, sort best-first, number 1..N. Returns NEW Detections
    (inputs untouched) with ``mark_id`` set -- stable: highest confidence == mark 1.
    ``max_marks`` (>0) caps how many candidates get a mark (the VLM option list)."""
    kept = [d for d in detections if d.confidence >= conf_threshold]
    kept.sort(key=lambda d: (d.confidence, -d.cx, -d.cy), reverse=True)
    if max_marks and max_marks > 0:
        kept = kept[:max_marks]
    return [replace(d, mark_id=i) for i, d in enumerate(kept, start=1)]


def _mark_radius(image_w: int) -> int:
    return max(10, image_w // 40)


def render_setofmark(image_bgr, marked: Sequence[Detection]):
    """Draw numbered Set-of-Mark annotations on a COPY of the BGR frame and return it.

    Each candidate gets a green bbox and a filled badge with its mark_id at the
    top-left corner, plus a small ``#id label conf`` caption. Pure cv2; raises if
    cv2/numpy are unavailable."""
    if not _HAVE_CV:
        raise RuntimeError('render_setofmark requires cv2/numpy')
    out = image_bgr.copy()
    h, w = out.shape[:2]
    r = _mark_radius(w)
    for d in marked:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # badge anchored just inside the top-left of the box, clamped to the frame
        bx = min(max(x1 + r, r), w - r)
        by = min(max(y1 + r, r), h - r)
        cv2.circle(out, (bx, by), r, (0, 0, 255), -1)
        cv2.circle(out, (bx, by), r, (255, 255, 255), 2)
        tag = str(d.mark_id)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(out, tag, (bx - tw // 2, by + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        caption = '#%d %s %.2f' % (d.mark_id, d.label, d.confidence)
        cv2.putText(out, caption, (max(0, x1), max(y1 - 8, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out
