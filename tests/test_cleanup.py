import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icon2svg import Options, trace  # noqa: E402
from icon2svg.tidy import collapse_short  # noqa: E402


def test_tiny_rounded_corner_becomes_one_anchor():
    # square with a tiny 2-anchor rounding at one corner
    start = np.array([0.0, 0.0])
    segs = [("L", np.array([9.6, 0.0])), ("C", np.array([9.9, 0.0]), np.array([10.0, 0.1]), np.array([10.0, 0.4])),
            ("L", np.array([10.0, 10.0])), ("L", np.array([0.0, 10.0])), ("L", np.array([0.0, 0.0]))]
    s2, segs2 = collapse_short(start, segs, True, w=1.0, max_chord=1.0)
    assert len(segs2) == 4
    corner = segs2[0][-1]
    assert np.allclose(corner, [10.0, 0.0], atol=1e-6)


def pins_icon():
    img = np.full((400, 400, 3), 255, np.uint8)
    for cx in (80, 200, 320):  # three identical rounded boxes
        cv2.rectangle(img, (cx - 45, 140), (cx + 45, 260), (0, 0, 0), 14, cv2.LINE_AA)
    return img


def test_repeated_parts_are_identical():
    _, _, shapes = trace(pins_icon(), Options(mode="stroke"), return_shapes=True)
    boxes = [s for s in shapes if s[0] == "path"]
    assert len(boxes) == 3
    rel = []
    for kind, (start, segs) in boxes:
        pts = np.array([start] + [sg[-1] for sg in segs])
        rel.append(np.round(pts - pts.min(0), 2))
    rel.sort(key=lambda a: len(a))
    assert all(len(r) == len(rel[0]) for r in rel)
    assert all(np.allclose(np.sort(r, axis=0), np.sort(rel[0], axis=0), atol=0.05) for r in rel)
