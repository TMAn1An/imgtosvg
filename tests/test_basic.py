import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icon2svg import Options, trace  # noqa: E402


def line_icon():
    img = np.full((512, 512, 3), 255, np.uint8)
    cv2.rectangle(img, (80, 120), (430, 400), (0, 0, 0), 22, cv2.LINE_AA)
    cv2.line(img, (80, 220), (430, 220), (0, 0, 0), 22, cv2.LINE_AA)
    cv2.circle(img, (255, 300), 50, (0, 0, 0), 22, cv2.LINE_AA)
    return img


def glyph_png():
    img = np.zeros((256, 256, 4), np.uint8)
    cv2.circle(img, (90, 100), 50, (40, 40, 200, 255), -1, cv2.LINE_AA)
    cv2.fillPoly(img, [np.array([[140, 60], [230, 60], [230, 200], [140, 200]], np.int32)],
                 (40, 40, 200, 255), cv2.LINE_AA)
    return img


def test_line_icon_is_clean_strokes():
    svg, info = trace(line_icon(), Options())
    ET.fromstring(svg)
    assert info["mode"] == "stroke"
    assert info["anchors"] <= 14
    assert 'stroke-linecap="round"' in svg
    assert "a49" in svg or "a50" in svg  # the ring became a true circle


def test_glyph_becomes_outline():
    svg, info = trace(glyph_png(), Options())
    ET.fromstring(svg)
    assert info["mode"] == "outline"
    assert info["anchors"] <= 24


def test_size_option_scales_viewbox():
    svg, _ = trace(line_icon(), Options(size=24))
    assert 'viewBox="0 0 24 24"' in svg


def test_outline_mode_forced():
    svg, info = trace(line_icon(), Options(mode="outline"))
    assert info["mode"] == "outline"
    assert 'fill-rule="evenodd"' in svg
