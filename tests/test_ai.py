import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icon2svg import ai  # noqa: E402

HERE = os.path.dirname(__file__)


def icon():
    img = np.full((130, 130, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 40), (110, 110), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(img, (65, 22), 10, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def test_extract_svg_takes_last_real_svg():
    txt = 'Reply with <svg> only. Draft: <svg viewBox="0 0 1 1"></svg> final:\n```svg\n<svg viewBox="0 0 2 2"><circle/></svg>\n```'
    assert ai.extract_svg(txt) == '<svg viewBox="0 0 2 2"><circle/></svg>'


def test_sanitize_removes_unsafe_content():
    raw = ('<svg xmlns="http://www.w3.org/2000/svg" width="500" viewBox="0 0 130 130" onload="alert(1)">'
           '<script>alert(1)</script><g stroke="#000" onclick="x()"><circle cx="5" cy="5" r="3" style="x"/></g>'
           '<image href="http://evil"/></svg>')
    out = ai.sanitize_svg(raw, 130, 130)
    assert "script" not in out and "onload" not in out and "onclick" not in out and "image" not in out
    assert 'width="500"' not in out and 'viewBox="0 0 130 130"' in out and "<circle" in out


def test_score_detects_missing_parts():
    img = icon()
    from icon2svg.tracer import to_ink
    ink, _ = to_ink(img)
    full = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 130 130"><g fill="none" stroke="#000" '
            'stroke-width="3"><rect x="20" y="40" width="90" height="70"/><circle cx="65" cy="22" r="10"/></g></svg>')
    half = full.replace('<circle cx="65" cy="22" r="10"/>', "")
    s_full, d_full, _ = ai.score_svg(full, ink, 3)
    s_half, d_half, _ = ai.score_svg(half, ink, 3)
    assert s_full > 0.95 and s_half < s_full and d_half["recall"] < 0.9


def test_redraw_loop_repairs_a_bad_first_answer():
    prov = ai.make_provider("mock")
    svg, info = ai.redraw(icon(), prov, rounds=2, log=lambda *a: None)
    assert info["rounds"][0]["score"] < info["score"]
    assert info["score"] > 0.95 and prov.calls == 2


def test_clean_reference_drops_hidden_layers_and_flattens_transforms():
    raw = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
           '<g style="display:none"><circle cx="1" cy="1" r="1" stroke="#000"/></g>'
           '<g transform="translate(10 20)"><line x1="0" y1="0" x2="10" y2="0" stroke="#000" stroke-width="2"/></g>'
           '</svg>')
    out = ai.clean_reference_svg(raw)
    assert "<circle" not in out
    assert 'x1="10"' in out and 'y1="20"' in out and 'stroke-width="2"' in out


def test_count_anchors():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M0 0L5 5C6 6 7 7 8 8Z"/>'
           '<circle cx="1" cy="1" r="1"/><line x1="0" y1="0" x2="1" y2="1"/></svg>')
    assert ai.count_anchors(svg) == 3 + 4 + 2
