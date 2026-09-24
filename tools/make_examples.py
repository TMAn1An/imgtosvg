"""Turn reference SVGs (e.g. Illustrator exports with hidden layers and nested
transforms) into small, clean style examples for the AI redraw.

Usage: python tools/make_examples.py INPUT_IMAGES_DIR REFERENCE_SVG_DIR OUT_DIR
Pairs are matched by file name (Icon_1.jpg <-> Icon_1.svg).
"""
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icon2svg import load_image  # noqa: E402
from icon2svg.ai import clean_reference_svg, score_svg  # noqa: E402
from icon2svg.tracer import to_ink  # noqa: E402

MIN_MATCH = 0.84  # a redraw that changed the drawing would teach the model the wrong thing

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main(img_dir, svg_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(svg_dir)):
        if not name.lower().endswith(".svg"):
            continue
        base = os.path.splitext(name)[0]
        img = next((os.path.join(img_dir, base + e) for e in EXTS
                    if os.path.exists(os.path.join(img_dir, base + e))), None)
        if img is None:
            continue
        with open(os.path.join(svg_dir, name), encoding="utf-8") as fh:
            clean = clean_reference_svg(fh.read())
        ink, _ = to_ink(load_image(img))
        sw = float(re.search(r'stroke-width="([\d.]+)"', clean).group(1)) if 'stroke-width="' in clean else 2
        match, _, _ = score_svg(clean, ink, sw)
        if match < MIN_MATCH:
            print(f"{base}: skipped (the SVG matches its image only {match:.0%}, it was redrawn differently)")
            continue
        with open(os.path.join(out_dir, base + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(clean)
        shutil.copy(img, os.path.join(out_dir, base + os.path.splitext(img)[1].lower()))
        n += 1
        print(f"{base}: ok (match {match:.0%})")
    print(f"{n} example pairs written to {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
