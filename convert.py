#!/usr/bin/env python3
"""icon2svg - convert JPG/PNG line icons into clean, low-anchor SVGs.

Examples
  python convert.py Input/                    # whole folder -> Output/
  python convert.py icon.png -o out/          # one file
  python convert.py Input/ --mode outline     # filled outlines instead of strokes
  python convert.py Input/ --size 64 --stroke-width 2
"""
import argparse
import os
import sys
import time

from icon2svg import Options, load_image, trace

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith(EXTS):
                    files.append(os.path.join(p, name))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"skip (not found): {p}")
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description="Raster icon -> clean SVG")
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("-o", "--out", help="output folder (default: <input folder>/Output or next to the file)")
    ap.add_argument("--mode", choices=["auto", "stroke", "outline"], default="outline",
                    help="stroke = centre-line strokes (editable line icons), outline = filled shapes")
    ap.add_argument("--tolerance", type=float, default=0.30,
                    help="curve fit tolerance in px @130px; higher = fewer anchors (default 0.30)")
    ap.add_argument("--stroke-width", type=float, default=0, help="force stroke width (output units)")
    ap.add_argument("--size", type=float, default=0, help="scale so the longest side equals SIZE (e.g. 24, 64, 512)")
    ap.add_argument("--color", default="auto", help='fill/stroke colour, e.g. "#000000" or "currentColor"')
    ap.add_argument("--no-circles", action="store_true", help="do not replace round shapes by true circles")
    ap.add_argument("--no-symmetry", action="store_true", help="do not make mirror-symmetric icons exactly symmetric")
    ap.add_argument("--no-primitives", action="store_true",
                    help="do not redraw regions as exact rectangles/ellipses/polygons (pure tracing)")
    ap.add_argument("--decimals", type=int, default=2)
    a = ap.parse_args(argv)

    files = collect(a.inputs)
    if not files:
        print("No images found.")
        return 1
    opt = Options(mode=a.mode, tolerance=a.tolerance, stroke_width=a.stroke_width, size=a.size,
                  color=a.color, detect_circles=not a.no_circles, decimals=a.decimals,
                  symmetry=not a.no_symmetry, extra={} if not a.no_primitives else {"primitives": False})
    ok = 0
    for f in files:
        if a.out:
            out_dir = a.out
        elif os.path.isdir(a.inputs[0]) and len(a.inputs) == 1:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(a.inputs[0])), "Output")
        else:
            out_dir = os.path.dirname(os.path.abspath(f))
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + ".svg")
        t = time.time()
        try:
            svg, info = trace(load_image(f), opt)
        except Exception as e:  # keep going with the rest of the batch
            print(f"FAILED {f}: {e}")
            continue
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(svg)
        ok += 1
        print(f"{os.path.basename(f):30s} -> {dst}  [{info['mode']}, {info['anchors']} anchors, "
              f"{time.time() - t:.2f}s]")
    print(f"\nDone: {ok}/{len(files)} converted.")
    return 0 if ok == len(files) else 2


if __name__ == "__main__":
    sys.exit(main())
