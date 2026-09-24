#!/usr/bin/env python3
"""AI redraw of raster icons into designer-quality SVG.

Examples
  python ai_redraw.py Input/                                   # Gemini (key from config.json / GEMINI_API_KEY)
  python ai_redraw.py Input/ --key YOUR_GEMINI_KEY --save-key  # remember the key
  python ai_redraw.py icon.png --provider openrouter --model "qwen/qwen2.5-vl-72b-instruct:free"
  python ai_redraw.py Input/ --provider ollama --model qwen2.5vl:7b      # local, offline, free
  python ai_redraw.py --list-models                            # what your key can use
"""
import argparse
import json
import os
import sys
import time

from icon2svg import load_image
from icon2svg.ai import (ProviderError, key_for, load_config, load_examples, make_provider, redraw,
                         save_config)

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description="AI redraw: raster icon -> designer-quality SVG")
    ap.add_argument("inputs", nargs="*", help="image files and/or folders")
    ap.add_argument("-o", "--out", help="output folder (default: Output_ai next to the input)")
    ap.add_argument("--provider", default=None,
                    help="gemini (default) | openrouter | ollama | lmstudio | openai | anthropic")
    ap.add_argument("--model", default=None, help='model id, or "auto" (Gemini: newest Flash)')
    ap.add_argument("--key", default=None, help="API key (or env var / config.json)")
    ap.add_argument("--save-key", action="store_true", help="store --key in config.json for next time")
    ap.add_argument("--base-url", default=None, help="custom endpoint for OpenAI-compatible providers")
    ap.add_argument("--rounds", type=int, default=None, help="correction rounds after the first drawing (default 2)")
    ap.add_argument("--examples", default=None, help="folder with style examples (NAME.png + NAME.svg)")
    ap.add_argument("--n-examples", type=int, default=2, help="how many examples to show the model")
    ap.add_argument("--stroke-width", type=float, default=0, help="force the stroke width")
    ap.add_argument("--no-trace-hint", action="store_true", help="do not give the model the automatic trace")
    ap.add_argument("--list-models", action="store_true", help="list the models your key can use and exit")
    a = ap.parse_args(argv)

    cfg = load_config()
    provider_name = a.provider or cfg.get("provider") or "gemini"
    model = a.model or cfg.get("model", {}).get(provider_name) or "auto"
    key = a.key or key_for(provider_name, cfg)
    if a.save_key and a.key:
        cfg.setdefault("keys", {})[provider_name] = a.key
        cfg["provider"] = provider_name
        save_config(cfg)
        print("key saved to config.json")
    try:
        prov = make_provider(provider_name, key, model, a.base_url or cfg.get("base_url", {}).get(provider_name))
        if a.list_models:
            for m in prov.list_models():
                print(m)
            return 0
    except ProviderError as e:
        print(f"ERROR: {e}")
        return 1
    files = []
    for p in a.inputs:
        if os.path.isdir(p):
            files += [os.path.join(p, n) for n in sorted(os.listdir(p)) if n.lower().endswith(EXTS)]
        elif os.path.isfile(p):
            files.append(p)
    if not files:
        ap.print_help()
        return 1
    ex_dir = a.examples or os.path.join(HERE, "examples")
    examples = load_examples(ex_dir, a.n_examples)
    rounds = a.rounds if a.rounds is not None else int(cfg.get("rounds", 2))
    print(f"provider: {provider_name}, model: {prov.resolve_model()}, examples: {min(len(examples), a.n_examples)}, rounds: {rounds}")
    ok = 0
    for f in files:
        out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(f)) if not os.path.isdir(a.inputs[0])
                                        else os.path.dirname(os.path.abspath(a.inputs[0])), "Output_ai")
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + ".svg")
        print(f"\n{os.path.basename(f)}")
        t = time.time()
        try:
            svg, info = redraw(load_image(f), prov, rounds=rounds, examples=examples,
                               stroke_width=a.stroke_width or None, use_trace_hint=not a.no_trace_hint,
                               n_examples=a.n_examples)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(svg)
        ok += 1
        print(f"  -> {dst}  [match {info['score']:.1%}, {info['anchors']} anchors, {time.time() - t:.0f}s]")
    print(f"\nDone: {ok}/{len(files)}")
    return 0 if ok == len(files) else 2


if __name__ == "__main__":
    sys.exit(main())
