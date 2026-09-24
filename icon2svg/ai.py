"""AI redraw: a vision language model redraws the icon as a designer would,
and a render -> compare -> correct loop keeps it faithful to the input.

Providers (all over plain HTTPS, no SDKs needed):
  gemini     Google Gemini API (has a free tier for Flash models)
  openai     any OpenAI-compatible /chat/completions endpoint:
             OpenRouter (incl. free models), Ollama / LM Studio (local, free), ...
  anthropic  Claude (Anthropic API)
  mock       offline stand-in used by the tests
"""
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import cv2
import numpy as np

SVG_NS = "http://www.w3.org/2000/svg"


# =============================================================== providers ==
class ProviderError(RuntimeError):
    pass


def _http_json(url, payload=None, headers=None, timeout=180, method=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        err = ProviderError(f"HTTP {e.code}: {body[:600]}")
        err.status = e.code
        err.body = body
        raise err
    except urllib.error.URLError as e:
        raise ProviderError(f"Network error: {e.reason}")


def _retry(fn, log, tries=5):
    """Retry on rate limits / overload with backoff (free tiers are strict)."""
    delay = 8.0
    for k in range(tries):
        try:
            return fn()
        except ProviderError as e:
            status = getattr(e, "status", None)
            if status not in (429, 500, 502, 503, 504) or k == tries - 1:
                raise
            m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', getattr(e, "body", "") or "")
            wait = float(m.group(1)) + 1 if m else delay
            log(f"  rate limited / busy ({status}), waiting {wait:.0f}s ...")
            time.sleep(wait)
            delay = min(delay * 2, 90)


class GeminiProvider:
    """Google Gemini. Uses the official google-genai SDK when installed (the
    same call that works in the user's icon_tool), else plain REST.

    Several keys (one per line / comma separated) are rotated: when one hits
    its free-tier limit (429) the next key is used."""
    name = "gemini"
    base = "https://generativelanguage.googleapis.com/v1beta"
    # most free-tier quota and cheapest; proven to work in the user's own tool
    DEFAULT_MODEL = "gemini-flash-lite-latest"

    def __init__(self, api_key, model="auto", min_interval=4.0):
        keys = [k.strip() for k in re.split(r"[\s,;]+", api_key or "") if k.strip()]
        if not keys:
            raise ProviderError("Gemini API key missing (get one free at https://aistudio.google.com/apikey)")
        self.keys = keys
        self.ki = 0
        self.key = keys[0]
        self.model = model
        self.min_interval = min_interval
        self._last = 0.0
        try:
            from google import genai  # noqa: F401
            self.sdk = True
        except ImportError:
            self.sdk = False

    def _next_key(self, log):
        if len(self.keys) < 2:
            return False
        self.ki = (self.ki + 1) % len(self.keys)
        self.key = self.keys[self.ki]
        log(f"  switching to API key #{self.ki + 1}")
        return True

    def list_models(self):
        out = _http_json(f"{self.base}/models?pageSize=200", headers={"x-goog-api-key": self.key})
        names = []
        for m in out.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                names.append(m["name"].split("/", 1)[-1])
        return names

    def resolve_model(self):
        if not self.model or self.model == "auto":
            self.model = self.DEFAULT_MODEL
        return self.model

    def _call_sdk(self, model, prompt_parts):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.key)
        parts = []
        for p in prompt_parts:
            if "text" in p:
                parts.append(types.Part.from_text(text=p["text"]))
            else:
                parts.append(types.Part.from_bytes(data=p["image"], mime_type="image/png"))
        try:
            resp = client.models.generate_content(model=model, contents=parts)
        except Exception as e:  # map SDK errors onto ours (429 -> retry / next key)
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            err = ProviderError(f"Gemini error {code or ''}: {e}")
            err.status = code if isinstance(code, int) else (429 if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) else None)
            err.body = str(e)
            raise err
        return resp.text or ""

    def _call_rest(self, model, prompt_parts):
        gparts = []
        for p in prompt_parts:
            if "text" in p:
                gparts.append({"text": p["text"]})
            else:
                gparts.append({"inline_data": {"mime_type": "image/png",
                                               "data": base64.b64encode(p["image"]).decode()}})
        body = {"contents": [{"role": "user", "parts": gparts}]}
        out = _http_json(f"{self.base}/models/{model}:generateContent", body, {"x-goog-api-key": self.key})
        try:
            cparts = out["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            raise ProviderError(f"Empty answer from Gemini: {json.dumps(out)[:400]}")
        return "".join(p.get("text", "") for p in cparts if not p.get("thought"))

    def generate(self, system, parts, log=print):
        model = self.resolve_model()
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        # like the user's working tool: the instructions go in as the first text part
        prompt_parts = [{"text": system}] + list(parts)
        call = self._call_sdk if self.sdk else self._call_rest

        def attempt():
            tried = 0
            while True:
                try:
                    return call(model, prompt_parts)
                except ProviderError as e:
                    if getattr(e, "status", None) == 429 and tried < len(self.keys) - 1 and self._next_key(log):
                        tried += 1
                        continue
                    raise
        out = _retry(attempt, log)
        self._last = time.time()
        return out


class OpenAICompatProvider:
    """OpenRouter, Ollama (http://localhost:11434/v1), LM Studio, Groq, OpenAI ..."""
    name = "openai"

    def __init__(self, api_key, model, base_url="https://openrouter.ai/api/v1", min_interval=0.0):
        if not model or model == "auto":
            raise ProviderError("Please set a model name for the OpenAI-compatible provider")
        self.key = api_key or ""
        self.model = model
        self.base = base_url.rstrip("/")
        self.min_interval = min_interval
        self._last = 0.0

    def list_models(self):
        h = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        out = _http_json(f"{self.base}/models", headers=h)
        return [m.get("id") for m in out.get("data", [])]

    def resolve_model(self):
        return self.model

    def generate(self, system, parts, log=print):
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        content = []
        for p in parts:
            if "text" in p:
                content.append({"type": "text", "text": p["text"]})
            else:
                url = "data:image/png;base64," + base64.b64encode(p["image"]).decode()
                content.append({"type": "image_url", "image_url": {"url": url}})
        body = {"model": self.model, "temperature": 0.4, "max_tokens": 16384,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        h = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        h["HTTP-Referer"] = "https://github.com/icon2svg"
        h["X-Title"] = "icon2svg"
        out = _retry(lambda: _http_json(f"{self.base}/chat/completions", body, h, timeout=600), log)
        self._last = time.time()
        try:
            msg = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ProviderError(f"Empty answer: {json.dumps(out)[:400]}")
        if isinstance(msg, list):
            msg = "".join(x.get("text", "") for x in msg if isinstance(x, dict))
        return msg or ""


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key, model="auto"):
        if not api_key:
            raise ProviderError("Anthropic API key missing")
        self.key = api_key
        self.model = "claude-sonnet-5" if model in (None, "", "auto") else model

    def list_models(self):
        out = _http_json("https://api.anthropic.com/v1/models?limit=100",
                         headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        return [m.get("id") for m in out.get("data", [])]

    def resolve_model(self):
        return self.model

    def generate(self, system, parts, log=print):
        content = []
        for p in parts:
            if "text" in p:
                content.append({"type": "text", "text": p["text"]})
            else:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                            "data": base64.b64encode(p["image"]).decode()}})
        body = {"model": self.model, "max_tokens": 16000, "system": system,
                "messages": [{"role": "user", "content": content}]}
        h = {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        out = _retry(lambda: _http_json("https://api.anthropic.com/v1/messages", body, h, timeout=600), log)
        return "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")


class MockProvider:
    """Offline stand-in: answers with the local tracer's drawing (optionally a
    deliberately broken one first) so the whole loop can be tested."""
    name = "mock"

    def __init__(self, *a, **k):
        self.model = "mock"
        self.calls = 0
        self.first_bad = k.get("first_bad", True)

    def resolve_model(self):
        return "mock"

    def list_models(self):
        return ["mock"]

    def generate(self, system, parts, log=print):
        time.sleep(float(os.environ.get("MOCK_DELAY", 0)))
        if any("text" in p and "Automatic trace" in p["text"] for p in parts):
            self.calls = 0  # a new icon starts
            self.__dict__.pop("_full", None)
        self.calls += 1
        texts = [p["text"] for p in parts if "text" in p and "<svg " in p["text"]]
        svg = extract_svg(texts[-1]) if texts else None
        if self.calls > 1 and hasattr(self, "_full"):
            svg = self._full  # "fixes" its drawing after seeing the comparison
        svg = svg or '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"></svg>'
        self._full = svg
        if self.first_bad and self.calls == 1:
            # drop half of every path's sub-paths: the loop must notice and fix it
            def half(m):
                subs = re.split(r"(?=[Mm])", m.group(1))
                subs = [x for x in subs if x.strip()]
                return 'd="' + "".join(subs[: max(1, len(subs) // 2)]) + '"'
            svg = re.sub(r'd="([^"]*)"', half, svg)
        return "Here you go:\n```svg\n" + svg + "\n```"


def make_provider(name, api_key=None, model="auto", base_url=None, **kw):
    name = (name or "gemini").lower()
    if name == "gemini":
        return GeminiProvider(api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                              model, **kw)
    if name in ("openai", "openrouter", "ollama", "lmstudio"):
        defaults = {"openrouter": "https://openrouter.ai/api/v1", "openai": "https://openrouter.ai/api/v1",
                    "ollama": "http://localhost:11434/v1", "lmstudio": "http://localhost:1234/v1"}
        key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return OpenAICompatProvider(key, model, base_url or defaults[name], **kw)
    if name in ("anthropic", "claude"):
        return AnthropicProvider(api_key or os.environ.get("ANTHROPIC_API_KEY"), model)
    if name == "mock":
        return MockProvider(**kw)
    raise ProviderError(f"Unknown provider: {name}")


# ================================================================== SVG I/O ==
ALLOWED = {
    "svg": {"viewBox", "xmlns"},
    "g": {"fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "stroke-miterlimit"},
    "path": {"d", "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "fill-rule"},
    "circle": {"cx", "cy", "r", "fill", "stroke", "stroke-width"},
    "ellipse": {"cx", "cy", "rx", "ry", "fill", "stroke", "stroke-width"},
    "rect": {"x", "y", "width", "height", "rx", "ry", "fill", "stroke", "stroke-width", "stroke-linejoin"},
    "line": {"x1", "y1", "x2", "y2", "stroke", "stroke-width", "stroke-linecap"},
    "polyline": {"points", "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin"},
    "polygon": {"points", "fill", "stroke", "stroke-width", "stroke-linejoin"},
}


def extract_svg(text):
    """The last complete <svg ...>...</svg> in a model answer."""
    found = re.findall(r"<svg\s[^>]*>.*?</svg>", text or "", re.S | re.I)
    return found[-1] if found else None


def _local(tag):
    return tag.split("}", 1)[-1]


def sanitize_svg(svg_text, width, height, stroke_color="#000000"):
    """Keep only simple drawing elements and safe attributes; enforce the
    canvas (viewBox only, no width/height) and the stroke style."""
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as e:
        raise ValueError(f"invalid SVG: {e}")
    if _local(root.tag) != "svg":
        raise ValueError("root is not <svg>")
    ET.register_namespace("", SVG_NS)

    def clean(el):
        tag = _local(el.tag)
        new = ET.Element(tag)
        for k, v in el.attrib.items():
            k = _local(k)
            if k in ALLOWED.get(tag, ()) and "url(" not in v and "javascript" not in v.lower():
                new.set(k, v)
        for ch in el:
            if _local(ch.tag) in ALLOWED and _local(ch.tag) != "svg":
                new.append(clean(ch))
        return new
    out = ET.Element("svg")
    out.set("xmlns", SVG_NS)
    out.set("viewBox", root.get("viewBox") or f"0 0 {width} {height}")
    for ch in root:
        if _local(ch.tag) in ALLOWED and _local(ch.tag) != "svg":
            out.append(clean(ch))
    if len(out) == 0:
        raise ValueError("SVG has no shapes")
    xml = ET.tostring(out, encoding="unicode")
    xml = xml.replace(' xmlns:ns0="http://www.w3.org/2000/svg"', "").replace("ns0:", "")
    xml = xml.replace("><", ">\n<")
    return xml + "\n"


def render_svg(svg_text, width, height, bg="#ffffff"):
    """Rasterise with resvg (no system libraries needed). Returns gray float [0,1]."""
    import resvg_py
    png = resvg_py.svg_to_bytes(svg_string=svg_text, width=int(width), height=int(height), background=bg)
    img = cv2.imdecode(np.frombuffer(bytes(png), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("render failed")
    return img.astype(np.float32) / 255.0


def count_anchors(svg_text):
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return 0
    n = 0
    arity = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}
    for el in root.iter():
        tag = _local(el.tag)
        if tag == "path":
            toks = re.findall(r"[a-zA-Z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", el.get("d", ""))
            i, cmd = 0, None
            while i < len(toks):
                if toks[i].isalpha():
                    cmd = toks[i]
                    i += 1
                    if cmd in "Zz":
                        continue
                if cmd is None or cmd.upper() not in arity:
                    i += 1
                    continue
                i += arity[cmd.upper()]
                n += 1
                if cmd in "Mm":
                    cmd = "l" if cmd == "m" else "L"
        elif tag in ("circle", "ellipse"):
            n += 4
        elif tag == "rect":
            n += 8 if float(el.get("rx", 0) or 0) > 0 else 4
        elif tag == "line":
            n += 2
        elif tag in ("polyline", "polygon"):
            n += len(re.findall(r"-?\d*\.?\d+", el.get("points", ""))) // 2
    return n


def clean_reference_svg(svg_text, decimals=2):
    """Flatten an exported SVG (hidden layers, nested transforms, CSS styles)
    into a compact example: visible shapes only, absolute coordinates."""
    from svgelements import SVG, Circle, Ellipse, Path, Polygon, Polyline, Rect, Shape, SimpleLine
    doc = SVG.parse(io.StringIO(svg_text), reify=True)
    vb = doc.viewbox
    W, H = (vb.width, vb.height) if vb is not None else (doc.width, doc.height)
    f = lambda v: (f"{v:.{decimals}f}".rstrip("0").rstrip(".") or "0").replace("-0", "0") \
        if abs(v) >= 10 ** -decimals else "0"
    items, widths = [], []
    for e in doc.elements():
        if not isinstance(e, Shape):
            continue
        sw = e.stroke_width if e.stroke is not None and e.stroke.value is not None else 0
        if sw:
            widths.append(float(sw))
        if isinstance(e, (Circle, Ellipse)) and not isinstance(e, Path):
            rx, ry = float(e.rx), float(e.ry)
            if abs(rx - ry) < 1e-3:
                items.append(f'<circle cx="{f(e.cx)}" cy="{f(e.cy)}" r="{f(rx)}"/>')
            else:
                items.append(f'<ellipse cx="{f(e.cx)}" cy="{f(e.cy)}" rx="{f(rx)}" ry="{f(ry)}"/>')
        elif isinstance(e, Rect):
            rx = f' rx="{f(e.rx)}"' if e.rx else ""
            items.append(f'<rect x="{f(e.x)}" y="{f(e.y)}" width="{f(e.width)}" height="{f(e.height)}"{rx}/>')
        elif isinstance(e, SimpleLine):
            items.append(f'<line x1="{f(e.x1)}" y1="{f(e.y1)}" x2="{f(e.x2)}" y2="{f(e.y2)}"/>')
        elif isinstance(e, (Polyline, Polygon)):
            pts = " ".join(f"{f(p.x)},{f(p.y)}" for p in e.points)
            items.append(f'<{"polygon" if isinstance(e, Polygon) else "polyline"} points="{pts}"/>')
        else:
            d = Path(e).d()
            d = re.sub(r"-?\d+\.\d+(?:e-?\d+)?", lambda m: f(float(m.group(0))), d)
            items.append(f'<path d="{d}"/>')
    sw = float(np.median(widths)) if widths else 2
    body = "\n  ".join(items)
    return (f'<svg xmlns="{SVG_NS}" viewBox="0 0 {f(W)} {f(H)}">\n'
            f'<g fill="none" stroke="#000000" stroke-width="{f(sw)}" stroke-linecap="round" '
            f'stroke-linejoin="round">\n  {body}\n</g>\n</svg>\n')


# =============================================================== scoring ==
def score_svg(svg_text, ink, stroke_w, zoom=4):
    """Compare a candidate with the input icon. Returns (score 0..1, details,
    diff image BGR). Tolerant to small offsets (a redraw is never pixel-exact)."""
    h, w = ink.shape
    W, H = w * zoom, h * zoom
    cand = 1 - render_svg(svg_text, W, H)
    ref = cv2.resize(ink, (W, H), interpolation=cv2.INTER_CUBIC)
    a, b = ref > 0.5, cand > 0.5
    t = max(1, int(round(0.7 * stroke_w * zoom)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
    da = cv2.dilate(a.astype(np.uint8), k) > 0
    db = cv2.dilate(b.astype(np.uint8), k) > 0
    recall = (a & db).sum() / max(a.sum(), 1)       # how much of the icon is drawn
    precision = (b & da).sum() / max(b.sum(), 1)    # how much of the drawing belongs
    f1 = 2 * recall * precision / max(recall + precision, 1e-9)
    missing = a & ~db
    extra = b & ~da
    diff = np.full((H, W, 3), 255, np.uint8)
    diff[a] = (190, 190, 190)
    diff[b & ~extra] = (40, 40, 40)
    diff[missing] = (40, 40, 230)   # red  = in the original, missing in the SVG
    diff[extra] = (230, 120, 30)    # blue = in the SVG, not in the original
    return float(f1), {"recall": float(recall), "precision": float(precision)}, diff


def png_bytes(img_bgr_or_gray):
    ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    return buf.tobytes()


# ================================================================ prompts ==
SYSTEM = """You are a senior icon designer. You redraw raster line icons as clean, professional SVG line icons, built the way a designer builds them in Adobe Illustrator or Figma.

Hard rules for your answer:
- Reply with ONE complete <svg> element and nothing else (no markdown, no explanation).
- Root: <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"> with NO width/height attributes.
- Put every shape inside one <g fill="none" stroke="#000000" stroke-width="{SW}" stroke-linecap="round" stroke-linejoin="round">. No fills, no transforms, no style/class attributes, no text, no gradients.
- Build the icon from simple geometric primitives: <circle>, <ellipse>, <rect> (with rx for rounded corners), <line>, <polyline>, <polygon>. Use <path> only for genuinely free-form curves, with as few anchor points as possible (cubic curves with anchors at the extremes, smooth tangents, no wobble).
- Draw each object as ONE complete shape and let shapes overlap like strokes drawn on top of each other: a head is a full circle even where it touches the shoulders, a map pin is a full teardrop resting on a full base ellipse, a line runs straight through a crossing. Never cut a shape into pieces at a junction.
- Straight edges are exactly horizontal or vertical when they are meant to be. Parallel and repeated parts (windows, bars, rows) share exactly the same coordinates and sizes. Round coordinates to 0.5.
- If the original is mirror-symmetric, your drawing must be exactly symmetric.
- Match the original's composition closely: every element in the same place, same size and proportion. Do not add, drop, restyle or simplify away any element."""

FIRST = """Redraw this icon (canvas {W}x{H}, stroke width {SW}).

The first image is the icon. Below is an automatic trace of it: its coordinates show where everything is (use them to place your shapes precisely), but its shapes are rough and have far too many anchor points. Redraw it properly, do not copy its paths.

Automatic trace:
{TRACE}"""

REFINE = """Your previous drawing is below. The second image compares it with the original icon:
  grey/black = correct, RED = in the original but missing or misplaced in your drawing, BLUE = in your drawing but not in the original.
Match score: {SCORE:.1%} (recall {REC:.1%}, precision {PREC:.1%}).

Fix every red and blue area: move, resize or add the shapes so they sit exactly on the original. Keep all the rules (primitives, complete overlapping shapes, symmetry, few anchors). Reply with the complete corrected <svg> only.

Your previous drawing:
{SVG}"""


def _thumb(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def load_examples(folder, max_n=2, size=256):
    """Style examples: pairs NAME.(png|jpg) + NAME.svg in `folder`, simplest
    first. Returns [(png_bytes, svg_text, thumbnail)]."""
    if not folder or not os.path.isdir(folder) or max_n <= 0:
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".svg"):
            continue
        base = os.path.splitext(name)[0]
        img = None
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = os.path.join(folder, base + ext)
            if os.path.exists(p):
                img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
                break
        if img is None:
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as fh:
            svg = fh.read().strip()
        big = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
        out.append((png_bytes(big), svg, _thumb(img)))
    out.sort(key=lambda e: len(e[1]))
    return out[: max_n + 1]  # one spare in case one of them is the icon itself


def _input_png(img, size=512):
    """Clean, enlarged copy of the icon for the model (alpha -> white)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
    h, w = img.shape[:2]
    f = size / max(h, w)
    big = cv2.resize(img, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_CUBIC)
    return png_bytes(big)


# ============================================================== pipeline ==
class Cancelled(Exception):
    pass


def redraw(img, provider, rounds=2, examples=None, stroke_width=None, target=0.97, log=print,
           use_trace_hint=True, n_examples=2, should_stop=None, fallback=True):
    """Redraw one icon with the model, refining against the input.
    Returns (svg, info)."""
    from .tracer import Options, fmt, to_ink, trace
    H, W = img.shape[:2]
    ink, color = to_ink(img)
    local_svg, local_info = trace(img, Options(mode="designer"))
    sw = stroke_width or local_info.get("stroke_width") or max(W, H) / 48
    sw = round(float(sw) * 2) / 2 or 0.5
    sys_prompt = SYSTEM.replace("{W}", fmt(W, 2)).replace("{H}", fmt(H, 2)).replace("{SW}", fmt(sw, 2))

    parts = []
    me = _thumb(img)
    # never show the icon itself as an example (the model would just copy it)
    examples = [e for e in (examples or []) if len(e) < 3 or np.abs(e[2] - me).mean() > 4][:n_examples]
    for k, (ex_png, ex_svg, *_) in enumerate(examples):
        parts.append({"text": f"Style example {k + 1}: this raster icon ..."})
        parts.append({"image": ex_png})
        parts.append({"text": f"... redrawn by a professional designer as:\n{ex_svg}"})
    parts.append({"image": _input_png(img)})
    trace_txt = local_svg if use_trace_hint else "(not available)"
    parts.append({"text": FIRST.replace("{W}", fmt(W, 2)).replace("{H}", fmt(H, 2))
                  .replace("{SW}", fmt(sw, 2)).replace("{TRACE}", trace_txt)})

    best = None
    history = []
    model = provider.resolve_model()
    for rnd in range(rounds + 1):
        if should_stop and should_stop():
            if best is None:
                raise Cancelled("stopped")
            log("  stopped by user, keeping the best drawing so far")
            break
        t0 = time.time()
        log(f"  round {rnd}: {model} is drawing ..." if rnd == 0 else f"  round {rnd}: {model} is correcting ...")
        try:
            answer = provider.generate(sys_prompt, parts, log=log)
        except ProviderError as e:
            log(f"  model error: {e}")
            if best is None:
                if not fallback:
                    raise
                # never leave the user empty-handed: the offline Designer drawing
                log("  -> using the offline Designer result instead")
                info = {"mode": "designer", "fallback": True, "ai_error": str(e)[:300],
                        "anchors": local_info.get("anchors"), "stroke_width": local_info.get("stroke_width")}
                return local_svg, info
            break
        raw = extract_svg(answer)
        if raw is None:
            log("  the model answered without an <svg>; asking again")
            parts = parts + [{"text": "Your answer contained no <svg>. Reply with the complete <svg> only."}]
            continue
        try:
            svg = sanitize_svg(raw, fmt(W, 2), fmt(H, 2))
            score, det, diff = score_svg(svg, ink, sw)
        except ValueError as e:
            log(f"  unusable SVG ({e}); asking again")
            parts = parts + [{"text": f"That SVG was invalid ({e}). Reply with a valid complete <svg> only."}]
            continue
        anchors = count_anchors(svg)
        history.append({"round": rnd, "score": score, "anchors": anchors, **det})
        log(f"  round {rnd}: match {score:.1%} (recall {det['recall']:.1%}, precision {det['precision']:.1%}),"
            f" {anchors} anchors, {time.time() - t0:.1f}s")
        if best is None or score > best[0] + 0.002 or (abs(score - best[0]) <= 0.002 and anchors < best[2]):
            best = (score, svg, anchors, diff)
        if score >= target or rnd == rounds:
            break
        prev = best[1]
        _, det_b, diff_b = score_svg(prev, ink, sw)
        parts = [{"image": _input_png(img)}, {"image": png_bytes(cv2.resize(diff_b, (512, 512)))},
                 {"text": REFINE.replace("{SCORE:.1%}", f"{best[0]:.1%}").replace("{REC:.1%}", f"{det_b['recall']:.1%}")
                  .replace("{PREC:.1%}", f"{det_b['precision']:.1%}").replace("{SVG}", prev)}]
    if best is None:
        raise ProviderError("The model did not produce a usable SVG")
    info = {"mode": "ai", "model": provider.resolve_model(), "score": round(best[0], 4),
            "anchors": best[2], "stroke_width": sw, "rounds": history,
            "local_anchors": local_info.get("anchors")}
    return best[1], info


# ================================================================ config ==
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")


def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


def key_for(provider, cfg=None):
    cfg = cfg if cfg is not None else load_config()
    return (cfg.get("keys") or {}).get(provider) or ""
