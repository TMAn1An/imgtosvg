# icon2svg — JPG/PNG আইকন → প্রফেশনাল SVG

লাইন-আইকন (JPG/PNG) দিলে এটা হাতে আঁকা মানের পরিষ্কার SVG বানায়:

- **শার্প ও স্মুথ**: সোজা লাইন একদম সোজা (H/V লাইন axis-এ snap করা), কার্ভ মসৃণ
- **কম anchor point**: Illustrator-এর *Simplify* এর মতো অপ্রয়োজনীয় পয়েন্ট বাদ দেয়
- **Stroke মোড**: centre-line stroke, round cap/join, এক stroke-width. Illustrator/Figma-তে ঠিক যেভাবে লাইন আইকন আঁকা হয়
- **Outline মোড**: filled/solid আইকনের জন্য filled path
- **Auto মোড**: দুটোর মধ্যে যেটা ভালো মেলে সেটা নিজে বেছে নেয়
- গোল জিনিস (রিং, মাথা, ডট) **পারফেক্ট সার্কেল** হয়, ভাঙা রিংও জোড়া লাগিয়ে পুরো সার্কেল করে
- **ডিজাইনারের মতো রিড্র**: প্রতিটা বন্ধ অংশ (জানালা, মাথা, বেস, হ্যান্ডেল…) চেনে এবং exact
  rectangle / rounded-rectangle / ellipse / circle / polygon দিয়ে আঁকে। দুটো অংশ মিলে rectangle হলে
  (যেমন দেয়াল + দরজা) পুরো rectangle আঁকে
- **সিমেট্রি**: আইকন আয়নার মতো সিমেট্রিক হলে দুই পাশ হুবহু সমান করে
- **অ্যালাইনমেন্ট**: প্রায়-সমান x/y গুলো এক করে দেয়, টুকরো সোজা লাইন জোড়া দিয়ে একটা লাইন বানায়
- রঙ ইমেজ থেকে নেয় (বা নিজে দিতে পারেন / `currentColor`)
- Transparent PNG সাপোর্ট করে

---

## ১) Windows-এ চালানো (সবচেয়ে সহজ)

1. **Python 3.10 বা নতুন** ইন্সটল করুন: <https://www.python.org/downloads/>
   ইন্সটলের সময় **"Add python.exe to PATH"** টিক দেবেন।
2. zip ফাইলটা extract করুন।
3. **`run_windows.bat`**-এ ডাবল ক্লিক করুন।
   - প্রথমবার ১-২ মিনিট লাগবে (লাইব্রেরি ইন্সটল হবে, ইন্টারনেট লাগবে)।
   - ব্রাউজারে আপনা-আপনি `http://127.0.0.1:5000` খুলবে।
4. আইকনগুলো ড্র্যাগ-ড্রপ করুন, প্রিভিউ দেখুন, **Download all (ZIP)** চাপুন।

পুরো ফোল্ডার একবারে কনভার্ট করতে: ফোল্ডারটা **`convert_folder_windows.bat`**-এর উপর ড্র্যাগ-ড্রপ করুন।
SVG গুলো ইনপুট ফোল্ডারের পাশে `Output` ফোল্ডারে তৈরি হবে।

## ২) Mac / Linux

```bash
./run_mac_linux.sh            # ওয়েব অ্যাপ
./run_mac_linux.sh Input/     # পুরো ফোল্ডার কনভার্ট -> Output/
```

## ৩) কমান্ড লাইন (সব অপশন)

```bash
python convert.py Input/                         # ফোল্ডার -> Output/
python convert.py icon.png -o out/               # একটা ফাইল
python convert.py Input/ --mode stroke           # সবসময় stroke
python convert.py Input/ --mode outline          # সবসময় filled outline
python convert.py Input/ --tolerance 0.5         # আরও স্মুথ, আরও কম পয়েন্ট
python convert.py Input/ --stroke-width 2.4      # নির্দিষ্ট stroke width
python convert.py Input/ --size 64               # 64x64 viewBox
python convert.py Input/ --color currentColor    # CSS থেকে রঙ নিতে
```

| অপশন | কাজ |
|---|---|
| `--mode auto/stroke/outline` | আউটপুট স্টাইল (ডিফল্ট `auto`) |
| `--tolerance` | কার্ভ ফিটিং টলারেন্স (130px আইকন অনুযায়ী px)। বাড়ালে পয়েন্ট কমে ও বেশি স্মুথ হয়। ডিফল্ট `0.30`, স্বাভাবিক রেঞ্জ `0.2 – 0.7` |
| `--stroke-width` | stroke-এর পুরুত্ব জোর করে সেট করা (আউটপুট ইউনিটে) |
| `--size` | লম্বা দিকটা এই সাইজে স্কেল হবে (24, 48, 64, 512…) |
| `--color` | `auto` (ইমেজ থেকে), `#000000`, `currentColor` ইত্যাদি |
| `--no-circles` | গোল শেপকে পারফেক্ট সার্কেলে snap করবে না |
| `--no-symmetry` | সিমেট্রিক আইকনকে হুবহু সিমেট্রিক করবে না |
| `--no-primitives` | জানালা/মাথা ইত্যাদি exact rect/ellipse/polygon দিয়ে রিড্র করবে না (শুধু ট্রেস) |

## টিপস

- **আপনার রেফারেন্স SVG-র মতো পাতলা লাইন চাইলে**: রেফারেন্সগুলো 130×130 viewBox-এ মোটামুটি `2.4` stroke-width দিয়ে আঁকা,
  কিন্তু ইনপুট JPG-র লাইন প্রায় `3` px মোটা। হুবহু রেফারেন্সের মতো চাইলে `--stroke-width 2.4` দিন।
- Stroke SVG-কে Illustrator-এ filled shape বানাতে: *Object → Expand* (বা *Outline Stroke*)।
- ইনপুট যত বড়/পরিষ্কার হবে (যেমন 512px PNG), রেজাল্ট তত নিখুঁত হবে। 130px-এর JPG-ও ভালো কাজ করে।
- সবচেয়ে ভালো ফল পাওয়া যায় **এক রঙের** লাইন/গ্লিফ আইকনে। মাল্টি-কালার ইলাস্ট্রেশন এর জন্য না।

## কিভাবে কাজ করে (সংক্ষেপে)

1. ইমেজ থেকে "ink" ম্যাপ (auto background/polarity, alpha, contrast)
2. Bicubic super-sampling + sub-pixel contour/skeleton
3. **Stroke মোড**: skeleton → graph → ছোট spur ছাঁটা → junction-এ সোজা চলে যাওয়া লাইন জোড়া →
   anti-aliased coverage থেকে stroke-width মাপা → এক বৃত্তে পড়া টুকরোগুলো মিলিয়ে পুরো সার্কেল/আর্ক
4. প্রতিটা path: corner detection → সোজা অংশ = line (H/V snap) → বাকি অংশ tangent-continuous cubic Bezier
   (anchor x/y extrema-তে রাখার চেষ্টা) → Simplify (যতক্ষণ tolerance-এর মধ্যে থাকে anchor বাদ) →
   দুটো সোজা লাইনের মাঝের কোণা হয় পরিষ্কার sharp corner নাহয় ২-anchor fillet
5. Auto মোড stroke রেজাল্ট রেন্ডার করে ইনপুটের সাথে মিলিয়ে দেখে; না মিললে outline দেয়

## ফাইল

```
app.py                      লোকাল ওয়েব অ্যাপ (Flask)
convert.py                  কমান্ড লাইন / ব্যাচ কনভার্টার
icon2svg/tracer.py          ইমেজ লোড, outline মোড, path fitting, SVG লেখা
icon2svg/stroke.py          stroke (centre-line) মোড
icon2svg/bezier.py          Bezier fitting, simplify, fillet, axis snap
web/index.html              ওয়েব UI
run_windows.bat             Windows লঞ্চার (প্রথমবার নিজেই সেটআপ করে)
convert_folder_windows.bat  ফোল্ডার ড্র্যাগ-ড্রপ করে ব্যাচ কনভার্ট
run_mac_linux.sh            Mac/Linux লঞ্চার
tools/evaluate.py           ফোল্ডার কনভার্ট করে মান যাচাই (dev)
tools/debug_view.py         anchor/handle সহ ডিবাগ ছবি (dev)
tests/                      pytest
```

---

## English quick start

Needs Python 3.10+. On Windows double-click `run_windows.bat`; on Mac/Linux run `./run_mac_linux.sh`.
A browser page opens at `http://127.0.0.1:5000`. Drop icons there and download the SVGs.
Batch mode: `python convert.py Input/` writes to `Output/`. See the table above for options.
