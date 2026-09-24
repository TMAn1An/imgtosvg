"""Check whether this PC can run a local vision AI model (Ollama) and which one.

Usage: python tools/check_pc.py      (or double-click check_pc_windows.bat)
Prints the hardware it finds and a recommendation. Nothing is installed.
"""
import os
import platform
import shutil
import subprocess


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, shell=isinstance(cmd, str)).stdout
    except Exception:
        return ""


def ram_gb():
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEM()
            m.dwLength = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1024 ** 3
        if platform.system() == "Darwin":
            return int(run(["sysctl", "-n", "hw.memsize"]).strip() or 0) / 1024 ** 3
        with open("/proc/meminfo") as fh:
            return int(fh.readline().split()[1]) / 1024 ** 2
    except Exception:
        return 0.0


def gpus():
    """[(name, vram_gb or None)]"""
    out = []
    smi = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    for line in smi.strip().splitlines():
        if "," in line:
            name, mem = line.rsplit(",", 1)
            try:
                out.append((name.strip(), float(mem) / 1024))
            except ValueError:
                out.append((name.strip(), None))
    if out:
        return out
    if platform.system() == "Windows":
        txt = run('powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | '
                  'ForEach-Object { $_.Name + \'|\' + $_.AdapterRAM }"')
        for line in txt.strip().splitlines():
            if "|" in line:
                name, mem = line.split("|", 1)
                try:
                    v = int(mem) / 1024 ** 3  # capped at 4 GB by Windows for some cards
                except ValueError:
                    v = None
                out.append((name.strip(), v))
    elif platform.system() == "Darwin":
        chip = run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if "Apple" in chip:
            out.append((chip + " (unified memory)", None))
    return out


def main():
    ram = ram_gb()
    free = shutil.disk_usage(os.path.abspath(os.sep)).free / 1024 ** 3
    cpu = platform.processor() or platform.machine()
    gl = gpus()
    nvidia = [(n, v) for n, v in gl if "NVIDIA" in n.upper() or "GEFORCE" in n.upper() or "RTX" in n.upper()]
    vram = max([v for _, v in nvidia if v] or [0])
    apple = any("Apple" in n for n, _ in gl)
    ollama = shutil.which("ollama") is not None

    print("=" * 60)
    print(" PC check for local AI (Ollama)")
    print("=" * 60)
    print(f" System     : {platform.system()} {platform.release()}")
    print(f" Processor  : {cpu}  ({os.cpu_count()} threads)")
    print(f" RAM        : {ram:.1f} GB")
    print(f" Free disk  : {free:.0f} GB (drive {os.path.abspath(os.sep)})")
    if gl:
        for n, v in gl:
            print(f" Graphics   : {n}" + (f"  ({v:.1f} GB)" if v else ""))
    else:
        print(" Graphics   : (not detected)")
    print(f" Ollama     : {'installed' if ollama else 'not installed (https://ollama.com/download)'}")
    print("-" * 60)

    if free < 8:
        print(" NOT ENOUGH DISK: free at least 8-10 GB first.")
    if vram >= 10 or (apple and ram >= 16):
        verdict = ("GOOD: runs well, a few seconds per icon.",
                   ["qwen2.5vl:7b   (best local choice for this job)", "gemma3:12b     (alternative)"])
    elif vram >= 6:
        verdict = ("GOOD: runs on your graphics card, ~5-15 s per icon.",
                   ["qwen2.5vl:7b   (fits in ~6 GB)", "gemma3:4b      (lighter, faster)"])
    elif vram >= 3.5 or (apple and ram >= 8):
        verdict = ("OK: small models run well.", ["gemma3:4b", "qwen2.5vl:3b"])
    elif ram >= 14:
        verdict = ("SLOW BUT POSSIBLE: no suitable graphics card, runs on the processor "
                   "(~1-3 minutes per icon).", ["gemma3:4b", "qwen2.5vl:3b"])
    elif ram >= 7:
        verdict = ("VERY SLOW: only the smallest models, several minutes per icon.", ["qwen2.5vl:3b"])
    else:
        verdict = ("NOT RECOMMENDED: too little memory for a vision model.", [])
    print(" Verdict    : " + verdict[0])
    for m in verdict[1]:
        print("   ollama pull " + m)
    print("-" * 60)
    print(" Copy everything above and send it, so the tool can be tuned for this PC.")


if __name__ == "__main__":
    main()
    if platform.system() == "Windows":
        input("\nPress Enter to close ...")
