"""Monitor de GPU ao vivo no terminal (ideal para deixar aberto no VSCode durante o treino).

    python src/gpu_monitor.py              # atualiza a cada 1 s
    python src/gpu_monitor.py --log results/gpu_uso.csv

Apple Silicon: lê a utilização da GPU pelo ioreg (não precisa de sudo).
NVIDIA: usa o nvidia-smi.
"""
import argparse
import re
import shutil
import subprocess
import time
from datetime import datetime


def sample_apple() -> dict:
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout

    def num(key):
        m = re.search(rf'"{re.escape(key)}"=(\d+)', out)
        return int(m.group(1)) if m else 0

    return {
        "gpu_%": num("Device Utilization %"),
        "renderer_%": num("Renderer Utilization %"),
        "mem_gpu_gb": num("In use system memory") / 1024**3,
        "mem_alocada_gb": num("Alloc system memory") / 1024**3,
    }


def sample_nvidia() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split(",")
    util, used, total, temp, power = (float(x) for x in out[:5])
    return {"gpu_%": util, "mem_gpu_gb": used / 1024, "mem_total_gb": total / 1024, "temp_c": temp, "power_w": power}


def bar(pct: float, width: int = 30) -> str:
    filled = round(width * pct / 100)
    return "█" * filled + "░" * (width - filled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--log", default=None, help="salva as amostras em CSV")
    args = ap.parse_args()

    nvidia = shutil.which("nvidia-smi") is not None
    sample = sample_nvidia if nvidia else sample_apple
    print(f"Monitorando GPU {'NVIDIA (nvidia-smi)' if nvidia else 'Apple Silicon (Metal)'} — Ctrl+C para sair\n")
    log = open(args.log, "a") if args.log else None
    try:
        while True:
            s = sample()
            now = datetime.now().strftime("%H:%M:%S")
            extra = f" | {s['temp_c']:.0f}°C {s['power_w']:.0f} W" if nvidia else ""
            print(f"\r{now}  GPU {bar(s['gpu_%'])} {s['gpu_%']:5.1f}%  | memória GPU {s['mem_gpu_gb']:5.2f} GB{extra}   ",
                  end="", flush=True)
            if log:
                log.write(f"{now}," + ",".join(f"{v:.2f}" for v in s.values()) + "\n")
                log.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
