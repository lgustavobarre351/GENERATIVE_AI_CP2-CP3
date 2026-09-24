"""Mostra (e salva) as informações da GPU usada no projeto.

Rode este script no terminal/notebook do VSCode e tire o print da saída
para comprovar que a GPU está instalada e funcionando.
"""
import json
import platform
import subprocess
import time

import torch

from config import RESULTS_DIR, get_device


def gpu_info() -> dict:
    device = get_device()
    info = {
        "python": platform.python_version(),
        "sistema": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "device": device,
        "cuda_disponivel": torch.cuda.is_available(),
    }
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update({
            "cuda_versao_torch": torch.version.cuda,
            "cudnn_versao": torch.backends.cudnn.version(),
            "qtd_gpus": torch.cuda.device_count(),
            "gpu_nome": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "memoria_total_gb": round(props.total_memory / 1024**3, 2),
            "multiprocessadores": props.multi_processor_count,
            "suporta_bf16": torch.cuda.is_bf16_supported(),
        })
    elif device == "mps":
        info["gpu_nome"] = "Apple Silicon (Metal Performance Shaders)"
    return info


def benchmark(device: str, n: int = 4096, reps: int = 20) -> float:
    """Multiplicação de matrizes na GPU, para provar que ela está realmente processando."""
    dtype = torch.float16 if device != "cpu" else torch.float32
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    sync = torch.cuda.synchronize if device == "cuda" else (torch.mps.synchronize if device == "mps" else (lambda: None))
    a @ b
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    sync()
    elapsed = time.perf_counter() - t0
    return round(2 * n**3 * reps / elapsed / 1e12, 2)  # TFLOPS


def main():
    info = gpu_info()
    info["benchmark_matmul_tflops"] = benchmark(info["device"])

    print("=" * 60)
    print(" INFORMAÇÕES DA GPU")
    print("=" * 60)
    for k, v in info.items():
        print(f"{k:>26}: {v}")
    print("=" * 60)

    if info["device"] == "cuda":
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "gpu_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
