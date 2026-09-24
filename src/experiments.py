"""Roda a grade de experimentos (treino + avaliação) variando os hiperparâmetros.

Cada experimento roda em um subprocesso separado para liberar a memória da GPU entre eles.

    python src/experiments.py                  # todos
    python src/experiments.py --only r16_lr2e-4 r8_lr2e-4
    python src/experiments.py --quick          # teste rápido com poucos exemplos
"""
import argparse
import subprocess
import sys
from pathlib import Path

from config import OUTPUT_DIR, RESULTS_DIR

SRC = Path(__file__).parent

# Referência: r=16, lr=2e-4, 1 época, LoRA em todas as camadas lineares.
# Cada experimento muda UM parâmetro em relação à referência.
EXPERIMENTS = {
    "r16_lr2e-4":       ["--lora_r", "16", "--lr", "2e-4", "--epochs", "1"],
    "r4_lr2e-4":        ["--lora_r", "4",  "--lr", "2e-4", "--epochs", "1"],
    "r64_lr2e-4":       ["--lora_r", "64", "--lr", "2e-4", "--epochs", "1"],
    "r16_lr2e-5":       ["--lora_r", "16", "--lr", "2e-5", "--epochs", "1"],
    "r16_lr1e-3":       ["--lora_r", "16", "--lr", "1e-3", "--epochs", "1"],
    "r16_lr2e-4_ep3":   ["--lora_r", "16", "--lr", "2e-4", "--epochs", "3"],
    "r16_lr2e-4_attn":  ["--lora_r", "16", "--lr", "2e-4", "--epochs", "1", "--target", "attn"],
}


def run(script: str, *args: str):
    cmd = [sys.executable, str(SRC / script), *args]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="nomes dos experimentos a rodar")
    ap.add_argument("--quick", action="store_true", help="200 exemplos de treino / 60 de teste")
    ap.add_argument("--skip_baseline", action="store_true")
    args = ap.parse_args()

    quick_train = ["--train_limit", "200", "--val_limit", "30"] if args.quick else []
    quick_eval = ["--limit", "60"] if args.quick else []

    if not args.skip_baseline and not (RESULTS_DIR / "baseline_zero_shot.json").exists():
        run("evaluate.py", "--name", "baseline_zero_shot", *quick_eval)

    for name in args.only or EXPERIMENTS:
        run("train.py", "--run_name", name, *EXPERIMENTS[name], *quick_train)
        run("evaluate.py", "--name", name, "--adapter", str(OUTPUT_DIR / name), *quick_eval)

    run("compare.py")


if __name__ == "__main__":
    main()
