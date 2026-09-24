"""Configurações centrais do projeto (modelo, rótulos, prompt e caminhos)."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"   # adaptadores LoRA treinados
RESULTS_DIR = ROOT / "results"  # métricas, gráficos e previsões

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42

# Dataset B2W-Reviews01 (avaliações reais da Americanas/Submarino/Shoptime, CC BY 4.0)
DATASET_REPO = "ruanchaves/b2w-reviews01"
DATASET_REVISION = "refs/convert/parquet"
DATASET_FILE = "default/train/0000.parquet"

# A ordem define os índices usados em métricas e matriz de confusão
LABELS = ["negativo", "neutro", "positivo"]

SYSTEM_PROMPT = (
    "Você é um classificador de sentimento de avaliações de produtos escritas em português. "
    "Responda apenas com uma palavra: positivo, negativo ou neutro."
)


def user_prompt(text: str) -> str:
    return f"Classifique o sentimento da avaliação abaixo.\n\nAvaliação: {text}"


def get_device() -> str:
    """Escolhe a melhor GPU disponível: CUDA (NVIDIA) > MPS (Apple) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str, training: bool = False) -> torch.dtype:
    if device == "cuda":
        # GPUs Ampere+ (A100, RTX 30xx/40xx) suportam bf16; a T4 do Colab usa fp16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        # Treino em fp16 no MPS é instável sem AMP; inferência em fp16 funciona bem
        return torch.float32 if training else torch.float16
    return torch.float32
