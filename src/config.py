"""Configurações centrais do projeto (modelo, aspectos, prompt e caminhos)."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"   # adaptadores LoRA treinados
RESULTS_DIR = ROOT / "results"  # métricas, gráficos e previsões

BASE_MODEL = "Qwen/Qwen3-1.7B"
SEED = 42

# Aspectos avaliados e sentimentos possíveis (a ordem define a ordem no JSON e nos gráficos)
ASPECTS = ["entrega", "qualidade", "preço", "atendimento", "embalagem"]
SENTIMENTS = ["positivo", "negativo", "neutro"]

SYSTEM_PROMPT = (
    "Você analisa avaliações de produtos escritas em português e identifica o sentimento de cada aspecto mencionado.\n"
    f"Aspectos possíveis: {', '.join(ASPECTS)}.\n"
    f"Sentimentos possíveis: {', '.join(SENTIMENTS)}.\n"
    "Responda somente com um JSON em que cada chave é um aspecto mencionado e o valor é o sentimento dele. "
    "Inclua apenas os aspectos mencionados na avaliação. Se nenhum aspecto for mencionado, responda {}.\n"
    'Exemplo: "Paguei barato, mas a tampa quebrou." -> {"qualidade": "negativo", "preço": "positivo"}'
)


def user_prompt(text: str) -> str:
    return f"Avaliação: {text}"


def get_device() -> str:
    """Escolhe a melhor GPU disponível: CUDA (NVIDIA) > MPS (Apple Metal) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str, training: bool = False) -> torch.dtype:
    if device == "cuda":
        # GPUs Ampere+ (RTX 30xx/40xx, A100) suportam bf16; a T4 usa fp16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        # Apple Silicon: bf16 no treino (os pesos do LoRA ficam em fp32, ver train.py)
        return torch.bfloat16 if training else torch.float16
    return torch.float32
