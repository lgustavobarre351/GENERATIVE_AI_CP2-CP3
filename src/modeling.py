"""Carregamento do modelo, montagem do prompt, análise por aspecto e geração livre.

Compartilhado por treino, avaliação e frontend, para que todos usem exatamente o mesmo prompt.
"""
import json
import re
import unicodedata
from threading import Thread

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from config import ASPECTS, BASE_MODEL, SENTIMENTS, SYSTEM_PROMPT, get_device, get_dtype, user_prompt


def load_tokenizer(base_model: str = BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(adapter: str | None = None, base_model: str = BASE_MODEL, device: str | None = None):
    """Carrega o modelo base e, opcionalmente, o adaptador LoRA treinado por cima."""
    device = device or get_device()
    tok = load_tokenizer(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=get_dtype(device)).to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def chat_prompt(tok, text: str, system: str = SYSTEM_PROMPT, thinking: bool = False) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt(text)}]
    # enable_thinking=False: o Qwen3 responde direto, sem o bloco <think> de raciocínio
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).strip().lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


_ASPECT_KEYS = {_norm(a): a for a in ASPECTS}
_SENT_KEYS = {_norm(s): s for s in SENTIMENTS}


def parse_output(raw: str) -> dict | None:
    """Converte a resposta do modelo em {aspecto: sentimento}. Retorna None se não houver JSON válido.

    É tolerante a variações comuns do modelo base (texto antes/depois, acentos, maiúsculas),
    mas descarta aspectos e sentimentos fora da lista.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    match = re.search(r"\{.*?\}", raw, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out = {}
    for k, v in data.items():
        aspect, sent = _ASPECT_KEYS.get(_norm(k)), _SENT_KEYS.get(_norm(v))
        if aspect and sent:
            out[aspect] = sent
    return {a: out[a] for a in ASPECTS if a in out}


@torch.no_grad()
def analyze(model, tok, texts: list[str], batch_size: int = 16, max_new_tokens: int = 64) -> list[str]:
    """Gera a resposta (JSON em texto) para cada avaliação, em lotes e de forma determinística."""
    tok.padding_side = "left"
    outputs = []
    for i in range(0, len(texts), batch_size):
        prompts = [chat_prompt(tok, t) for t in texts[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        outputs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outputs]


def generate_stream(model, tok, system: str, prompt: str, max_new_tokens: int = 256,
                    temperature: float = 0.7, top_p: float = 0.9, thinking: bool = False):
    """Geração livre com streaming, usada na aba 'Prompt livre' do frontend."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                     return_dict=True, enable_thinking=thinking).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    Thread(target=model.generate, kwargs=kwargs).start()
    text = ""
    for piece in streamer:
        text += piece
        yield text
