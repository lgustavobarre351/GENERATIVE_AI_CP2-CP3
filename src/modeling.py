"""Carregamento do modelo, montagem do prompt, classificação e geração de texto.

Compartilhado por treino, avaliação e frontend, para que todos usem exatamente o mesmo prompt.
"""
from threading import Thread

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from config import BASE_MODEL, LABELS, SYSTEM_PROMPT, get_device, get_dtype, user_prompt


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


def chat_prompt(tok, text: str, system: str = SYSTEM_PROMPT) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt(text)}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def classify(model, tok, texts: list[str], batch_size: int = 8, max_prompt_tokens: int = 512) -> torch.Tensor:
    """Retorna as probabilidades (n_textos x n_rótulos) de cada rótulo.

    Em vez de gerar texto livre e tentar interpretar a resposta, calcula a
    log-verossimilhança que o modelo atribui a cada rótulo como continuação do
    prompt. Assim o modelo sempre responde um rótulo válido e ainda temos uma
    medida de confiança.
    """
    device = model.device
    label_ids = [tok(label, add_special_tokens=False).input_ids for label in LABELS]
    max_label_len = max(len(ids) for ids in label_ids)
    all_probs = []

    for i in range(0, len(texts), batch_size):
        seqs, lens = [], []
        for text in texts[i:i + batch_size]:
            prompt_ids = tok(chat_prompt(tok, text), add_special_tokens=False).input_ids[-max_prompt_tokens:]
            for ids in label_ids:
                seqs.append(prompt_ids + ids)
                lens.append(len(ids))

        # Padding à esquerda: o rótulo fica sempre no final, então basta calcular
        # os logits das últimas posições (economiza muita memória com vocabulário de 150k).
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(seqs), width), dtype=torch.long)
        for j, s in enumerate(seqs):
            input_ids[j, width - len(s):] = torch.tensor(s)
            attention[j, width - len(s):] = 1
        position_ids = (attention.cumsum(-1) - 1).clamp(min=0)

        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention.to(device),
            position_ids=position_ids.to(device),
            logits_to_keep=max_label_len + 1,
        ).logits.float()
        logp = F.log_softmax(logits, dim=-1)  # (seqs, max_label_len + 1, vocab)

        scores = []
        for j, n in enumerate(lens):
            # Os logits na posição t predizem o token t+1
            targets = input_ids[j, width - n:].to(device)
            preds = logp[j, -n - 1:-1]
            scores.append(preds.gather(-1, targets[:, None]).sum())
        scores = torch.stack(scores).view(-1, len(LABELS))
        all_probs.append(scores.softmax(-1).cpu())

    return torch.cat(all_probs)


def generate_stream(model, tok, system: str, prompt: str, max_new_tokens: int = 256,
                    temperature: float = 0.7, top_p: float = 0.9):
    """Geração livre com streaming, usada na aba 'Prompt livre' do frontend."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                     return_dict=True).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens,
                  pad_token_id=tok.pad_token_id)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs.update(do_sample=False)
    Thread(target=model.generate, kwargs=kwargs).start()
    text = ""
    for piece in streamer:
        text += piece
        yield text
