"""Fine-tuning do Qwen3 com LoRA para sentimento por aspecto (resposta em JSON).

Exemplo:
    python src/train.py --run_name r16_lr2e-4 --lora_r 16 --lr 2e-4 --epochs 1
"""
import argparse
import json
import time

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

from config import BASE_MODEL, DATA_DIR, OUTPUT_DIR, SEED, get_device, get_dtype
from modeling import chat_prompt, load_tokenizer

TARGET_MODULES = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


class EmptyMPSCache(TrainerCallback):
    """No Apple Silicon, o cache de memória do MPS cresce ao longo do treino e o Mac começa a usar swap.
    Liberar o cache periodicamente mantém a velocidade estável."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            torch.mps.empty_cache()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=None, help="padrão: 2 * r")
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target", choices=TARGET_MODULES, default="all")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="desliga gradient checkpointing (mais rápido, usa mais memória)")
    ap.add_argument("--train_limit", type=int, default=None, help="usar só N exemplos (teste rápido)")
    ap.add_argument("--val_limit", type=int, default=None)
    return ap.parse_args(argv)


def load_split(name: str, limit: int | None) -> pd.DataFrame:
    df = pd.read_json(DATA_DIR / f"{name}.jsonl", lines=True)
    return df.head(limit) if limit else df


def tokenize(df: pd.DataFrame, tok, max_len: int) -> Dataset:
    """Prompt + resposta; a loss só é calculada nos tokens da resposta (o JSON)."""
    rows = []
    for text, label in zip(df["text"], df["label"]):
        prompt_ids = tok(chat_prompt(tok, text), add_special_tokens=False).input_ids
        answer = json.dumps(dict(label), ensure_ascii=False)
        answer_ids = tok(answer, add_special_tokens=False).input_ids + [tok.eos_token_id]
        prompt_ids = prompt_ids[-(max_len - len(answer_ids)):]
        rows.append({
            "input_ids": prompt_ids + answer_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
            "labels": [-100] * len(prompt_ids) + answer_ids,
        })
    return Dataset.from_list(rows)


def main(argv=None):
    args = parse_args(argv)
    args.lora_alpha = args.lora_alpha or 2 * args.lora_r
    set_seed(SEED)
    device = get_device()
    dtype = get_dtype(device, training=True)
    out_dir = OUTPUT_DIR / args.run_name
    print(f"Dispositivo: {device} | dtype: {dtype} | saída: {out_dir}")

    tok = load_tokenizer(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    model.config.use_cache = False
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=TARGET_MODULES[args.target], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # Pesos do LoRA em fp32 (o modelo base continua em fp16) para o treino misto ser estável
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    trainable, total = model.get_nb_trainable_parameters()

    train_ds = tokenize(load_split("train", args.train_limit), tok, args.max_len)
    val_ds = tokenize(load_split("val", args.val_limit), tok, args.max_len)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # 5% dos passos (float = proporção no transformers v5)
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=0.5,  # avalia na metade e no fim do treino
        save_strategy="no",
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16 and device == "cuda",  # no MPS o modelo já está em bf16, sem autocast
        report_to="none",
        seed=SEED,
        dataloader_pin_memory=device == "cuda",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
                      # Comprimentos múltiplos de 32: menos formatos distintos de tensor = menos recompilação na GPU
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, pad_to_multiple_of=32,
                                                           label_pad_token_id=-100),
                      callbacks=[EmptyMPSCache()] if device == "mps" else None)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0

    model.save_pretrained(out_dir)
    info = {
        "run_name": args.run_name,
        "params": vars(args),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "dtype": str(dtype),
        "trainable_params": trainable,
        "total_params": total,
        "train_examples": len(train_ds),
        "train_time_s": round(train_time, 1),
        "train_loss": round(result.training_loss, 4),
        "final_eval_loss": round(trainer.evaluate()["eval_loss"], 4),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None,
        "log_history": trainer.state.log_history,
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"Treino concluído em {train_time:.0f}s | loss {info['train_loss']} | eval_loss {info['final_eval_loss']}")
    return out_dir


if __name__ == "__main__":
    main()
