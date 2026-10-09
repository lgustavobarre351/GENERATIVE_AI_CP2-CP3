"""Avalia o modelo base (zero-shot) ou um adaptador LoRA nos conjuntos de teste.

Conjuntos:
    teste  -> 400 avaliações sintéticas com frases que NUNCA aparecem no treino
    manual -> 40 avaliações escritas à mão, com linguagem natural, gírias e erros de digitação

Exemplos:
    python src/evaluate.py --name baseline_zero_shot
    python src/evaluate.py --name r16_lr2e-4 --adapter outputs/r16_lr2e-4
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from config import ASPECTS, BASE_MODEL, DATA_DIR, RESULTS_DIR, SENTIMENTS
from modeling import analyze, load_model, parse_output

EVAL_SETS = {"teste": "test.jsonl", "manual": "teste_manual.jsonl"}


def prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precisao": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def compute_metrics(golds: list[dict], preds: list[dict | None]) -> dict:
    preds_ok = [p if p is not None else {} for p in preds]
    gold_pairs = [set(g.items()) for g in golds]
    pred_pairs = [set(p.items()) for p in preds_ok]
    tp_pairs = sum(len(g & p) for g, p in zip(gold_pairs, pred_pairs))
    tp_asp = sum(len(set(g) & set(p)) for g, p in zip(golds, preds_ok))

    # Polaridade: entre os aspectos que o modelo detectou corretamente, quantos sentimentos acertou
    cm = [[0] * len(SENTIMENTS) for _ in SENTIMENTS]
    for g, p in zip(golds, preds_ok):
        for a in set(g) & set(p):
            cm[SENTIMENTS.index(g[a])][SENTIMENTS.index(p[a])] += 1
    both = sum(sum(row) for row in cm)

    per_aspect = {}
    for a in ASPECTS:
        tp = sum(1 for g, p in zip(golds, preds_ok) if a in g and g.get(a) == p.get(a))
        per_aspect[a] = prf(tp, sum(a in p for p in preds_ok), sum(a in g for g in golds))["f1"]

    return {
        "n": len(golds),
        "json_valido": round(sum(p is not None for p in preds) / len(preds), 4),
        "acerto_exato": round(sum(g == p for g, p in zip(golds, preds_ok)) / len(golds), 4),
        "pares": prf(tp_pairs, sum(map(len, pred_pairs)), sum(map(len, gold_pairs))),
        "deteccao_aspectos": prf(tp_asp, sum(map(len, preds_ok)), sum(map(len, golds))),
        "acuracia_polaridade": round(sum(cm[i][i] for i in range(len(SENTIMENTS))) / both, 4) if both else 0.0,
        "f1_por_aspecto": per_aspect,
        "matriz_polaridade": cm,
    }


def plot_confusion(cm, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(SENTIMENTS)), SENTIMENTS)
    ax.set_yticks(range(len(SENTIMENTS)), SENTIMENTS)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(title, fontsize=9)
    vmax = max(max(r) for r in cm) or 1
    for i in range(len(SENTIMENTS)):
        for j in range(len(SENTIMENTS)):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="white" if cm[i][j] > vmax / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="limita o conjunto de teste sintético")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args(argv)

    model, tok = load_model(args.adapter, args.base_model)
    run_info = json.loads((Path(args.adapter) / "run_info.json").read_text()) if args.adapter else None
    result = {
        "name": args.name,
        "adapter": args.adapter,
        "device": str(model.device),
        "train": {k: v for k, v in run_info.items() if k != "log_history"} if run_info else None,
        "sets": {},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    n_total, t_total = 0, 0.0

    for set_name, file in EVAL_SETS.items():
        df = pd.read_json(DATA_DIR / file, lines=True)
        if args.limit and set_name == "teste":
            df = df.head(args.limit)
        golds = [dict(label) for label in df["label"]]

        t0 = time.time()
        raw = analyze(model, tok, df["text"].tolist(), batch_size=args.batch_size)
        t_total += time.time() - t0
        n_total += len(df)
        preds = [parse_output(r) for r in raw]

        m = compute_metrics(golds, preds)
        result["sets"][set_name] = m
        pd.DataFrame({
            "text": df["text"],
            "esperado": [json.dumps(g, ensure_ascii=False) for g in golds],
            "resposta_modelo": raw,
            "previsto": [json.dumps(p, ensure_ascii=False) if p is not None else "JSON INVÁLIDO" for p in preds],
            "correto": [g == p for g, p in zip(golds, preds)],
        }).to_csv(RESULTS_DIR / f"{args.name}_{set_name}_predicoes.csv", index=False)

        print(f"\n== {args.name} | conjunto: {set_name} ({len(df)} exemplos, {result['device']}) ==")
        print(f"JSON válido: {m['json_valido']:.1%} | Acerto exato: {m['acerto_exato']:.1%} | "
              f"F1 pares: {m['pares']['f1']:.3f} | F1 detecção: {m['deteccao_aspectos']['f1']:.3f} | "
              f"Acurácia polaridade: {m['acuracia_polaridade']:.1%}")
        print("F1 por aspecto:", m["f1_por_aspecto"])
        if set_name == "manual":
            for text, g, p in list(zip(df["text"], golds, preds))[:40]:
                if g != p:
                    print(f"  ERRO: {text[:90]}\n        esperado={g}\n        previsto={p}")

    teste = result["sets"]["teste"]
    plot_confusion(teste["matriz_polaridade"],
                   f"{args.name} — polaridade (teste)\nF1 pares={teste['pares']['f1']:.3f}",
                   RESULTS_DIR / f"{args.name}_confusao.png")
    result["ms_por_exemplo"] = round(1000 * t_total / n_total, 1)
    (RESULTS_DIR / f"{args.name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n{result['ms_por_exemplo']} ms/exemplo")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    main()
