"""Avalia o modelo base (zero-shot) ou um adaptador LoRA no conjunto de teste.

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import BASE_MODEL, DATA_DIR, LABELS, RESULTS_DIR
from modeling import classify, load_model


def plot_confusion(cm, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(title, fontsize=10)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args(argv)

    test = pd.read_json(DATA_DIR / "test.jsonl", lines=True)
    if args.limit:
        test = test.head(args.limit)

    model, tok = load_model(args.adapter, args.base_model)
    device = str(model.device)
    t0 = time.time()
    probs = classify(model, tok, test["text"].tolist(), batch_size=args.batch_size)
    elapsed = time.time() - t0

    test["pred"] = [LABELS[i] for i in probs.argmax(-1).tolist()]
    test["confianca"] = probs.max(-1).values.double().numpy().round(3)
    for k, label in enumerate(LABELS):
        test[f"p_{label}"] = probs[:, k].double().numpy().round(3)

    y_true, y_pred = test["label"], test["pred"]
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    # Acurácia "polar": só positivo x negativo, ignorando a classe neutra (ambígua por natureza)
    polar = test[test["label"] != "neutro"]
    run_info = json.loads((Path(args.adapter) / "run_info.json").read_text()) if args.adapter else None

    metrics = {
        "name": args.name,
        "adapter": args.adapter,
        "device": device,
        "n_test": len(test),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, labels=LABELS, average="macro"), 4),
        "accuracy_polar": round(accuracy_score(polar["label"], polar["pred"]), 4),
        "report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "ms_por_exemplo": round(1000 * elapsed / len(test), 1),
        "train": {k: v for k, v in run_info.items() if k != "log_history"} if run_info else None,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{args.name}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    test.to_csv(RESULTS_DIR / f"{args.name}_predicoes.csv", index=False)
    plot_confusion(cm, f"{args.name}\nacc={metrics['accuracy']:.3f}  F1={metrics['f1_macro']:.3f}",
                   RESULTS_DIR / f"{args.name}_confusao.png")

    print(f"\n== {args.name} ({device}) ==")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    print(f"Acurácia: {metrics['accuracy']} | F1 macro: {metrics['f1_macro']} | "
          f"Acurácia pos/neg: {metrics['accuracy_polar']} | {metrics['ms_por_exemplo']} ms/exemplo")
    errors = test[test["label"] != test["pred"]].head(5)
    if len(errors):
        print("\nExemplos de erros:")
        for _, r in errors.iterrows():
            print(f"  [real={r['label']} previsto={r['pred']} conf={r['confianca']}] {r['text'][:120]}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


if __name__ == "__main__":
    main()
