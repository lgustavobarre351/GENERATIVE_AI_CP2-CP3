"""Junta os resultados de todos os experimentos em uma tabela (Markdown) e gráficos."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, RESULTS_DIR


def main():
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if "f1_macro" in data:
            runs.append(data)
    if not runs:
        print("Nenhum resultado encontrado em results/")
        return
    runs.sort(key=lambda r: r["f1_macro"], reverse=True)

    lines = [
        "| Experimento | LoRA r | Alvo | LR | Épocas | Params treináveis | Tempo treino | Loss val | Acurácia | F1 macro | Acc pos/neg |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        t = r.get("train")
        if t:
            p = t["params"]
            cols = [p["lora_r"], p["target"], p["lr"], p["epochs"], f"{t['trainable_params'] / 1e6:.1f}M",
                    f"{t['train_time_s'] / 60:.1f} min", t["final_eval_loss"]]
        else:
            cols = ["—"] * 5 + ["—", "—"]
        cols += [f"{r['accuracy']:.3f}", f"**{r['f1_macro']:.3f}**", f"{r['accuracy_polar']:.3f}"]
        lines.append(f"| {r['name']} | " + " | ".join(str(c) for c in cols) + " |")

    table = "\n".join(lines)
    (RESULTS_DIR / "RESULTADOS.md").write_text(f"# Resultados dos experimentos\n\n{table}\n")
    print(table)

    # Gráfico 1: acurácia e F1 por experimento
    names = [r["name"] for r in runs]
    x = range(len(runs))
    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.1), 4))
    ax.bar([i - 0.2 for i in x], [r["accuracy"] for r in runs], width=0.4, label="Acurácia")
    ax.bar([i + 0.2 for i in x], [r["f1_macro"] for r in runs], width=0.4, label="F1 macro")
    ax.set_xticks(list(x), names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Comparação dos experimentos (conjunto de teste)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "comparacao.png", dpi=150)
    plt.close(fig)

    # Gráfico 2: curvas de loss de treino
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in runs:
        info_path = OUTPUT_DIR / r["name"] / "run_info.json"
        if not info_path.exists():
            continue
        hist = [h for h in json.loads(info_path.read_text())["log_history"] if "loss" in h]
        ax.plot([h["step"] for h in hist], [h["loss"] for h in hist], label=r["name"])
    ax.set_xlabel("Passo")
    ax.set_ylabel("Loss de treino")
    ax.set_title("Curvas de loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "curvas_loss.png", dpi=150)
    plt.close(fig)
    print(f"\nTabela: {RESULTS_DIR / 'RESULTADOS.md'} | gráficos: comparacao.png, curvas_loss.png")


if __name__ == "__main__":
    main()
