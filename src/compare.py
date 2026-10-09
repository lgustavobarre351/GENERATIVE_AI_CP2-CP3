"""Junta os resultados de todos os experimentos em uma tabela (Markdown) e gráficos."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ASPECTS, OUTPUT_DIR, RESULTS_DIR


def main():
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if "sets" in data:
            runs.append(data)
    if not runs:
        print("Nenhum resultado encontrado em results/")
        return
    runs.sort(key=lambda r: r["sets"]["teste"]["pares"]["f1"], reverse=True)

    lines = [
        "| Experimento | LoRA r | Camadas | LR | Épocas | Params treináveis | Tempo de treino | Loss val "
        "| JSON válido | Acerto exato | F1 pares (teste) | F1 pares (manual) | Acerto exato (manual) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        t, te, ma = r.get("train"), r["sets"]["teste"], r["sets"]["manual"]
        if t:
            p = t["params"]
            cols = [p["lora_r"], "todas" if p["target"] == "all" else "atenção", p["lr"], p["epochs"],
                    f"{t['trainable_params'] / 1e6:.1f}M", f"{t['train_time_s'] / 60:.0f} min", t["final_eval_loss"]]
        else:
            cols = ["—"] * 7
        cols += [f"{te['json_valido']:.0%}", f"{te['acerto_exato']:.1%}", f"**{te['pares']['f1']:.3f}**",
                 f"{ma['pares']['f1']:.3f}", f"{ma['acerto_exato']:.1%}"]
        lines.append(f"| {r['name']} | " + " | ".join(str(c) for c in cols) + " |")

    table = "\n".join(lines)
    (RESULTS_DIR / "RESULTADOS.md").write_text(
        "# Resultados dos experimentos\n\n"
        "Ordenado pelo F1 dos pares (aspecto, sentimento) no teste sintético com frases inéditas.\n\n"
        f"{table}\n")
    print(table)

    names = [r["name"] for r in runs]
    x = list(range(len(runs)))

    # Gráfico 1: F1 dos pares e acerto exato, nos dois conjuntos de teste
    fig, ax = plt.subplots(figsize=(max(7, len(runs) * 1.2), 4.2))
    series = [
        ("F1 pares (teste)", [r["sets"]["teste"]["pares"]["f1"] for r in runs]),
        ("F1 pares (manual)", [r["sets"]["manual"]["pares"]["f1"] for r in runs]),
        ("Acerto exato (teste)", [r["sets"]["teste"]["acerto_exato"] for r in runs]),
    ]
    w = 0.27
    for k, (label, vals) in enumerate(series):
        ax.bar([i + (k - 1) * w for i in x], vals, width=w, label=label)
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Comparação dos experimentos")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "comparacao.png", dpi=150)
    plt.close(fig)

    # Gráfico 2: F1 por aspecto (baseline x melhor modelo)
    best = runs[0]
    base = next((r for r in runs if r["name"] == "baseline_zero_shot"), None)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    compare = [base, best] if base and base is not best else [best]
    for k, r in enumerate(compare):
        vals = [r["sets"]["teste"]["f1_por_aspecto"][a] for a in ASPECTS]
        ax.bar([i + (k - 0.5 * (len(compare) - 1)) * 0.4 for i in range(len(ASPECTS))], vals, width=0.4, label=r["name"])
    ax.set_xticks(range(len(ASPECTS)), ASPECTS)
    ax.set_ylim(0, 1)
    ax.set_title("F1 por aspecto (teste)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "f1_por_aspecto.png", dpi=150)
    plt.close(fig)

    # Gráfico 3: curvas de loss de treino
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in runs:
        info_path = OUTPUT_DIR / r["name"] / "run_info.json"
        if not info_path.exists():
            continue
        hist = [h for h in json.loads(info_path.read_text())["log_history"] if "loss" in h]
        ax.plot([h["step"] for h in hist], [h["loss"] for h in hist], label=r["name"])
    ax.set_xlabel("Passo")
    ax.set_ylabel("Loss de treino")
    ax.set_yscale("log")
    ax.set_title("Curvas de loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "curvas_loss.png", dpi=150)
    plt.close(fig)
    print(f"\nTabela: {RESULTS_DIR / 'RESULTADOS.md'} | gráficos: comparacao.png, f1_por_aspecto.png, curvas_loss.png")


if __name__ == "__main__":
    main()
