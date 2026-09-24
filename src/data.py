"""Baixa o B2W-Reviews01, limpa e gera splits balanceados de treino/validação/teste.

Mapeamento das notas (1 a 5 estrelas) para sentimento:
    1-2 -> negativo | 3 -> neutro | 4-5 -> positivo
"""
import argparse
import json

import pandas as pd
from huggingface_hub import hf_hub_download

from config import DATA_DIR, DATASET_FILE, DATASET_REPO, DATASET_REVISION, LABELS, SEED

RATING_TO_LABEL = {1: "negativo", 2: "negativo", 3: "neutro", 4: "positivo", 5: "positivo"}


def download() -> pd.DataFrame:
    raw = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION)
    return pd.read_parquet(raw, columns=["review_title", "review_text", "overall_rating"])


def clean(df: pd.DataFrame, min_chars: int, max_chars: int) -> pd.DataFrame:
    df = df.dropna(subset=["review_text", "overall_rating"]).copy()
    title = df["review_title"].fillna("").str.strip()
    body = df["review_text"].str.strip()
    # Título + texto: o título costuma resumir bem a opinião ("Péssimo!", "Recomendo")
    df["text"] = (title + ". " + body).where(title != "", body)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len().between(min_chars, max_chars)]
    df = df.drop_duplicates(subset="text")
    df["rating"] = df["overall_rating"].astype(int)
    df["label"] = df["rating"].map(RATING_TO_LABEL)
    return df[["text", "label", "rating"]]


def split_balanced(df: pd.DataFrame, sizes: dict) -> dict:
    """Amostra o mesmo número de exemplos por classe em cada split, sem sobreposição."""
    splits = {name: [] for name in sizes}
    for label in LABELS:
        pool = df[df["label"] == label].sample(frac=1, random_state=SEED)
        start = 0
        for name, n in sizes.items():
            splits[name].append(pool.iloc[start:start + n])
            start += n
    return {name: pd.concat(parts).sample(frac=1, random_state=SEED) for name, parts in splits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_per_class", type=int, default=1000)
    ap.add_argument("--val_per_class", type=int, default=100)
    ap.add_argument("--test_per_class", type=int, default=200)
    ap.add_argument("--min_chars", type=int, default=20)
    ap.add_argument("--max_chars", type=int, default=800)
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    df = clean(download(), args.min_chars, args.max_chars)
    print(f"Avaliações após limpeza: {len(df)}")
    print(df["label"].value_counts().to_string())

    sizes = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}
    stats = {"total_limpo": len(df), "distribuicao_original": df["label"].value_counts().to_dict()}
    for name, part in split_balanced(df, sizes).items():
        path = DATA_DIR / f"{name}.jsonl"
        part.to_json(path, orient="records", lines=True, force_ascii=False)
        stats[name] = part["label"].value_counts().to_dict()
        print(f"{name}: {len(part)} exemplos -> {path}")
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
