"""Téléchargement et chargement du dataset IMDB hébergé sur Hugging Face."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text"
SPLITS = {
    "train": f"{BASE_URL}/train-00000-of-00001.parquet",
    "test": f"{BASE_URL}/test-00000-of-00001.parquet",
}
LABELS = {0: "négatif", 1: "positif"}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def download_split(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Télécharge un split (train ou test) une seule fois et renvoie son chemin local."""
    if split not in SPLITS:
        raise ValueError(f"Split inconnu : {split!r}. Valeurs possibles : {sorted(SPLITS)}")
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"imdb_{split}.parquet"
    if not target.exists():
        print(f"Téléchargement du split {split} depuis Hugging Face…")
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(SPLITS[split], tmp)
        tmp.replace(target)
    return target


def load_split(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Renvoie un DataFrame avec les colonnes ``text`` et ``label`` (0 = négatif, 1 = positif)."""
    df = pd.read_parquet(download_split(split, data_dir))
    return df[["text", "label"]].reset_index(drop=True)
