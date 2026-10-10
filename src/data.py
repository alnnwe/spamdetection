"""Загрузка, очистка, удаление дубликатов и разбиение датасета SMS Spam Collection.

Запуск:  python -m src.data
Результат: data/processed/{train,val,test}.csv и results/data_summary.json
"""
import csv
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (DATA_URL, PROCESSED_DIR, RANDOM_STATE, RAW_DIR, RAW_FILE,
                        RESULTS_DIR, TEST_SIZE, VAL_SIZE)


def download_raw(force: bool = False) -> Path:
    """Скачивает архив с UCI и извлекает файл SMSSpamCollection в data/raw/."""
    if RAW_FILE.exists() and not force:
        print(f"Файл уже есть: {RAW_FILE}")
        return RAW_FILE

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю датасет: {DATA_URL}")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            archive = response.read()
    except Exception as err:  # нет интернета, сайт недоступен и т.п.
        raise RuntimeError(
            "Не удалось скачать датасет автоматически. Скачайте архив вручную по ссылке\n"
            f"  {DATA_URL}\n"
            f"распакуйте и положите файл SMSSpamCollection в папку {RAW_DIR}"
        ) from err

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        with zf.open("SMSSpamCollection") as src, open(RAW_FILE, "wb") as dst:
            dst.write(src.read())
    print(f"Сохранено: {RAW_FILE}")
    return RAW_FILE


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Читает исходный файл: каждая строка = 'метка<TAB>текст'.

    quoting=QUOTE_NONE важен: в SMS встречаются кавычки, и без этого параметра
    pandas склеивает несколько сообщений в одно и строки теряются.
    """
    return pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                       quoting=csv.QUOTE_NONE, encoding="utf-8",
                       encoding_errors="replace", dtype=str)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Минимальная очистка: убираем пустые строки и некорректные метки.

    Текст намеренно НЕ чистим агрессивно (не удаляем цифры, ссылки, символы):
    номера телефонов, суммы и URL — сильные признаки спама.
    """
    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].str.strip().str.lower()
    df["text"] = df["text"].str.strip()
    df = df[(df["text"] != "") & (df["label"].isin(["ham", "spam"]))]
    df["y"] = (df["label"] == "spam").astype(int)
    return df.reset_index(drop=True)


def _normalize(text: str) -> str:
    """Ключ для поиска дубликатов: нижний регистр + схлопнутые пробелы."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Удаляет повторяющиеся сообщения ДО разбиения на выборки.

    Если одно и то же SMS попадёт и в train, и в test, модель «узнает» его,
    и метрики на test будут завышены (data leakage).
    """
    key = df["text"].map(_normalize)
    conflicting = int(df.assign(key=key).groupby("key")["y"].nunique().gt(1).sum())
    deduped = df.loc[~key.duplicated()].reset_index(drop=True)
    stats = {
        "rows_before_dedup": int(len(df)),
        "duplicates_removed": int(len(df) - len(deduped)),
        "texts_with_conflicting_labels": conflicting,
    }
    return deduped, stats


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Стратифицированное разбиение 70/15/15 (доля спама одинакова во всех выборках)."""
    train_val, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["y"],
                                       random_state=RANDOM_STATE)
    val_share = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(train_val, test_size=val_share, stratify=train_val["y"],
                                  random_state=RANDOM_STATE)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает уже подготовленные train/val/test (сначала запустите python -m src.data)."""
    out = []
    for name in ("train", "val", "test"):
        path = PROCESSED_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Нет {path}. Сначала выполните: python -m src.data")
        out.append(pd.read_csv(path, keep_default_na=False))
    return tuple(out)


def main() -> None:
    download_raw()
    raw = load_raw()
    df = clean(raw)
    df, dedup_stats = deduplicate(df)
    train, val, test = split(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["label", "y", "text"]
    for name, part in (("train", train), ("val", val), ("test", test)):
        part[cols].to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    summary = {
        "raw_rows": int(len(raw)),
        "rows_after_cleaning": dedup_stats["rows_before_dedup"],
        **dedup_stats,
        "rows_final": int(len(df)),
        "spam_share_final": round(float(df["y"].mean()), 4),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "spam_share_by_split": {n: round(float(p["y"].mean()), 4)
                                for n, p in (("train", train), ("val", val), ("test", test))},
        "random_state": RANDOM_STATE,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
