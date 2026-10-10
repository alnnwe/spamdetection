"""Обучение baseline-моделей и их сравнение на VALIDATION-выборке.

Запуск:  python -m src.train_baseline

Test-выборка здесь НЕ используется: она отложена до финальной защиты,
чтобы подбор моделей не «подглядывал» в неё.

Результаты:
  results/val_metrics.csv, results/val_metrics.md   — таблица метрик
  results/val_errors_<модель>.csv                   — ошибки лучшей модели (для error analysis)
  reports/figures/*.png                             — confusion matrix, PR-кривые, топ-признаки
  models/baseline.joblib                            — лучшая модель (для src/predict.py)
"""
import re

import joblib
import matplotlib

matplotlib.use("Agg")  # сохраняем графики в файлы без открытия окон
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, accuracy_score,
                             average_precision_score, f1_score, precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, LABEL_NAMES, MODELS_DIR, RANDOM_STATE, RESULTS_DIR
from src.data import load_splits


def build_models() -> dict:
    """Три уровня сравнения: «ничего не делаем» -> классика NB -> линейная модель."""
    return {
        # Всегда предсказывает самый частый класс (ham). Показывает, почему accuracy обманчива.
        "Majority (Dummy)": DummyClassifier(strategy="most_frequent"),
        # Классический baseline для текстов: мешок слов + наивный Байес
        "TF-IDF + MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True)),
            ("clf", MultinomialNB()),
        ]),
        # Линейная модель с учётом дисбаланса классов
        "TF-IDF + LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(model, X, y) -> tuple[dict, np.ndarray, np.ndarray]:
    """Метрики для класса spam (положительный класс)."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision_spam": precision_score(y, pred, zero_division=0),
        "recall_spam": recall_score(y, pred, zero_division=0),
        "f1_spam": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, proba),
    }
    return metrics, pred, proba


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_markdown_table(df: pd.DataFrame, path) -> None:
    header = "| model | " + " | ".join(df.columns) + " |"
    sep = "|---|" + "---|" * len(df.columns)
    rows = [f"| {idx} | " + " | ".join(f"{v:.4f}" for v in row) + " |"
            for idx, row in df.iterrows()]
    path.write_text("\n".join([header, sep, *rows]) + "\n", encoding="utf-8")


def plot_top_features(pipeline: Pipeline, path, k: int = 15) -> None:
    """Слова с наибольшим весом в сторону spam и ham (интерпретируемость логрегрессии)."""
    vocab = np.array(pipeline.named_steps["tfidf"].get_feature_names_out())
    coefs = pipeline.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    idx = np.concatenate([order[:k], order[-k:]])
    colors = ["tab:blue"] * k + ["tab:red"] * k
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(vocab[idx], coefs[idx], color=colors)
    ax.set_title("LogisticRegression: топ слов (синие → ham, красные → spam)")
    ax.set_xlabel("Вес признака")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    train, val, _ = load_splits()  # test не трогаем
    X_train, y_train = train["text"], train["y"]
    X_val, y_val = val["text"], val["y"]

    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    models = build_models()
    rows, preds, probas = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        rows[name], preds[name], probas[name] = evaluate(model, X_val, y_val)
        print(f"{name:32s} " + "  ".join(f"{k}={v:.3f}" for k, v in rows[name].items()))

    table = pd.DataFrame(rows).T
    table.to_csv(RESULTS_DIR / "val_metrics.csv")
    save_markdown_table(table, RESULTS_DIR / "val_metrics.md")

    # Confusion matrix для каждой модели
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for ax, name in zip(axes, models):
        ConfusionMatrixDisplay.from_predictions(
            y_val, preds[name], display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]],
            ax=ax, colorbar=False)
        ax.set_title(name, fontsize=10)
    fig.suptitle("Confusion matrix (validation)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices_val.png", dpi=150)
    plt.close(fig)

    # Precision-Recall кривые (важнее ROC при дисбалансе классов)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in models:
        PrecisionRecallDisplay.from_predictions(y_val, probas[name], name=name, ax=ax)
    ax.set_title("Precision-Recall (validation), положительный класс = spam")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curves_val.png", dpi=150)
    plt.close(fig)

    # Лучшая модель по F1 для спама -> сохраняем и выгружаем её ошибки
    best = table["f1_spam"].idxmax()
    best_model = models[best]
    joblib.dump(best_model, MODELS_DIR / "baseline.joblib")

    errors = val.assign(pred=preds[best], spam_proba=probas[best].round(4))
    errors = errors[errors["y"] != errors["pred"]].copy()
    errors["error_type"] = np.where(errors["y"] == 1, "FN (спам пропущен)",
                                    "FP (обычное SMS помечено как спам)")
    errors[["error_type", "spam_proba", "label", "text"]].to_csv(
        RESULTS_DIR / f"val_errors_{slug(best)}.csv", index=False)

    lr_name = "TF-IDF + LogisticRegression"
    plot_top_features(models[lr_name], FIGURES_DIR / "top_features_logreg.png")

    print(f"\nЛучшая модель по F1(spam) на validation: {best}")
    print(f"Ошибок на validation: {len(errors)} "
          f"(FN={int((errors['y'] == 1).sum())}, FP={int((errors['y'] == 0).sum())})")
    print(f"Модель сохранена: {MODELS_DIR / 'baseline.joblib'}")


if __name__ == "__main__":
    main()
