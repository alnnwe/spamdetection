"""Простая демонстрация: классификация SMS из командной строки.

Примеры:
  python -m src.predict "WINNER!! You have won a 1000 prize. Call 09061701461 now"
  python -m src.predict "Hey, are we still meeting at 6?" "Free entry to win a car, text WIN to 80086"
  python -m src.predict            # интерактивный режим: вводите сообщения по одному
"""
import sys

import joblib

from src.config import MODELS_DIR

THRESHOLD = 0.5  # порог; подбор порога — задача финальной части проекта


def load_model():
    path = MODELS_DIR / "baseline.joblib"
    if not path.exists():
        sys.exit("Модель не найдена. Сначала выполните: python -m src.data && python -m src.train_baseline")
    return joblib.load(path)


def classify(model, text: str) -> tuple[str, float]:
    proba = float(model.predict_proba([text])[0, 1])
    return ("SPAM" if proba >= THRESHOLD else "HAM"), proba


def main() -> None:
    model = load_model()
    texts = sys.argv[1:]
    if texts:
        for t in texts:
            label, p = classify(model, t)
            print(f"[{label}] p(spam)={p:.3f} | {t}")
        return

    print("Введите SMS (пустая строка — выход):")
    while True:
        t = input("> ").strip()
        if not t:
            break
        label, p = classify(model, t)
        print(f"  -> {label} (вероятность спама {p:.1%})")


if __name__ == "__main__":
    main()
