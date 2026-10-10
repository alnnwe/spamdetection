"""Общие пути и константы проекта. Все скрипты берут настройки отсюда."""
from pathlib import Path

# Корень репозитория
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"

# Открытый датасет
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_FILE = RAW_DIR / "SMSSpamCollection"

# Фиксированный seed
RANDOM_STATE = 42

# Доли выборок
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Положительный класс для метрик - спам
POS_LABEL = 1
LABEL_NAMES = {0: "ham", 1: "spam"}
