"""Общие пути и константы проекта. Все скрипты берут настройки отсюда."""
from pathlib import Path

# Корень репозитория (папка, в которой лежит src/)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"

# Открытый датасет: SMS Spam Collection (UCI Machine Learning Repository), лицензия CC BY 4.0
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_FILE = RAW_DIR / "SMSSpamCollection"

# Фиксированный seed -> одинаковое разбиение и результаты при каждом запуске
RANDOM_STATE = 42

# Доли выборок: 70% train / 15% validation / 15% test
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Положительный класс для метрик — спам
POS_LABEL = 1
LABEL_NAMES = {0: "ham", 1: "spam"}
