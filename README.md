# SMS Spam Detection — определение спам-SMS

Задача: бинарная классификация SMS-сообщений на **spam** и **ham** (обычные сообщения) методами машинного обучения.

> **Статус: промежуточная защита (≈50%).** Готовы постановка задачи, подготовка данных и EDA,
> baseline-модели и первые эксперименты на validation-выборке. Улучшение модели, финальная
> оценка на test, веб-демо и полный анализ рисков — во второй половине (см. [Roadmap](#roadmap)).

## Проблема

Спам и мошеннические SMS (smishing) — массовая проблема: ссылки на фишинговые сайты,
«выигрыши», платные подписки. Правила по ключевым словам легко обходятся небольшим изменением
текста, поэтому мы обучаем модель, которая выявляет закономерности по данным.

**Пользователи:** абоненты мобильной связи, операторы, разработчики SMS-фильтров.
**Критерии успеха (цели на финал):** F1(spam) ≥ 0.90 и precision(spam) ≥ 0.95 на отложенной
test-выборке (ложная блокировка нормального SMS для пользователя хуже, чем пропущенный спам).

## Данные

- **Датасет:** [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (UCI ML Repository), лицензия CC BY 4.0
- **Объём:** 5 574 англоязычных SMS, метки `ham` / `spam`, классы несбалансированы (~13% спама)
- **Источник:** Almeida T.A., Gómez Hidalgo J.M., Yamakami A. *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* DocEng 2011
- Данные в репозитории не хранятся — скачиваются скриптом `python -m src.data`

**Предобработка:** удаление пустых строк → удаление дубликатов (до разбиения, против data leakage) →
стратифицированное разбиение 70/15/15 (`random_state=42`). Цифры, ссылки и символы валют
сохраняются, так как это сильные признаки спама.

## Структура репозитория

```
sms-spam-detection/
├── data/
│   ├── raw/                  # исходный файл SMSSpamCollection (скачивается)
│   └── processed/            # train.csv / val.csv / test.csv
├── notebooks/
│   └── 01_eda_and_data_preparation.ipynb   # EDA и обоснование предобработки
├── src/
│   ├── config.py             # пути, seed, доли выборок
│   ├── data.py               # загрузка, очистка, дубликаты, разбиение
│   ├── train_baseline.py     # baseline-модели + метрики на validation
│   └── predict.py            # демо: классификация SMS из консоли
├── models/                   # сохранённая модель (создаётся при обучении)
├── results/                  # метрики, ошибки модели, сводка по данным
├── reports/
│   ├── midterm_report.md     # промежуточный отчёт
│   └── figures/              # графики EDA и экспериментов
├── docs/roles.md             # распределение ролей и вклад участников
├── requirements.txt
└── README.md
```

## Как запустить

```bash
# 1. Окружение (Python 3.11+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Данные: скачать, очистить, разбить на train/val/test
python -m src.data

# 3. Обучить baseline-модели и получить метрики на validation
python -m src.train_baseline

# 4. Проверить модель на своих сообщениях
python -m src.predict "WINNER!! You have won a prize. Call 09061701461 now"
python -m src.predict            # интерактивный режим

# 5. EDA
jupyter lab notebooks/01_eda_and_data_preparation.ipynb
```

Если автоматическая загрузка не работает, скачайте архив по ссылке из `src/config.py`,
распакуйте и положите файл `SMSSpamCollection` в `data/raw/`.

## Методы (промежуточный этап)

| Модель | Назначение |
|---|---|
| Majority class (DummyClassifier) | Нижняя граница: всегда отвечает «ham» — показывает, что accuracy при дисбалансе обманчива |
| TF-IDF + Multinomial Naive Bayes | Классический baseline для классификации текстов |
| TF-IDF + Logistic Regression (`class_weight="balanced"`) | Линейная модель с учётом дисбаланса, интерпретируемые веса слов |

**Метрики:** precision, recall, F1 для класса spam и PR-AUC (accuracy — справочно).
Все сравнения делаются на **validation**; test-выборка отложена до финала.

**Результаты на validation (774 сообщения, 96 спам):**

| model | accuracy | precision_spam | recall_spam | f1_spam | pr_auc |
|---|---|---|---|---|---|
| Majority (Dummy) | 0.876 | 0.000 | 0.000 | 0.000 | 0.124 |
| TF-IDF + MultinomialNB | 0.964 | 1.000 | 0.708 | 0.829 | 0.957 |
| TF-IDF + LogisticRegression | 0.975 | 0.874 | 0.938 | 0.905 | 0.947 |


