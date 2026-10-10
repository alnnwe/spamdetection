# Распределение ролей и вклад участников

Команда: **Mukanova Alana**, **Kairanbay Arailym**


| Участник | Роль | Что сделано (мидтерм) | Файлы | План на финал |
|---|---|---|---|---|
| Mukanova Alana | Данные, EDA, инженерия | Загрузка и очистка данных, удаление дубликатов и проверка на data leakage, стратифицированное разбиение train/val/test, EDA (баланс классов, длина сообщений, признаки спама, частые слова, артефакт gt/lt), настройка окружения, структура репозитория, README, requirements, GitHub | `src/config.py`, `src/data.py`, `notebooks/01_eda_and_data_preparation.ipynb`, `README.md`, `requirements.txt` | Новые признаки: токены `<PHONE>`, `<MONEY>`, `<URL>`, декодирование HTML-сущностей; веб-демо (Streamlit); раздел Responsible AI |
| Kairanbay Arailym | Модели, эксперименты, анализ | Выбор метрик (precision/recall/F1 для spam, PR-AUC), три baseline-модели, сравнение на validation, confusion matrix и PR-кривые, анализ ошибок (19 ошибок: FP = 13, FN = 6), консольное демо, промежуточный отчёт | `src/train_baseline.py`, `src/predict.py`, `reports/midterm_report.md`, `results/` | Биграммы и символьные n-граммы, подбор гиперпараметров и порога классификации, сравнение с трансформером, learning curves, финальная оценка на test |
