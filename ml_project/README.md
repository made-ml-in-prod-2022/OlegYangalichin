ml_project
==============================

Homework_01 ML in production

For train classification model using dataset from https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

**Installation**:
~~~
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

**Usage**:

**Run full pipeline**:
```shell script
python ml_project/train_and_predict_pipeline.py run_pipeline --config_path CONFIG_PATH --save_model SAVE_MODEL --save_results SAVE_RESULTS --model_path MODEL_PATH
```
for example:
```shell script
python ml_project/train_and_predict_pipeline.py run_pipeline --config_path configs/logreg_train_config.yaml --save_model False --save_results False 
```

**Train and save model**:
```shell script
python ml_project/train_and_predict_pipeline.py train_and_save --config_path CONFIG_PATH 
```
for example:
```shell script
python ml_project/train_and_predict_pipeline.py train_and_save --config_path configs/logreg_train_config.yaml
```

**Load and Predict**:
```shell script
python ml_project/train_and_predict_pipeline.py load_and_predict --config_path CONFIG_PATH --model_path MODEL_PATH --save_results SAVE_RESULTS 
```
for example:
```shell script
python ml_project/train_and_predict_pipeline.py load_and_predict --config_path configs/logreg_train_config.yaml --model_path MODEL_PATH models/lr_model.pkl
```

**Test**:
```shell script
pytest tests/
```


Dataset can load with link https://drive.google.com/file/d/1PiBD7lFKmGgX8fuaaeN-Q8_kXhaeEErO/view?usp=sharing

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │      └── heart_cleveland_upload.csv
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── train_and_predict_pipeline.py  <- CLI Pipeline
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ml_example                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── __init__.py
    │   │   ├── prepare_data.py
    │   │   └── read_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── __init__.py
    │   │   └── feature_extraction.py
    │   │
    │   ├── models         <- Model wrapper class
    │   │   ├── __init__.py
    │   │   └── model.py
    │   │
    │   └── params  <- dataclasses for configs
    │   │   ├── __init__.py
    │   │   ├── data_split_params.py
    │   │   ├── data_split_params.py
    │   │   ├── pipeline_config.py
    │   │   └── training_params.py
    └── 

Self-estimation
------------

Критерии (указаны максимальные баллы, по каждому критерию ревьюер может поставить баллы частично):

(Сделано) В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание того, что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код (1 балл)

(Сделано в README) В пулл-реквесте проведена самооценка, распишите по каждому пункту выполнен ли критерий или нет и на сколько баллов(частично или полностью) (1 балл)

(Сделано) Выполнено EDA, закоммитьте ноутбук в папку с ноутбуками (1 балл) Вы так же можете построить в ноутбуке прототип(если это вписывается в ваш стиль работы)

(Не сделано) Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (за это + 1 балл)

(Сделано) Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкцию по запуску (3 балла)

(Сделано) Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (3 балла)

(Сделано) Проект имеет модульную структуру (2 балла)

(Сделано) Использованы логгеры (2 балла)

(Сделано) Написаны тесты на отдельные модули и на прогон обучения и predict (3 балла)

(Сделано) Для тестов генерируются синтетические данные, приближенные к реальным (2 балла)

(Сделано) Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)

(Сделано) Используются датаклассы для сущностей из конфига, а не голые dict (2 балла)

(Сделано) Напишите кастомный трансформер и протестируйте его (3 балла) https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

(Сделано) В проекте зафиксированы все зависимости (1 балл)

(Сделано) Настроен CI для прогона тестов, линтера на основе github actions (3 балла). Пример с пары: https://github.com/demo-ml-cicd/ml-python-package

Дополнительные баллы=)

(Не сделано) Используйте hydra для конфигурирования (https://hydra.cc/docs/intro/) - 3 балла

(Не сделано) Mlflow
