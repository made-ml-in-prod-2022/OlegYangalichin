from setuptools import find_packages, setup

setup(
    name='ml_project',
    packages=find_packages(),
    version='0.1.0',
    description='HW01',
    author='Yangalichin Oleg MADE-DS-22',
    entry_points={
        "console_scripts": [
            "pipeline = train_and_predict_pipeline:run_pipeline",
            "train = train_and_predict_pipeline:train_and_save",
            "predict = train_and_predict_pipeline:load_and_predict",
        ]
    },
    license='MIT',
)