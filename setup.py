from setuptools import setup, find_packages

setup(
    name="llama_fine_tuning",
    version="0.1",
    packages=find_packages(),
    description="Fine-tuning Llama 3.2 models with unsloth",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=[
        "torch",
        "unsloth",
        "transformers",
        "datasets",
        "evaluate",
        "rouge_score",
        "pandas",
        "scikit-learn",
    ],
)