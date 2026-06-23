from setuptools import setup, find_packages

setup(
    name="opcompass",
    version="0.1.0",
    description="SOL (Speed of Light) theoretical peak performance estimator for GPU operators",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="OpCompass Authors",
    python_requires=">=3.8",
    packages=find_packages(include=["opcompass", "opcompass.*"]),
    install_requires=[
        "click>=8.0",
        "fastapi>=0.100",
        "uvicorn>=0.23",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "compass=opcompass.cli:main",
        ],
    },
)
