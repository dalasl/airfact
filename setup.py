"""
Terminal Data Leakage Detection System
Based on User Data Feature Profiling

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="dlp-detection",
    version="1.0.0",
    description="Terminal Data Leakage Detection Based on User Data Feature Profiling",
    author="HUST Cyberspace Security Lab",
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "z3-solver>=4.12.0",
        "pymupdf>=1.23.0",
        "python-docx>=1.1.0",
        "openpyxl>=3.1.2",
        "Pillow>=10.0.0",
        "grpcio>=1.59.0",
        "protobuf>=4.25.0",
    ],
    extras_require={
        "gpu": [
            "faiss-gpu>=1.7.4",
            "auto-gptq>=0.6.0",
            "bitsandbytes>=0.41.0",
        ],
        "api": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "websockets>=12.0",
        ],
        "velociraptor": [
            "pyvelociraptor>=0.1.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dlp-detection=main:main",
        ],
    },
)
