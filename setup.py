"""
Setup script for Grounded Attention project
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="grounded-attention",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Anti-Hallucination Transformer Layer for Vision-Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/grounded-attention",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "peft>=0.7.0",
        "timm>=0.9.10",
        "einops>=0.7.0",
        "datasets>=2.15.0",
        "Pillow>=10.0.0",
        "pycocotools>=2.0.7",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "black>=23.11.0",
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "ipython>=8.18.0",
        ],
        "quantization": [
            "bitsandbytes>=0.41.0",
        ],
        "logging": [
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
    },
)
