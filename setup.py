#!/usr/bin/env python3
"""
HGVCS Setup Script

Installation and setup script for the Hand Gesture & Voice Control System.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="hgvcs",
    version="1.0.0",
    author="HGVCS Team",
    author_email="contact@hgvcs.io",
    description="Hand Gesture & Voice Control System - Next-Generation HCI Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hgvcs",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
            "sphinx>=6.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hgvcs=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "*.yaml",
            "*.json",
            "*.txt",
            "*.md",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hgvcs/issues",
        "Source": "https://github.com/yourusername/hgvcs",
        "Documentation": "https://docs.hgvcs.io",
    },
)
