#!/usr/bin/env python3

"""
Setup script for GLM-4.1V-Thinking Model Package

This package contains the extracted GLM-4V multimodal model implementation
with all necessary dependencies from the HuggingFace Transformers library.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="glm4v-thinking-model",
    version="1.0.0",
    author="ZhipuAI Inc. team and HuggingFace Inc. team",
    author_email="",
    description="GLM-4.1V multimodal vision-language model implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/THUDM/GLM-4.1V-Thinking",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    keywords="machine learning, artificial intelligence, vision language model, multimodal, GLM-4V",
    include_package_data=True,
    zip_safe=False,
) 