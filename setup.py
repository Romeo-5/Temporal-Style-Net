"""
Setup script for TemporalStyleNet
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith('#')]

setup(
    name="temporal-style-net",
    version="0.1.0",
    author="Romeo Nickel",
    author_email="your.email@example.com",
    description="Real-time video style transfer with temporal consistency",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Romeo-5/temporal-style-net",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
        "full": [
            "diffusers>=0.21.0",
            "transformers>=4.33.0",
            "accelerate>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "temporal-style-transfer=scripts.inference:main",
        ],
    },
)