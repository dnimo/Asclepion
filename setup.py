#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏥 Kallipolis医疗共和国治理系统 (Asclepion) - 安装配置

基于多智能体博弈论的医院治理实时监控仿真平台
"""

from setuptools import setup, find_packages
import os
import re

# 读取版本号
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "src", "hospital_governance", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r", encoding="utf-8") as f:
            content = f.read()
            version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
            if version_match:
                return version_match.group(1)
    return "0.1.0"

# 读取README
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return "Kallipolis医疗共和国治理系统 - 基于多智能体博弈论的医院治理仿真平台"

# 读取requirements.txt，过滤注释和空行
def get_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    requirements = []
    
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                # 跳过注释、空行和可选依赖
                if line and not line.startswith('#') and not line.startswith('-'):
                    # 移除行内注释
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if line:
                        requirements.append(line)
    
    # 核心必需依赖（确保最小可用配置）
    core_requirements = [
        "numpy>=1.21.0,<2.0.0",
        "scipy>=1.7.0,<2.0.0", 
        "pandas>=1.3.0,<3.0.0",
        "matplotlib>=3.5.0,<4.0.0",
        "pyyaml>=6.0,<7.0.0",
        "websockets>=10.0,<12.0",
        "aiohttp>=3.8.0,<4.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "plotly>=5.0.0,<6.0.0",
        "loguru>=0.6.0,<1.0.0",
        "psutil>=5.8.0,<6.0.0",
    ]
    
    return requirements if requirements else core_requirements

# 可选依赖组
extras_require = {
    "ai": [
        "torch>=1.12.0,<3.0.0",
        "stable-baselines3>=1.6.0",
        "transformers>=4.20.0,<5.0.0",
        "openai>=1.0.0,<2.0.0",
    ],
    "dev": [
        "pytest>=6.0.0,<8.0.0",
        "pytest-asyncio>=0.20.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0,<24.0.0",
        "isort>=5.10.0,<6.0.0",
        "flake8>=4.0.0,<7.0.0",
        "mypy>=0.910,<2.0.0",
        "pre-commit>=2.15.0,<4.0.0",
    ],
    "docs": [
        "sphinx>=4.0.0,<8.0.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0,<2.0.0",
        "ipywidgets>=7.6.0,<9.0.0",
        "notebook>=6.4.0,<8.0.0",
    ],
    "full": [
        "dash>=2.0.0,<3.0.0",
        "flask>=2.0.0,<3.0.0",
        "sqlalchemy>=1.4.0,<3.0.0",
        "nltk>=3.7,<4.0.0",
        "spacy>=3.4.0,<4.0.0",
    ]
}

# 所有可选依赖
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="kallipolis-asclepion",
    version=get_version(),
    author="dnimo",
    author_email="dnimo@example.com",
    description="🏥 Kallipolis医疗共和国治理系统 - 基于多智能体博弈论的医院治理仿真平台",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/dnimo/Asclepion",
    project_urls={
        "Bug Tracker": "https://github.com/dnimo/Asclepion/issues",
        "Documentation": "https://github.com/dnimo/Asclepion/docs",
        "Source Code": "https://github.com/dnimo/Asclepion",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "hospital_governance": [
            "config/*.yaml",
            "config/*.yml",
            "frontend/*.html",
            "frontend/*.css",
            "frontend/*.js",
        ]
    },
    python_requires=">=3.8,<4.0",
    install_requires=get_requirements(),
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "kallipolis-server=hospital_governance.main:main",
            "kallipolis-demo=hospital_governance.examples.demo:main",
            "kallipolis-export=hospital_governance.tools.export:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Web Environment",
        "Framework :: AsyncIO",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
    ],
    keywords=[
        "医疗治理", "多智能体系统", "博弈论", "强化学习", "WebSocket", 
        "仿真平台", "人工智能", "医院管理", "决策支持", "实时监控",
        "healthcare", "multi-agent", "game-theory", "reinforcement-learning",
        "simulation", "artificial-intelligence", "hospital-management"
    ],
    zip_safe=False,
    test_suite="tests",
    tests_require=[
        "pytest>=6.0.0",
        "pytest-asyncio>=0.20.0",
        "pytest-cov>=4.0.0",
    ],
)