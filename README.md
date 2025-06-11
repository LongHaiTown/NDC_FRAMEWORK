# Cipher Analysis Toolkit

A Python toolkit for cipher analysis with CPU and GPU support, plus tools for data generation, model management, and quality evaluation.


## Installation

We **highly recommend** using [Conda](https://docs.conda.io/) to handle the environment and dependencies.

### 1. Create and Activate Conda Environment

```bash
conda create -n cipher-env python=3.10
conda activate cipher-env
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the Project in Development Mode

```bash
pip install -e .

```

## Usage
You can now run project scripts directly from the root folder:

```bash
python run.py # configure .yaml file (not yet)
```

Or execute modules and tests inside src/ (e.g., from tests/):
```bash
python src/tests/cpu_vs_gpu.py
```
