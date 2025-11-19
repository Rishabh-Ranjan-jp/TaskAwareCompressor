# Quick Start Guide for TaskAwareCompressor

These instructions assume you already have **Anaconda installed**.  
If not, download it here: [Anaconda Individual Edition](https://www.anaconda.com/products/distribution)

No need to touch system Python or worry about versions. This guide sets up a **self-contained Python 3.10 environment** with all dependencies for running the project in **Jupyter Notebook**.

---

## 1. Create the Project Environment

Open **Anaconda Navigator** or **Anaconda Prompt** and run:

```bash
# Create a new environment with Python 3.10
conda create -n miniproject python=3.10 -y

# Activate it
conda activate miniproject
````

> This ensures everyone uses **Python 3.10**, which is fully compatible with PyTorch and all dependencies.

---

## 2. Install Required Packages

Copy-paste these commands **one by one**:

```bash
# Core data and visualization packages
conda install numpy pandas matplotlib seaborn scikit-learn jupyter -c conda-forge -y
```

### Install PyTorch

```bash
# CPU-only (works on any machine)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

```bash
# GPU-enabled (NVIDIA GPU only)
# Make sure to verify that the CUDA version is compatible with your GPU
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

```bash
# Make the environment available in Jupyter
conda install ipykernel -c conda-forge -y
python -m ipykernel install --user --name=miniproject --display-name "Python (miniproject)"
```

---

## 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

1. Navigate to the project folder in the browser interface.
2. Open any notebook (`.ipynb`) and select the **Python (miniproject)** kernel.
3. Run cells in order — that’s it, no IDE setup required.

---
