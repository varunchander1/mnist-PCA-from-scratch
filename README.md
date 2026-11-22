# PCA from First Principles on MNIST

**Eigen decomposition, dimensionality reduction and reconstruction quality (PSNR) on handwritten digits, implemented by hand in NumPy and cross-checked against scikit-learn.**

Part of the *AI & ML for Industry* programme, IIT Delhi (Module 1: Mathematical Foundations for AI/ML).

---

## Overview

MNIST digits are 28×28 images, so each one is a 784-dimensional vector, but most of those pixels are redundant. This project builds Principal Component Analysis from the ground up to show how few dimensions are actually needed:

1. **Eigen decomposition:** compute the sample covariance matrix of the pixel features, decompose it with `np.linalg.eigh`, and plot cumulative explained variance against the number of components.
2. **Dimensionality reduction:** project the data onto the top *p* eigenvectors for p ∈ {50, 250, 300, 500, 600, 700}.
3. **Reconstruction:** map the reduced data back to pixel space using the orthonormality of the eigenvector matrix (X̂ = Z·Vₚᵀ).
4. **Error measurement:** compare originals and reconstructions with Peak Signal-to-Noise Ratio (PSNR).

The manual implementation is then repeated with scikit-learn's `PCA` to validate it.

## Results

Mean PSNR between original and reconstructed images (higher is better):

| Components (p) | Manual eigen decomposition | scikit-learn PCA |
|---:|---:|---:|
| 50 | 18.8 dB | 17.4 dB |
| 250 | 27.0 dB | 22.8 dB |
| 300 | 29.5 dB | 23.9 dB |
| 500 | 56.2 dB | 28.1 dB |
| 600 | 84.0 dB | 31.8 dB |

**Key findings**
- Explained variance rises steeply over the first few dozen components and then flattens: digit images are highly structured, not random noise.
- At p = 50, reconstructions keep only the coarse stroke shape. By p ≈ 500–600 they are visually indistinguishable from the originals.
- Reconstruction quality improves steadily with p in both implementations.

## Method notes

- **Data:** 42,000 labelled images from the Kaggle MNIST training CSV (label + 784 pixel columns).
- **Scaling:** pixels are standardised with `StandardScaler` (zero mean, unit variance) and centred before computing the covariance matrix. Reconstructions are mapped back to pixel space with the scaler's inverse transform.
- **Eigen solver:** `np.linalg.eigh`, since the covariance matrix is real and symmetric. Eigenpairs are sorted by descending eigenvalue.

## Repository contents

| File | Description |
|---|---|
| `MNIST_PCA_Eigen_Decomposition.ipynb` | Full implementation: loading, eigen decomposition, PCA, reconstruction, PSNR |
| `MNIST_PCA_Eigen_Decomposition_Report.pdf` | Written report with method, figures and discussion |
| `requirements.txt` | Python dependencies |

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook MNIST_PCA_Eigen_Decomposition.ipynb
```

**Data:** download `train.csv` from Kaggle's [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data) competition, save it in this folder as `mnist_train.csv`, and point the `read_csv` path in the first code cell at it. The dataset is not included in this repo because of its size (77 MB).

## Tech stack

Python · NumPy · pandas · scikit-learn · Matplotlib

---

**Author:** Varun Chander · [GitHub](https://github.com/varunchander1)
