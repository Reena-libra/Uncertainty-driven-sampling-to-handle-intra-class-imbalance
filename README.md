# Uncertainty-driven-sampling-to-handle-intra-class-imbalance


This repository contains the code and experimental framework for our paper:

**"Uncertainty-Driven Sampling for Handling Intra-Class Variability in 3D Plant Part Segmentation"**



---

## 🧠 Overview

This work proposes a novel training pipeline that incorporates **uncertainty estimation** to improve segmentation performance on **3D point cloud data** of wheat plants. By identifying uncertain samples using **Monte Carlo (MC) Dropout**, we iteratively refine model learning using the most informative examples.

### 🔬 Key Highlights

- Applies **MC Dropout** during training to estimate epistemic uncertainty per sample.
- Reintroduces high-uncertainty samples for **resampling and focused learning**.
- Evaluated on **Wheat3D PartNet** – a challenging dataset with high intra-class variation.
- Demonstrates improved mIoU and class-wise segmentation performance.

---

## 📁 Project Structure

