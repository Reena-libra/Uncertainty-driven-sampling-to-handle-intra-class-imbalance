# Uncertainty-driven-sampling-to-handle-intra-class-imbalance


This repository contains the code and experimental framework for our paper:

**"Uncertainty-Driven Sampling for Handling Intra-Class Variability in 3D Plant Part Segmentation"**
Please cite the paper
@INPROCEEDINGS{10924512,

  author={Reena and Doonan, John H. and Zhang, Huaizhong and Liu, Yonghuai and Williams, Kevin and Corke, Fiona M. K.},
  
  booktitle={2025 IEEE 6th International Conference on Image Processing, Applications and Systems (IPAS)}, 
  
  title={Uncertainty Driven Sampling to Handle Intra-class Imbalance Part Segmentation in Wheat}, 
  
  year={2025},
  
  volume={CFP2540Z-ART},
  
  pages={1-7},
  
  keywords={Training;Solid modeling;Uncertainty;Three-dimensional displays;Accuracy;Shape;Focusing;Ear;Robustness;Resilience;Imbalance;Uncertainty;Part-segmentation;Wheat},
  
  doi={10.1109/IPAS63548.2025.10924512}}



---

## 🧠 Overview

This work proposes a novel training pipeline that incorporates **uncertainty estimation** to improve segmentation performance on **3D point cloud data** of wheat plants. By identifying uncertain samples using **Monte Carlo (MC) Dropout**, we iteratively refine model learning using the most informative examples.

### 🔬 Key Highlights

- Applies **MC Dropout** during training to estimate epistemic uncertainty per sample.
- Reintroduces high-uncertainty samples for **resampling and focused learning**.
- Evaluated on **Wheat3D PartNet** – a challenging dataset with high intra-class variation.
- Demonstrates improved mIoU and class-wise segmentation performance.

---



