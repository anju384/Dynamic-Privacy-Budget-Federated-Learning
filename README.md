# Dynamic Privacy Budget Allocation (DPBA) in Federated Learning

An implementation of decentralized deep learning on image and tabular data using **Federated Learning (FL)** integrated with **Differential Privacy (DP)**. This repository explores a **Dynamic Privacy Budget Allocation (DPBA)** algorithm that optimizes the privacy-utility trade-off compared to conventional fixed privacy budget configurations.

## 📌 Project Overview
Traditional machine learning methods often require centralized data collection, creating single points of failure for sensitive data. While Federated Learning reduces this risk by training models locally on decentralized client devices, sensitive user parameters can still leak via captured model gradients. 

This project integrates Differential Privacy by adding controlled noise to gradients during training rounds. Instead of applying a uniform uniform privacy budget ($\epsilon$), the implemented **DPBA** mechanism dynamically allocates budgets to clients based on data sensitivity evaluation, client participation frequency, and local contribution to model convergence.

---

## 🚀 Key Features
- **Multi-Dataset Benchmarking:** Evaluated across **MNIST** (Handwritten digits using FCNN), **CIFAR-10** (Complex RGB Image Classification using a **Convolutional Neural Network (CNN)**), and the highly-imbalanced **Bank Account Fraud (BAF)** tabular dataset.
- **Dynamic Optimization:** Leverages a custom sensitivity assessment loop utilizing gradient-based layer importance calculations.
- **Class Imbalance Management:** Mitigates severe tabular class skew via weighted random samplers and modified loss-function adjustments.
- **Privacy Attack Verification:** Validates privacy leakage preservation scores directly using an adversarial **Membership Inference Attack (MIA)** pipeline.

---

## 📊 Performance Summary

### 1. MNIST Dataset (10 Clients, 10 Rounds)
- **DPBA Approach:** Reached an accuracy of **99.05%** by round 10.
- **Fixed Privacy Budget:** Reached an accuracy of **98.73%** by round 10.

### 2. CIFAR-10 Dataset (CNN Architecture)
- Implemented a multi-layer **Convolutional Neural Network (CNN)** to extract spatial features from 3-channel color images under decentralized constraints.
- DPBA dynamically scaled the privacy budget ($\epsilon$) across convolutional layers based on feature map sensitivity, matching baseline performance while minimizing parameters exposed to reconstruction risks.

### 3. Tabular Bank Account Fraud (BAF) Dataset
- Achieved a stable fraud classification **Recall rate of ~80%** while maintaining precision.
- **MIA Privacy Evaluation:** Under DPBA tracking, the Membership Inference Attack accuracy was significantly minimized compared to the Fixed Budget, illustrating a lower data leakage footprint.

---

## 📁 Repository Structure
- `model_mnist.py`: Core implementation code files for MNIST Fully Connected Neural Network training loops.
- `model_cifar10_cnn.py`: Convolutional Neural Network (CNN) architecture and training scripts for processing the CIFAR-10 image dataset.
- `model_baf.py`: Tabular fraud classification scripts dealing with class imbalances.
- `dpba_utils.py`: Contains core DPBA sensitivity math, layer importance computations, and MIA pipeline calculations.
- `FINAL_REPORT_DPBA.pdf`: Complete analytical project documentation, mathematical formulations, and data visualization graphs.
