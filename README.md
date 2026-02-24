# Runtime and Duration Class Prediction in HPC Systems

This repository contains the code and data used for a series of experiments on **runtime and duration class prediction** in High-Performance Computing (HPC) systems.  
The goal is to replace inaccurate user-provided runtime estimates with machine learning and deep learning predictions, improving job scheduling efficiency.

---

## Overview

This project represents the first phase of a larger effort to integrate ML techniques into HPC workload dispatching.  
It focuses on predicting:

- **Runtime** (regression task)  
- **Duration class** (classification task)

for jobs submitted on the **Marconi100 supercomputer** (CINECA).

Machine learning methods are compared against traditional baselines:

- **User-provided estimates**  
- **Simple historical heuristic**  

and include:

- Online Decision Tree Regression  
- Online Ridge Normalized Polynomial Regression (custom Ridge-based model)  
- Online k-Nearest Neighbors (k-NN) Regression  
- Online Retrieval-Augmented Language Model
- Online k-NN Classification with 4 and 7 duration classes  

Each model is tested with **two feature sets**:

- **SET 1**: Minimal, directly available at job submission  
- **SET 2**: Extends SET 1 with historical runtimes and user runtime averages  

---

## Key Components

### DataLoader (`modules/data_loader.py`)
The `DataLoader` class:

- Loads the original **PM100 dataset**  
- Enriches it with additional historical features  
- Defines **duration classes**:
  - **4-class**: _Very-Short, Short, Medium, Long_  
  - **7-class**: _Very-Short, Short, Medium-Short, Medium, Medium-Long, Long, Very-Long_  
- Splits data chronologically (70% train / 30% test)  
- Saves resulting **parquet files** for downstream experiments  

### Prediction Models (`modules/prediction_models.py`)
Implements **online incremental models** for runtime prediction:

1. **OnlineRidgePolynomialRegressor (RNP)**  
   - Ridge-regularized polynomial regression with incremental training  
   - Configurable polynomial degree, batch size, and history window  

2. **OnlineDecisionTreeRegressor (DT)**  
   - Decision tree regression trained incrementally  
   - Supports limiting historical samples  

3. **OnlineKNNRegressor (KNN)**  
   - k-Nearest Neighbors regression with distance weighting  
   - Maintains a buffer of past samples  

4. **OnlineKNNClassifier (KNN_C4 / KNN_C7)**  
   - k-Nearest Neighbors classification for 4 or 7 duration classes  
   - Returns class predictions and probabilities  

> Note: The **Online Retrieval-Augmented Language Model (LLM)** is defined directly in the testing notebook (`5 - LLM.ipynb`).

### Utilities (`modules/utils.py`)
Provides helper functions for:

- Computing evaluation metrics  
- Plotting histograms and scatter plots to compare predicted vs. actual runtimes  

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `0 - extend_dataset.ipynb` | Use `DataLoader` to prepare training and testing datasets |
| `1 - heuristic.ipynb` | Compute baselines (`user` and `heuristic`) and save predictions |
| `2 - DT.ipynb` | Test Online Decision Tree Regressor and save predictions |
| `3 - RNP.ipynb` | Test Online Ridge Normalized Polynomial Regressor and save predictions |
| `4 - KNN.ipynb` | Test Online KNN Regressor and save predictions |
| `5 - LLM.ipynb` | Define and test Online Retrieval-Augmented Language Model and save predictions |
| `6_1 - KNN_C4.ipynb` | Test Online KNN Classifier (4 classes) and save predictions |
| `6_2 - KNN_C7.ipynb` | Test Online KNN Classifier (7 classes) and save predictions |
| `7 - fix_predictions.ipynb` | Fix regressors’ prediction errors (convert to integers and cap at user estimate) |
| `8 - compute_metrics.ipynb` | Compute metrics (MAE, MAPE, EA) for the whole test dataset and individual workloads |

---

## Directory Structure
```
├── datasets/    # Original PM100 dataset + train/test parquet files
├── metrics/     # CSV files with computed metrics
├── modules/     # Core Python modules (DataLoader, Prediction Models, Utils)
├── predictions/ # CSV files with predictions from each model
├── workloads/   # Subsets of the test dataset for scheduling simulations
├── *.ipynb      # Notebooks for data preparation, model testing, and metric computation
└── README.md    # This file
```

---

## Evaluation Metrics

The models are evaluated using:

- **MAE** – Mean Absolute Error  
- **MAPE** – Mean Absolute Percentage Error  
- **EA** – Estimation Accuracy  

Metrics are computed both globally (entire test set) and per individual workload to support downstream scheduling simulations.

---
