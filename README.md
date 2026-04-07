# SafeSpan: Predictive Bridge Analytics

SafeSpan is a machine learning project designed to predict the structural condition of U.S. bridges using historical inspection data. By leveraging the National Bridge Inventory (NBI) dataset, we aim to move from reactive maintenance to a proactive, data-driven approach for infrastructure safety.

## 🚀 Project Overview

The U.S. bridge infrastructure is aging, with approximately 600,000 public road bridges requiring regular inspection. This project utilizes a massive dataset of over **5 million records** (2014–2025) to classify bridge conditions into four priority categories: **Critical**, **Poor**, **Fair**, and **Good**.

### Key Objectives
- **Data Aggregation**: Automated collection and cleaning of the FHWA National Bridge Inventory.
- **Predictive Modeling**: Implementation of Classical Baselines, Tree-Based Systems (XGBoost/Random Forest), and Margin/Kernel Methods (SVM).
- **Interpretability**: Using SHAP and LIME to provide transparent, safety-critical failure analysis.
- **Class Imbalance Handling**: Addressing rare "Critical" events using SMOTETomek and balanced loss functions.

## 📁 Repository Structure

- `data_collection.py`: Script to download, unzip, and aggregate NBI ASCII files from FHWA.
- `EDA_GUIDE.md`: Comprehensive guide for data cleaning and initial exploratory data analysis.
- `presentation_outline.md`: Outline and speaker notes for the April 7 Progress Presentation.
- `DATA245_Machine_Learning_Technologies_Project_Guidelines (1).pdf`: Official project requirements.
- `ML Project Proposal.pdf`: Initial project scoping and technical proposal.

## 📊 Dataset

The core dataset is compiled from the [Federal Highway Administration (FHWA)](https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm). Due to its size (~2.1GB), the raw CSV is not included in this repository. Use the `data_collection.py` script to generate it locally.

## 🛠️ Next Steps

1. **Initial EDA**: Execute the cleaning steps outlined in `EDA_GUIDE.md`.
2. **Feature Engineering**: Derive structural and traffic indicators (Age, Traffic Density, etc.).
3. **Model Benchmarking**: Implement baseline classifiers for initial performance metrics.

---
**Team Members**: Akashkumar Senthilkumar, Pramod Satya Dindukurthi, Shriram Dundigalla, Shruthi Thirukumaran
**Course**: DATA 245 – Machine Learning Technologies (Spring 2026)
