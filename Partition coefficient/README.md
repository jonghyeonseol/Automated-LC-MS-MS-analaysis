# XLogP3_prediction_PROTAC_Linker

## Overview
This project implements machine learning models to predict XLogP3 values for molecular compounds based on their physicochemical properties. The project includes Explarotary data analysis, feature selection, data preprocessing, and visualization components. This project used the linker dataset from PROTAC-DB

## Features
- Feature selection using SelectKBest with F-regression
- Data preprocessing with StandardScaler
- Visualization of:
  - Feature selection scores
  - Feature correlations
  - Feature distributions
  - Feature vs target relationships
  - Feature importance scores

## Dataset
The project uses a dataset containing molecular properties including:
- Molecular Weight
- Exact Mass
- Heavy Atom Count
- Ring Count
- Hydrogen Bond Acceptor Count
- Hydrogen Bond Donor Count
- Rotatable Bond Count
- Topological Polar Surface Area
- XLogP3 (target variable)

## Requirements
```python
pandas>=1.0.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
xgboost>=1.3.0
