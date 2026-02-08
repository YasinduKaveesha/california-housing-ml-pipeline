# California Housing Regression ML

A comprehensive machine learning project for predicting median house values using the **California Housing dataset** through systematic model evaluation, feature engineering, and hyperparameter optimization.

## Project Overview

This project implements a complete ML pipeline that evaluates multiple regression approaches—from simple linear models to advanced ensemble methods—to predict housing prices in California. The workflow is structured to provide insights at each modeling stage and demonstrates best practices in model selection, interpretability, and production deployment.

**Key Achievement:** Selected **LightGBM** as the final production model after comprehensive benchmarking, feature engineering, and SHAP-based interpretability analysis.

---

## Final Results

After evaluating 7+ models including linear, regularized, tree-based, and boosting approaches, **advanced boosting methods achieved the strongest performance**:

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **LightGBM (FINAL)** | **0.8597** | **0.4287** | **0.2799** |
| XGBoost | 0.8533 | 0.4385 | 0.2859 |
| Random Forest (Tuned + FE) | 0.8263 | 0.4772 | 0.3133 |
| Gradient Boosting | 0.84 | 0.45 | 0.30 |
| Decision Tree | 0.70 | 0.58 | 0.41 |
| Ridge Regression (α=100) | 0.61 | 0.70 | 0.54 |
| Linear Regression | 0.57 | 0.74 | 0.53 |

**Why LightGBM?** Housing prices involve complex non-linear relationships and feature interactions (e.g., income × location effects). LightGBM captures these automatically through leaf-wise tree growth and optimal regularization, achieving 86% variance explained.

---

## Project Structure

```
california-housing-regression-ml/
├── notebooks/
│   ├── data_exploration.ipynb              # Phase 1: EDA & Data Analysis
│   ├── preprocessing.ipynb                 # Phase 2: Data Cleaning & Scaling
│   ├── baseline_models.ipynb               # Phase 3: Linear & Regularized Models
│   ├── advanced_models.ipynb               # Phase 4: Tree-Based & Boosting Models
│   └── model_optimization_and_interpretation.ipynb  # Phase 5: FE, Tuning, SHAP
├── src/
│   ├── data_loader.py                      # Data loading utilities
│   ├── preprocessing.py                    # Preprocessing functions
│   ├── models.py                           # Model implementations
│   └── evaluation.py                       # Evaluation metrics
├── data/
│   ├── raw/                                # Original dataset (from sklearn)
│   ├── v1_train_test/                      # Train/test split (80/20)
│   │   ├── X_train_scaled.npy
│   │   ├── X_test_scaled.npy
│   │   ├── y_train.npy
│   │   └── y_test.npy
│   └── v2_train_val_test/                  # Train/val/test split for tuning
│       ├── X_train_scaled.npy
│       ├── X_val_scaled.npy
│       ├── X_test_scaled.npy
│       ├── y_train.npy, y_val.npy, y_test.npy
│       ├── feature_names.json
│       └── scaler.joblib
├── models/
│   └── final_lightgbm/
│       ├── lightgbm_model.joblib           # Trained LightGBM model
│       └── kmeans_location_cluster.joblib  # K-Means for LocationCluster
├── results/                                # Output visualizations & metrics
├── README.md                               # This file
├── SUMMARY.txt                             # Detailed execution summary
└── LICENSE                                 # Project license
```

---

## Notebook Workflow (5 Phases)

### **Phase 1: data_exploration.ipynb** — Exploratory Data Analysis
**Objective:** Understand dataset structure and feature relationships.

**Key Analyses:**
- Load California Housing dataset (20,640 samples, 8 features)
- Descriptive statistics and data types
- Target variable (MedHouseVal) distribution — right-skewed pattern
- Univariate relationships: MedInc, Population, Latitude, Longitude vs House Value
- Skewness/kurtosis analysis

**Key Finding:** Strong positive correlation between Median Income and prices; clear geographic pricing patterns.

---

### **Phase 2: preprocessing.ipynb** — Data Preparation & Scaling
**Objective:** Prepare clean, scaled data with proper train-test separation.

**Key Steps:**
1. **Train-Test Split** (80/20, random_state=42)
   - Train: 16,512 samples | Test: 4,128 samples
2. **StandardScaler Normalization**
   - Fit scaler on **training data only** (prevent data leakage)
   - Applied to all features (mean=0, std=1)
3. **Two Dataset Versions Created:**
   - **v1 (train/test):** For final model evaluation
   - **v2 (train/val/test):** For hyperparameter tuning without touching test set

**Scaling Rationale:**
- **Required for linear/regularized models** (Ridge, Lasso): Without scaling, large-range features dominate regularization
- **Not required for tree-based models** (Decision Trees, Random Forest, LightGBM): Scale-invariant
- Applied universally for pipeline consistency

---

### **Phase 3: baseline_models.ipynb** — Linear & Regularized Models
**Objective:** Establish baseline performance with interpretable models.

**Models Tested:**
1. **Linear Regression (OLS)**
   - RMSE: 0.74 | MAE: 0.53 | R²: 0.57
   
2. **Ridge Regression (L2 Regularization)**
   - Best alpha: 100
   - RMSE: 0.70 | MAE: 0.54 | R²: 0.61
   
3. **Lasso Regression (L1 Regularization)**
   - Best alpha: 0.01
   - RMSE: 0.73 | MAE: 0.53 | R²: 0.58
   - Feature selection: 8/8 coefficients retained

**Finding:** Linear models explain only ~57-61% of variance — non-linear patterns exist in the data.

---

### **Phase 4: advanced_models.ipynb** — Tree-Based & Boosting Models
**Objective:** Capture non-linear relationships and feature interactions.

**Models Evaluated:**
- **Decision Tree** (max_depth=5): R² 0.70
- **Random Forest** (300 trees): R² 0.80
- **Gradient Boosting** (500 trees, lr=0.05): R² 0.84
- **XGBoost** (800 trees, lr=0.05, subsample=0.8): R² 0.85
- **LightGBM** (2000 trees, lr=0.03, num_leaves=31): R² 0.86

**Progressive Improvement:** Decision Tree (+13%) → Random Forest (+10%) → Gradient Boosting (+4%) → XGBoost (+1%) → LightGBM (+1%)

---

### **Phase 5: model_optimization_and_interpretation.ipynb** — Feature Engineering & SHAP Analysis
**Objective:** Optimize model through feature engineering and provide explainability.

#### **5.1 Feature Engineering**
**Domain-Informed Features:**
- `Rooms_per_Household` = AveRooms / (Population / AveOccup)
- `Bedrooms_per_Room` = AveBedrms / AveRooms
- `Population_per_Household` = Population / (Population / AveOccup)

**Spatial Clustering:**
- `LocationCluster` — K-Means clustering (k=8) on Latitude/Longitude
  - Fit on training data only (no leakage)
  - Captures geographic pricing regions beyond raw coordinates
  - Treated as **categorical for LightGBM**, one-hot encoded for RF/XGBoost

#### **5.2 Hyperparameter Tuning**
**RandomizedSearchCV Configuration:**
- Base model: Random Forest
- Search space: 25 random samples from 4,000+ combinations
- CV: 5-fold cross-validation
- Metric: Negative RMSE
- Hyperparameters tuned: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

**Tuned Random Forest Performance (with FE):**
- CV RMSE: ~0.45 | MAE: 0.30 | R²: 0.85

#### **5.3 Advanced Boosting Models with Feature Engineering**
- **XGBoost (optimized):** R² 0.8533
- **LightGBM (final):** R² 0.8597 (marginal improvement from FE, but robustness increased)

#### **5.4 Interpretability with SHAP Analysis**
**Global Feature Importance:**
- Summary plots show feature impact distributions across all test samples
- **Top predictors:** MedInc (strongest), Latitude, Longitude, AveOccup, Rooms_per_Household

**Local Explanations:**
- **Low-Error Case:** Features align coherently → accurate standard predictions
- **High-Error Case:** Conflicting signals on rare feature combinations → model underestimates
- Waterfall plots show baseline + individual feature contributions per prediction

**Key Insight:** SHAP analysis revealed model underperforms on rare combinations not well-represented in training data.

---

## Requirements & Setup

### Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
pip install xgboost lightgbm shap joblib
```

### Python Version
- Python 3.8+

---

## Quick Start

### Running the Complete Pipeline

```bash
# Phase 1: Exploratory Data Analysis
jupyter notebook notebooks/data_exploration.ipynb

# Phase 2: Data Preprocessing
jupyter notebook notebooks/preprocessing.ipynb

# Phase 3: Baseline Models
jupyter notebook notebooks/baseline_models.ipynb

# Phase 4: Advanced Models
jupyter notebook notebooks/advanced_models.ipynb

# Phase 5: Optimization & Interpretability
jupyter notebook notebooks/model_optimization_and_interpretation.ipynb
```

### Outputs After Running

- **Preprocessed datasets:** `data/v1_train_test/` and `data/v2_train_val_test/` (numpy arrays)
- **Trained models:** Available in notebook cells
- **Performance metrics:** Printed outputs (RMSE, MAE, R²)
- **Visualizations:** EDA plots, feature relationships, model comparisons
- **SHAP explanations:** Global summary plots + local waterfall plots

---

## Saved Artifacts (Production)

### Pre-Trained Models
Location: `models/final_lightgbm/`

| File | Purpose | Notes |
|------|---------|-------|
| `lightgbm_model.joblib` | Trained LightGBM regressor | Direct predictions after preprocessing |
| `kmeans_location_cluster.joblib` | K-Means clustering model | Generates LocationCluster feature |

### Inference Pipeline

To use the saved model for predictions:

1. **Load preprocessing components:**
   ```python
   import joblib
   kmeans = joblib.load('models/final_lightgbm/kmeans_location_cluster.joblib')
   model = joblib.load('models/final_lightgbm/lightgbm_model.joblib')
   ```

2. **Feature engineering (match training):**
   ```python
   # Add derived features
   X['Rooms_per_Household'] = X['AveRooms'] / (X['Population'] / X['AveOccup'])
   X['Bedrooms_per_Room'] = X['AveBedrms'] / X['AveRooms']
   X['Population_per_Household'] = X['Population'] / (X['Population'] / X['AveOccup'])
   
   # Add spatial cluster
   coords = X[['Latitude', 'Longitude']].values
   X['LocationCluster'] = kmeans.predict(coords)
   ```

3. **Predict:**
   ```python
   predictions = model.predict(X[feature_columns])  # Returns house values in $100,000s
   ```

**Note:** LightGBM handles categorical features natively. No additional encoding needed.

---

## Dataset Information

**Source:** California Housing dataset (1990 U.S. Census, via sklearn)

**Features (8 numeric continuous):**
- `MedInc` — Median household income ($10,000s)
- `HouseAge` — Median house age (years, capped at 52)
- `AveRooms` — Average rooms per household
- `AveBedrms` — Average bedrooms per household
- `Population` — Total block group population
- `AveOccup` — Average household occupancy
- `Latitude`, `Longitude` — Geographic coordinates

**Target (numeric continuous):**
- `MedHouseVal` — Median house value ($100,000s, capped at 5.0)

**Dataset Size:**
- 20,640 samples (block groups) | 8 features | No missing values

---

## Key Learning Outcomes

This project demonstrates:
- ✅ End-to-end ML pipeline from EDA to production deployment
- ✅ Systematic model evaluation and comparison (7+ models)
- ✅ Feature scaling, preprocessing, and data leakage prevention
- ✅ Linear vs non-linear model trade-offs
- ✅ Ensemble methods and gradient boosting optimization
- ✅ Hyperparameter tuning with cross-validation
- ✅ Feature engineering and domain-informed improvements
- ✅ Model interpretability through SHAP analysis
- ✅ Local explanations for individual predictions
- ✅ Production artifact management and inference pipeline

---

## Model Selection Rationale

### Why LightGBM?

1. **Superior Performance**
   - Highest R² (0.8597) → Explains 86% of price variance
   - Lowest RMSE (0.4287) and MAE (0.2799)

2. **Algorithmic Advantages**
   - Leaf-wise tree growth → Optimal split selection
   - Lower learning rate (0.03) with more trees (2000) → Stable convergence
   - Better gradient handling vs sklearn's GradientBoosting

3. **Production Features**
   - Fast inference (parallel processing)
   - Native categorical feature support
   - Built-in regularization (num_leaves=31)
   - Industry-standard (proven in large-scale systems)

4. **Robustness**
   - Handles feature interactions automatically
   - Subsample (0.8) + colsample_bytree (0.8) prevent overfitting
   - Minimal train-test gap indicates good generalization

---

## Notes

- **Reproducibility:** Random seeds fixed (random_state=42) across all models
- **Fair Comparison:** All models evaluated on identical 80/20 train-test split using same metrics
- **Scaling:** StandardScaler applied to all datasets for pipeline consistency (required for linear models)
- **No Data Leakage:** Scaler, k-means, and all preprocessing fit only on training data
- **Feature Engineering:** Applied consistently in Phase 5 for all tuned models

---

## License

See [LICENSE](LICENSE) file for details.

---

## Author

Yasindu Kaveesha — California Housing ML Regression Project

**Last Updated:** February 8, 2026
