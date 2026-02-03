# California Housing Regression ML

A professional machine learning project implementing a complete regression workflow on the California Housing dataset.

## Project Description
End-to-end regression modeling using baseline linear regression, regularization (Ridge/Lasso), ensemble models, and cross-validation with proper evaluation and interpretation.

## Dataset Source
- **Source**: California Housing dataset from scikit-learn
- **Task**: Regression—predict median house values
- **Features**: 8 socioeconomic and geographic attributes

## Evaluation Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

## Workflow Outline
1. **Exploratory Data Analysis** - Data inspection and visualization
2. **Data Preprocessing** - Scaling, missing value handling, train/test split
3. **Baseline Models** - Linear regression as performance baseline
4. **Regularized Models** - Ridge and Lasso regression
5. **Ensemble Models** - Advanced techniques (Random Forest, Gradient Boosting)
6. **Evaluation** - Cross-validation and performance comparison

## Project Structure
```
california-housing-regression-ml/
├── LICENSE
├── README.md
├── SUMMARY.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_advanced_models.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
└── results/
```

## Status
Project in development. Data exploration phase in progress.
