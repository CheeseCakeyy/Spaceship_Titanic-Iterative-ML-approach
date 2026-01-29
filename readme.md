# Spaceship Titanic – Iterative Machine Learning Approach

## Project Overview
This project tackles the **Kaggle Spaceship Titanic classification problem** using a structured, iterative machine learning workflow.  
Each iteration builds on insights from the previous one, focusing on **better preprocessing, feature engineering, model selection, and validation strategy**.

The goal was not just leaderboard performance, but **correct ML methodology, interpretability, and learning progression**.

---

## Iteration 1: Baseline Models & Classical ML

### Approach
- Basic preprocessing (imputation, encoding)
- Train–validation split (no cross-validation)
- Classical models:
  - Naive Bayes
  - Logistic Regression (with regularization)
  - Random Forest

### Key Observations
- Logistic Regression improved after tuning `C`
- Random Forest outperformed linear models
- Performance sensitive to random state
- No meaningful feature engineering

### Results
- **Validation Accuracy:** ~0.77–0.79  
- **Kaggle Score:** 0.798  
- **Leaderboard Rank:** 1522 / 2692

### Takeaway
A solid baseline, but:
- No CV → unstable estimates  
- Feature interactions poorly captured

---

## Iteration 2: Feature Engineering + Cross-Validation + Feature Selection

### Approach
- Proper **ColumnTransformer pipelines**
- Feature groups:
  - Numerical
  - Boolean
  - Categorical
  - Engineered / count-based
- New engineered features:
  - `Total_spend`
  - `Has_spent`
  - Spend-based ratios
- Feature selection using:
  - L1 (Logistic Regression)
- **5-Fold Cross-Validation**
- Models evaluated:
  - RF + LR-based feature selection (**RF_lr**)
  - RF + RF-based feature selection
  - LR baselines

### Key Observations
- RF with LR feature selection was the **most stable**
- CV reduced randomness
- Feature selection slightly improved generalization
- Accuracy gains were incremental

### Results
- **Best CV Accuracy (RF_lr):** ~0.800  
- **Kaggle Score:** 0.799  
- **Leaderboard Rank:** 1453 / 2692

### Takeaway
- Feature engineering helps only when it adds **new signal**
- Stability and correctness > raw accuracy
- RF_lr became the **strong classical ML baseline**

---

## Iteration 3: Gradient Boosting with XGBoost

### Approach
- Introduced **XGBoost (XGBClassifier)**
- Same clean preprocessing pipeline
- Cross-validation retained
- Minimal tuning:
  - Controlled tree depth
  - Moderate learning rate
  - No aggressive regularization

### Key Observations
- XGBoost outperformed RF-based pipelines
- Higher mean accuracy with slightly higher variance
- Indicates lower bias but increased sensitivity
- No evidence of data leakage or overfitting

### Results
- **CV Accuracy:** ~0.804–0.805  
- **Kaggle Score:** ~0.805  
- **Leaderboard Rank:** 741 / 2719

### Takeaway
- Boosting extracted additional non-linear structure
- Performance approached the **practical ceiling** for this dataset
- Further gains likely require ensembling or stronger features

---

## Iteration 4: Gradient Boosting with LightGBM

### Approach
- Introduced **LightGBM (LGBMClassifier)**
- Retained the same preprocessing and engineered features
- Stratified 5-Fold Cross-Validation
- Leveraged **leaf-wise tree growth** (vs level-wise in XGBoost)
- Moderate hyperparameters:
  - Controlled max depth
  - Stable learning rate
  - No heavy regularization

### Key Observations
- LightGBM **consistently outperformed XGBoost**
- Higher mean CV accuracy with **lower variance**
- Better bias–variance tradeoff
- Leaf-wise growth enabled faster and more effective loss reduction
- No signs of overfitting or leakage

### Results
- **CV Accuracy:** ~0.807–0.808  
- **CV Std Dev:** ~0.009  
- **Kaggle Score:** ~0.807  
- **Leaderboard Rank:** **293 / 2292**

### Takeaway
- LightGBM extracted additional structured and non-linear signal
- Achieved the **best single-model performance** in this workflow
- Model performance is now very close to the dataset’s empirical ceiling
- Meaningful gains beyond this point require:
  - Ensembling
  - Stacking
  - Domain-heavy feature engineering

---

## Final Comparison Summary

| Iteration | Model | CV / Val Accuracy | Kaggle Score | Rank |
|--------|------|------------------|-------------|------|
| Iter 1 | RF / LR | ~0.77–0.79 | 0.798 | 1522/2692 |
| Iter 2 | RF + LR FS | ~0.80 | 0.799 | 1453/2692 |
| Iter 3 | XGBoost | ~0.805 | 0.805 | 741/2719 |
| Iter 4 | **LightGBM** | **~0.808** | **0.807** | **293/2292** |
| Iter 5 | Catboost | ~0.816 | 0.8024 | No improvements |

---

## Key Learnings
- Cross-validation is **mandatory** for reliable evaluation
- Feature engineering only helps when it adds **new information**
- Tree-based models outperform linear methods on this dataset
- Boosting reduces bias but must be variance-controlled
- Small leaderboard gains can represent **significant methodological progress**

---

## Conclusion
This project demonstrates a **disciplined and correct ML workflow**, progressing from baseline models to advanced gradient boosting while maintaining reproducibility and interpretability.

> Achieving ~80.7% accuracy with clean pipelines and CV reflects strong modeling rather than leaderboard tricks.

---

## Future Work
- Model ensembling (RF + XGB + LGBM)
- Feature interaction discovery
- Explore CatBoost for native categorical handling (Explored)
- Automating the feature creation by wrapping it into a function

