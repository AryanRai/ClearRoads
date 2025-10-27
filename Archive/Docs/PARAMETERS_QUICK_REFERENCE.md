# Model Parameters - Quick Reference

## All 5 Models at a Glance

```python
# 1. k-Nearest Neighbors
KNeighborsClassifier(
    n_neighbors=5          # Look at 5 closest neighbors
)

# 2. Decision Tree
DecisionTreeClassifier(
    max_depth=10,          # Maximum 10 levels deep
    random_state=42        # Reproducibility seed
)

# 3. Random Forest (BEST: 96.35%)
RandomForestClassifier(
    n_estimators=100,      # Build 100 trees
    max_depth=15,          # Each tree up to 15 levels
    random_state=42        # Reproducibility seed
)

# 4. Neural Network
MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 2 layers: 100 → 50 neurons
    max_iter=500,                  # Train for 500 iterations
    random_state=42                # Reproducibility seed
)

# 5. XGBoost
XGBClassifier(
    n_estimators=100,      # Build 100 boosted trees
    max_depth=6,           # Each tree up to 6 levels
    learning_rate=0.1,     # 10% contribution per tree
    random_state=42,       # Reproducibility seed
    eval_metric='mlogloss' # Multi-class log loss
)
```

---

## Parameter Comparison

| Parameter | kNN | Decision Tree | Random Forest | Neural Net | XGBoost |
|-----------|-----|---------------|---------------|------------|---------|
| **Number of models** | 1 | 1 | 100 trees | 1 network | 100 trees |
| **Depth/Complexity** | k=5 | depth=10 | depth=15 | (100,50) | depth=6 |
| **Learning approach** | Instance | Greedy split | Ensemble | Gradient descent | Boosting |
| **Random seed** | - | 42 | 42 | 42 | 42 |
| **Special params** | - | - | - | max_iter=500 | lr=0.1 |

---

## Why These Values?

### kNN: n_neighbors=5
✅ Odd number (no ties)  
✅ Not too small (k=1 is noisy)  
✅ Not too large (k=100 is too smooth)

### Decision Tree: max_depth=10
✅ Prevents overfitting  
✅ Moderate complexity  
✅ Captures patterns without memorizing

### Random Forest: n_estimators=100, max_depth=15
✅ 100 trees = stable predictions  
✅ Deeper trees OK (ensemble reduces overfitting)  
✅ Best performer (96.35%)

### Neural Network: (100, 50), max_iter=500
✅ Funnel architecture (wide → narrow)  
✅ 100 neurons capture complexity  
✅ 500 iterations ensure convergence

### XGBoost: n_estimators=100, max_depth=6, lr=0.1
✅ 100 trees = good accuracy  
✅ Shallow trees (boosting needs less depth)  
✅ 0.1 learning rate = balanced speed/accuracy

---

## Pipeline (Applied to All Models)

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing with median
    ('scaler', StandardScaler()),                   # Normalize to mean=0, std=1
    ('classifier', model)                           # The actual model
])
```

**Why this pipeline?**
- **Imputer**: Handles any remaining missing values robustly
- **Scaler**: Puts all features on same scale (critical for kNN, NN)
- **Order matters**: Impute → Scale → Classify

---

## Cross-Validation Setup

```python
StratifiedKFold(
    n_splits=5,        # 5-fold CV
    shuffle=True,      # Randomize before splitting
    random_state=42    # Reproducible folds
)
```

**Result**: 80% train, 20% validate per fold, averaged across 5 folds

---

## Train-Test Split

```python
train_test_split(
    test_size=0.2,     # 20% test, 80% train
    random_state=42,   # Reproducible split
    stratify=y         # Maintain class balance
)
```

**Result**: 55,693 train, 13,924 test samples

---

## Performance Summary

| Model | Test Accuracy | Key Strength |
|-------|---------------|--------------|
| Random Forest | **96.35%** 🥇 | Ensemble stability |
| XGBoost | 96.29% 🥈 | Boosting power |
| Neural Network | 95.48% 🥉 | Non-linear patterns |
| Decision Tree | 95.03% | Interpretability |
| kNN | 79.09% | Simplicity |

---

## Quick Tuning Guide

### To improve Random Forest:
```python
n_estimators=200      # More trees (slower but more stable)
max_depth=20          # Deeper trees (more complex patterns)
min_samples_split=5   # Require more samples to split (regularization)
```

### To improve XGBoost:
```python
learning_rate=0.05    # Lower rate (needs more trees)
n_estimators=200      # More trees (with lower learning rate)
max_depth=8           # Deeper trees (more complexity)
subsample=0.8         # Use 80% of data per tree (regularization)
```

### To improve Neural Network:
```python
hidden_layer_sizes=(200, 100, 50)  # Deeper network
max_iter=1000                      # More training iterations
alpha=0.001                        # Stronger regularization
```

---

## Common Questions

**Q: Why random_state=42 everywhere?**  
A: Ensures reproducible results. 42 is a convention (any number works).

**Q: Why is Random Forest max_depth=15 but XGBoost max_depth=6?**  
A: Random Forest uses parallel trees (can be deeper), XGBoost uses sequential boosting (needs shallower trees to prevent overfitting).

**Q: Why not use more trees (e.g., 1000)?**  
A: Diminishing returns after ~100-200 trees. More trees = longer training with minimal accuracy gain.

**Q: Can I change these parameters?**  
A: Yes! These are starting points. Use GridSearchCV or RandomizedSearchCV to find optimal values for your specific data.

**Q: Why does kNN perform worst?**  
A: Struggles with high-dimensional data (19 features) and large datasets (69,617 samples). Distance-based methods don't scale well.

---

## Parameter Glossary

| Term | Meaning |
|------|---------|
| **n_neighbors** | Number of nearest neighbors to consider (kNN) |
| **max_depth** | Maximum tree depth (prevents overfitting) |
| **n_estimators** | Number of trees in ensemble |
| **hidden_layer_sizes** | Number of neurons per layer (Neural Net) |
| **max_iter** | Maximum training iterations |
| **learning_rate** | Step size for weight updates |
| **random_state** | Seed for reproducibility |
| **eval_metric** | Performance metric to optimize |
| **strategy** | Method for imputation (median/mean/most_frequent) |
| **test_size** | Proportion of data for testing |
| **stratify** | Maintain class distribution in splits |
| **n_splits** | Number of cross-validation folds |

---

**For detailed explanations, see MODEL_PARAMETERS_EXPLAINED.md**
