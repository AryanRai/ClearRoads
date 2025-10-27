# Model Parameters Explained

## Overview of All 5 Models

```python
models = {
    'kNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                             random_state=42, eval_metric='mlogloss')
}
```

---

## 1. k-Nearest Neighbors (kNN)

### Code:
```python
KNeighborsClassifier(n_neighbors=5)
```

### Parameters Explained:

#### `n_neighbors=5`
- **What it means**: Look at the 5 closest training examples to make a prediction
- **How it works**: 
  1. Calculate distance from new observation to all training observations
  2. Find the 5 nearest neighbors
  3. Take majority vote of their congestion classes
  4. Predict the most common class among the 5 neighbors

**Example:**
```
New observation: PM10=15, PM2.5=6, morning_rush=1200...

Find 5 nearest neighbors:
  Neighbor 1: "High" (distance: 0.12)
  Neighbor 2: "High" (distance: 0.15)
  Neighbor 3: "Very High" (distance: 0.18)
  Neighbor 4: "High" (distance: 0.21)
  Neighbor 5: "Low" (distance: 0.23)

Vote: High=3, Very High=1, Low=1
Prediction: "High" (majority wins)
```

**Why k=5?**
- Common default value
- Odd number prevents ties
- Not too small (k=1 would be noisy)
- Not too large (k=100 would be too smooth)

**Other parameters (using defaults):**
- `weights='uniform'`: All neighbors have equal vote
- `metric='minkowski'`: Euclidean distance
- `p=2`: Standard Euclidean distance formula

### Performance:
- **CV Accuracy**: 76.46% ± 0.31%
- **Test Accuracy**: 79.09%
- **Rank**: 5th (worst performer)

**Why it underperforms:**
- Struggles with high-dimensional data (19 features)
- Sensitive to feature scaling (we use StandardScaler to help)
- Computationally expensive for large datasets
- Doesn't capture complex patterns well

---

## 2. Decision Tree

### Code:
```python
DecisionTreeClassifier(max_depth=10, random_state=42)
```

### Parameters Explained:

#### `max_depth=10`
- **What it means**: Tree can have maximum 10 levels of decisions
- **How it works**: Limits how deep the tree can grow
- **Why it matters**: Prevents overfitting

**Visual Example:**
```
Level 1: Is peak_hour_traffic > 300?
         ├─ Yes → Level 2: Is evening_rush > 1000?
         │        ├─ Yes → Level 3: Is PM2.5 > 8?
         │        │        ├─ Yes → ... (up to level 10)
         │        │        └─ No → ...
         │        └─ No → ...
         └─ No → Level 2: Is morning_rush > 500?
                  └─ ...
```

**Why max_depth=10?**
- Balances complexity and generalization
- Too shallow (e.g., 3): Underfits, misses patterns
- Too deep (e.g., 30): Overfits, memorizes training data
- 10 is a moderate depth for our dataset size

#### `random_state=42`
- **What it means**: Seed for random number generator
- **Why it matters**: Ensures reproducible results
- **How it works**: When features have equal importance, tree randomly chooses which to split on
- **Why 42?**: Convention (from "Hitchhiker's Guide to the Galaxy"), any number works

**Other parameters (using defaults):**
- `criterion='gini'`: Uses Gini impurity to measure split quality
- `min_samples_split=2`: Minimum samples required to split a node
- `min_samples_leaf=1`: Minimum samples required in a leaf node

### Performance:
- **CV Accuracy**: 94.99% ± 0.24%
- **Test Accuracy**: 95.03%
- **Rank**: 4th

**Why it performs well:**
- Captures non-linear relationships
- Handles feature interactions naturally
- Interpretable decision rules
- max_depth=10 prevents overfitting

---

## 3. Random Forest (BEST MODEL)

### Code:
```python
RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
```

### Parameters Explained:

#### `n_estimators=100`
- **What it means**: Build 100 different decision trees
- **How it works**: 
  1. Create 100 trees, each trained on a random subset of data
  2. Each tree makes a prediction
  3. Take majority vote across all 100 trees

**Example:**
```
Tree 1 predicts: "High"
Tree 2 predicts: "High"
Tree 3 predicts: "Very High"
Tree 4 predicts: "High"
...
Tree 100 predicts: "High"

Vote count: High=65, Very High=25, Low=8, Very Low=2
Final prediction: "High" (majority)
```

**Why n_estimators=100?**
- More trees = more stable predictions
- 100 is a good balance between accuracy and speed
- Diminishing returns after ~100-200 trees
- More trees = longer training time

#### `max_depth=15`
- **What it means**: Each tree can be up to 15 levels deep
- **Why deeper than Decision Tree?**: Random Forest is less prone to overfitting
  - Each tree sees different data (bootstrap sampling)
  - Averaging across trees reduces overfitting
  - Can afford deeper trees without memorizing

**Why max_depth=15?**
- Allows trees to capture complex patterns
- Still prevents individual trees from overfitting
- Ensemble averaging provides additional regularization

#### `random_state=42`
- **What it means**: Seed for reproducibility
- **Controls**: 
  - Which samples each tree sees (bootstrap sampling)
  - Which features are considered at each split
  - Random tie-breaking

**Other parameters (using defaults):**
- `max_features='sqrt'`: Each split considers √19 ≈ 4 features randomly
- `bootstrap=True`: Each tree trained on random sample with replacement
- `min_samples_split=2`: Minimum samples to split a node
- `min_samples_leaf=1`: Minimum samples in a leaf

### Performance:
- **CV Accuracy**: 96.33% ± 0.10%
- **Test Accuracy**: 96.35%
- **Rank**: 🥇 1st (BEST)

**Why it's the best:**
- Ensemble of 100 trees reduces variance
- Handles non-linear relationships
- Robust to outliers and noise
- Provides feature importance
- Very stable (low CV variance)

---

## 4. Neural Network (MLP)

### Code:
```python
MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
```

### Parameters Explained:

#### `hidden_layer_sizes=(100, 50)`
- **What it means**: 2 hidden layers with 100 and 50 neurons respectively
- **Architecture**:

```
Input Layer (19 features)
    ↓
Hidden Layer 1 (100 neurons)
    ↓
Hidden Layer 2 (50 neurons)
    ↓
Output Layer (4 classes)
```

**Detailed view:**
```
Input: [PM10, PM2.5, NO2, ..., season_Winter]  (19 features)
         ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
Layer 1: [●●●●●●●●●●...●●●●●●●●●●]  (100 neurons)
         Each neuron: activation(w₁×PM10 + w₂×PM2.5 + ... + bias)
         ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
Layer 2: [●●●●●●●●●●...●●●●●●●●●●]  (50 neurons)
         Each neuron: activation(w₁×h₁ + w₂×h₂ + ... + bias)
         ↓ ↓ ↓ ↓
Output:  [Very Low, Low, High, Very High]  (4 probabilities)
```

**Why (100, 50)?**
- **First layer (100)**: Larger to capture complex patterns from 19 inputs
- **Second layer (50)**: Smaller to compress information
- **Funnel architecture**: Wide → Narrow → Output
- Common pattern for classification tasks

**Alternative architectures:**
- `(50,)`: Single layer with 50 neurons (simpler)
- `(100, 100)`: Two equal layers (more capacity)
- `(200, 100, 50)`: Three layers (deeper, more complex)

#### `max_iter=500`
- **What it means**: Train for maximum 500 iterations (epochs)
- **How it works**: 
  1. Pass all training data through network (1 epoch)
  2. Calculate error
  3. Update weights using backpropagation
  4. Repeat up to 500 times or until convergence

**Why max_iter=500?**
- Default is 200, often too few
- 500 gives enough time to converge
- Early stopping prevents overfitting if it converges sooner
- More iterations = longer training time

#### `random_state=42`
- **What it means**: Seed for weight initialization
- **Why it matters**: Neural networks start with random weights
- **Controls**: Initial weight values before training

**Other parameters (using defaults):**
- `activation='relu'`: ReLU activation function (max(0, x))
- `solver='adam'`: Adam optimizer for weight updates
- `alpha=0.0001`: L2 regularization strength
- `learning_rate='constant'`: Fixed learning rate
- `learning_rate_init=0.001`: Initial learning rate value

### Performance:
- **CV Accuracy**: 95.20% ± 0.23%
- **Test Accuracy**: 95.48%
- **Rank**: 🥉 3rd

**Why it performs well:**
- Captures complex non-linear patterns
- Two layers allow hierarchical feature learning
- ReLU activation prevents vanishing gradients
- Adam optimizer adapts learning rate

**Why not the best:**
- More variance than Random Forest (±0.23% vs ±0.10%)
- Black box (hard to interpret)
- Requires more tuning
- Can overfit without proper regularization

---

## 5. XGBoost

### Code:
```python
XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
              random_state=42, eval_metric='mlogloss')
```

### Parameters Explained:

#### `n_estimators=100`
- **What it means**: Build 100 boosted trees sequentially
- **How it works** (different from Random Forest):
  1. Train tree 1 on original data
  2. Find mistakes tree 1 made
  3. Train tree 2 to correct tree 1's mistakes
  4. Train tree 3 to correct trees 1+2's mistakes
  5. Repeat 100 times
  6. Final prediction = weighted sum of all trees

**Example:**
```
Tree 1: Predicts "High", actual "Very High" → Error = +1 class
Tree 2: Focuses on correcting this error → Predicts "Very High"
Tree 3: Refines further → Predicts "Very High"
...
Tree 100: Final refinement

Final prediction: Weighted combination of all trees
```

**Why n_estimators=100?**
- More trees = better accuracy (up to a point)
- 100 is standard for moderate-sized datasets
- Diminishing returns after 100-200
- Too many trees = overfitting risk

#### `max_depth=6`
- **What it means**: Each tree can be up to 6 levels deep
- **Why shallower than Random Forest (15)?**: 
  - XGBoost uses boosting (sequential correction)
  - Shallow trees prevent overfitting in boosting
  - Each tree focuses on specific errors
  - Depth 6 is XGBoost's default and works well

**Why max_depth=6?**
- Balances complexity and generalization
- Deeper trees in boosting → overfitting
- 6 is empirically proven to work well
- Shallower than Random Forest by design

#### `learning_rate=0.1`
- **What it means**: How much each tree contributes to final prediction
- **How it works**: 
  ```
  Prediction = Tree₁×0.1 + Tree₂×0.1 + Tree₃×0.1 + ... + Tree₁₀₀×0.1
  ```
- **Effect**: 
  - High (0.5): Fast learning, risk of overfitting
  - Low (0.01): Slow learning, needs more trees
  - 0.1: Good balance

**Why learning_rate=0.1?**
- Standard default value
- Balances speed and accuracy
- Works well with n_estimators=100
- Lower rate would need more trees

**Trade-off:**
```
learning_rate=0.3, n_estimators=50  → Fast but less accurate
learning_rate=0.1, n_estimators=100 → Balanced (our choice)
learning_rate=0.01, n_estimators=1000 → Slow but potentially more accurate
```

#### `random_state=42`
- **What it means**: Seed for reproducibility
- **Controls**: Random sampling and tie-breaking

#### `eval_metric='mlogloss'`
- **What it means**: Use multi-class log loss to evaluate performance
- **Why it matters**: 
  - Measures prediction confidence, not just correctness
  - Penalizes confident wrong predictions more
  - Standard for multi-class classification

**How mlogloss works:**
```
Prediction: [Very Low: 0.05, Low: 0.10, High: 0.80, Very High: 0.05]
Actual: "High"

Log loss = -log(0.80) = 0.22  (low is good)

If prediction was:
[Very Low: 0.05, Low: 0.80, High: 0.10, Very High: 0.05]
Actual: "High"

Log loss = -log(0.10) = 2.30  (high is bad - confident but wrong)
```

**Other parameters (using defaults):**
- `objective='multi:softprob'`: Multi-class classification with probabilities
- `booster='gbtree'`: Use tree-based boosting
- `subsample=1.0`: Use all samples for each tree
- `colsample_bytree=1.0`: Use all features for each tree

### Performance:
- **CV Accuracy**: 96.16% ± 0.12%
- **Test Accuracy**: 96.29%
- **Rank**: 🥈 2nd

**Why it performs well:**
- Boosting corrects mistakes iteratively
- Handles complex patterns
- Built-in regularization
- Very stable (low CV variance)

**Why not the best:**
- Slightly lower than Random Forest (96.29% vs 96.35%)
- More sensitive to hyperparameters
- Requires careful tuning
- Can overfit if not properly configured

---

## Pipeline Parameters (Applied to All Models)

### Code:
```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', model)
])
```

### 1. SimpleImputer

#### `strategy='median'`
- **What it means**: Fill any remaining missing values with median
- **Why median?**: 
  - Robust to outliers (unlike mean)
  - Works well for skewed distributions
  - Our air quality data has some outliers

**Example:**
```
Feature values: [10, 12, NaN, 15, 100]
Mean = 34.25 (affected by outlier 100)
Median = 12.5 (robust to outlier)

Fill NaN with 12.5
```

### 2. StandardScaler

**What it does**: Standardizes features to have mean=0 and std=1

**Formula:**
```
scaled_value = (value - mean) / std
```

**Example:**
```
PM10 values: [10, 15, 20, 25, 30]
Mean = 20, Std = 7.07

Scaled values:
10 → (10-20)/7.07 = -1.41
15 → (15-20)/7.07 = -0.71
20 → (20-20)/7.07 = 0.00
25 → (25-20)/7.07 = 0.71
30 → (30-20)/7.07 = 1.41
```

**Why it matters:**
- **kNN**: Requires features on same scale (distance-based)
- **Neural Network**: Converges faster with normalized inputs
- **Tree-based models**: Less affected but doesn't hurt
- **Prevents feature dominance**: Large-scale features don't dominate

**Example of why scaling matters:**
```
Without scaling:
  peak_hour_traffic: 0-5000 (large range)
  PM2.5: 0-50 (small range)
  
  Distance calculation dominated by peak_hour_traffic

With scaling:
  peak_hour_traffic: -2 to +2
  PM2.5: -2 to +2
  
  Both features contribute equally
```

---

## Cross-Validation Parameters

### Code:
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Parameters Explained:

#### `n_splits=5`
- **What it means**: Split data into 5 folds
- **How it works**:
  ```
  Fold 1: [Train][Train][Train][Train][Test]
  Fold 2: [Train][Train][Train][Test][Train]
  Fold 3: [Train][Train][Test][Train][Train]
  Fold 4: [Train][Test][Train][Train][Train]
  Fold 5: [Test][Train][Train][Train][Train]
  
  Average performance across all 5 folds
  ```

**Why 5 folds?**
- Standard practice
- Good balance: 80% train, 20% test per fold
- Not too few (3 would be unstable)
- Not too many (10 would be slow)

#### `shuffle=True`
- **What it means**: Randomly shuffle data before splitting
- **Why it matters**: Prevents bias from data ordering
- **Example**: If data is sorted by date, without shuffling each fold would have different time periods

#### `random_state=42`
- **What it means**: Seed for shuffle reproducibility
- **Why it matters**: Same folds every time we run

#### Stratified
- **What it means**: Each fold has same class distribution as original data
- **Why it matters**: Ensures balanced evaluation

**Example:**
```
Original data: 25% Very Low, 25% Low, 25% High, 25% Very High

Without stratification:
  Fold 1 might have: 40% Very Low, 20% Low, 30% High, 10% Very High
  Fold 2 might have: 10% Very Low, 30% Low, 20% High, 40% Very High
  → Inconsistent evaluation

With stratification:
  Every fold has: 25% Very Low, 25% Low, 25% High, 25% Very High
  → Consistent, fair evaluation
```

---

## Train-Test Split Parameters

### Code:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Parameters Explained:

#### `test_size=0.2`
- **What it means**: 20% of data for testing, 80% for training
- **Actual split**: 
  - Training: 55,693 samples (80%)
  - Testing: 13,924 samples (20%)

**Why 0.2?**
- Standard practice
- Enough test data for reliable evaluation
- Enough training data for good model learning

#### `random_state=42`
- **What it means**: Seed for reproducible split
- **Why it matters**: Same train/test split every time

#### `stratify=y`
- **What it means**: Maintain class distribution in both sets
- **Result**: Both train and test have ~25% of each class

---

## Summary Table

| Model | Key Parameters | Why These Values? | Performance |
|-------|----------------|-------------------|-------------|
| **kNN** | n_neighbors=5 | Odd number, prevents ties, not too small/large | 79.09% |
| **Decision Tree** | max_depth=10 | Prevents overfitting, moderate complexity | 95.03% |
| **Random Forest** | n_estimators=100, max_depth=15 | 100 trees for stability, deeper trees OK with ensemble | 96.35% 🥇 |
| **Neural Network** | hidden_layers=(100,50), max_iter=500 | Funnel architecture, enough iterations to converge | 95.48% |
| **XGBoost** | n_estimators=100, max_depth=6, learning_rate=0.1 | Balanced boosting, shallow trees, standard learning rate | 96.29% |

---

## How to Tune These Parameters

### If you want to improve performance:

#### Random Forest (already best):
```python
# Try more trees
n_estimators=200  # More stable, slower

# Try different depths
max_depth=20  # More complex patterns

# Try different feature sampling
max_features='log2'  # More randomness
```

#### XGBoost:
```python
# Lower learning rate + more trees
learning_rate=0.05
n_estimators=200

# Try different depths
max_depth=8

# Add regularization
reg_alpha=0.1  # L1 regularization
reg_lambda=1.0  # L2 regularization
```

#### Neural Network:
```python
# Try different architectures
hidden_layer_sizes=(200, 100, 50)  # Deeper

# More iterations
max_iter=1000

# Different activation
activation='tanh'  # Instead of 'relu'
```

---

## Key Takeaways

1. **Random Forest wins** with n_estimators=100, max_depth=15
2. **All models use random_state=42** for reproducibility
3. **Pipeline preprocessing** (imputation + scaling) applied to all
4. **5-fold stratified CV** ensures fair evaluation
5. **80/20 train-test split** with stratification
6. **Parameters are moderate** - not too aggressive, balanced for generalization

The current parameters are well-chosen for this dataset and achieve excellent performance (96.35% accuracy)!
