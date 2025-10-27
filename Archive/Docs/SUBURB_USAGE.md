# How suburb_std is Used in the Model

## Short Answer

**YES, we use `suburb_std`** - but **ONLY for data cleaning**, **NOT for prediction**.

---

## Detailed Breakdown

### Where suburb_std Appears in the Code:

#### 1. **Data Loading & Exploration** (Line 58)
```python
categorical_cols = ['station_key', 'traffic_direction_seq', 'cardinal_direction_seq', 
                   'classification_seq', 'suburb_std']
```
- Listed as a categorical column for documentation
- Shown in the console output
- **Not used for modeling**

#### 2. **Data Cleaning - Missing Value Imputation** (Line 128-131)
```python
for col in air_quality_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean.groupby('suburb_std')[col].transform(
            lambda x: x.fillna(x.median())
        )
```
- **This is the ONLY functional use of suburb_std**
- Used to impute missing air quality values with suburb-specific medians

#### 3. **Feature Selection** (prepare_features function)
```python
air_quality_features = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO', 'AQI_composite']
temporal_features = ['month', 'day_of_week', 'public_holiday', 'school_holiday', 
                    'is_weekend', 'year']
traffic_features = ['morning_rush', 'evening_rush', 'peak_hour_traffic']
```
- **suburb_std is NOT included in any feature list**
- **NOT used as a predictor**

---

## What Does This Mean?

### ✅ suburb_std IS Used For:

**Data Cleaning Only** - Imputing missing air quality values

**Example:**
```python
# If PM10 is missing for a Parramatta observation:
# Fill it with the median PM10 value from ALL Parramatta observations

Parramatta observations:
  Date 1: PM10 = 15.3
  Date 2: PM10 = NaN  ← Missing!
  Date 3: PM10 = 18.2
  Date 4: PM10 = 12.5

Median PM10 for Parramatta = 15.3

Date 2: PM10 = 15.3  ← Filled with Parramatta's median
```

**Why this approach?**
- More accurate than using global median (all suburbs combined)
- Accounts for location-specific air quality patterns
- Urban suburbs have different pollution levels than rural suburbs
- Preserves local environmental characteristics

### ❌ suburb_std is NOT Used For:

1. **NOT a predictor feature** - Not included in the model training
2. **NOT used for predictions** - Model doesn't know which suburb an observation is from
3. **NOT used for suburb-specific models** - One model for all locations
4. **NOT used for filtering** - All suburbs are included in training

---

## Visual Flow

```
suburb_std Usage Flow:

┌─────────────────────────────────────────────────────────────┐
│ 1. DATA LOADING                                             │
│    ✓ suburb_std loaded from CSV                             │
│    ✓ Listed as categorical column (for documentation)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA CLEANING (ONLY FUNCTIONAL USE)                      │
│    ✓ Group by suburb_std                                    │
│    ✓ Calculate median PM10, PM2.5, NO2, NO, CO per suburb   │
│    ✓ Fill missing values with suburb-specific medians       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. FEATURE SELECTION                                        │
│    ✗ suburb_std NOT selected as a feature                   │
│    ✓ Only 19 features selected (air quality, temporal, etc.)│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. MODEL TRAINING                                           │
│    ✗ suburb_std NOT in training data                        │
│    ✓ Model learns from 19 features only                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. PREDICTION                                               │
│    ✗ suburb_std NOT needed for predictions                  │
│    ✓ Model predicts using 19 features only                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Why Not Use suburb_std as a Predictor?

### Reasons it's excluded from modeling:

1. **High Cardinality**
   - 50+ unique suburbs in the dataset
   - Would require 50+ one-hot encoded features
   - Increases model complexity significantly

2. **Data Imbalance**
   - Some suburbs have 1000s of observations
   - Others have only dozens
   - Model would be biased toward well-represented suburbs

3. **Generalization Goal**
   - Want a model that works for ANY location
   - Including suburb limits applicability to only known suburbs
   - Can't predict for new traffic stations in unobserved suburbs

4. **Overfitting Risk**
   - Model might memorize suburb-specific patterns
   - Poor generalization to new locations
   - Reduces model robustness

5. **Location Information Already Captured**
   - Air quality values implicitly encode location
   - Urban areas have different pollution profiles than rural
   - Traffic patterns reflect location characteristics

---

## Could We Use suburb_std as a Predictor?

### Yes, but with trade-offs:

#### Option 1: One-Hot Encoding
```python
# Add to feature selection:
suburb_dummies = pd.get_dummies(df['suburb_std'], prefix='suburb')
X = pd.concat([X, suburb_dummies], axis=1)
```

**Pros:**
- Captures suburb-specific patterns
- May improve accuracy for known suburbs

**Cons:**
- Adds 50+ features (one per suburb)
- Doesn't work for new suburbs
- Risk of overfitting
- Computationally expensive

#### Option 2: Separate Models per Suburb
```python
# Train one model for each suburb
for suburb in df['suburb_std'].unique():
    suburb_data = df[df['suburb_std'] == suburb]
    model = train_model(suburb_data)
    models[suburb] = model
```

**Pros:**
- Highly customized predictions per suburb
- Captures local patterns perfectly

**Cons:**
- Requires sufficient data per suburb
- 50+ models to maintain
- Can't predict for new suburbs
- Some suburbs have too little data

#### Option 3: Hierarchical/Mixed Effects Model
```python
# Use suburb as a random effect
# Shares information across suburbs while allowing customization
```

**Pros:**
- Best of both worlds
- Generalizes to new suburbs

**Cons:**
- More complex implementation
- Requires specialized libraries

---

## Current Approach: Why It Works

### Benefits of NOT using suburb_std as a predictor:

1. **Simplicity** ✅
   - Only 19 features
   - Easy to interpret
   - Fast training and prediction

2. **Generalization** ✅
   - Works for any location with required features
   - Can be applied to new traffic stations
   - Not limited to suburbs in training data

3. **Robustness** ✅
   - Learns general patterns
   - Less prone to overfitting
   - Stable predictions across locations

4. **Practical Deployment** ✅
   - Single model to maintain
   - No need to know suburb for prediction
   - Works with incomplete location information

5. **Still Captures Location Effects** ✅
   - Air quality values differ by location
   - Traffic patterns reflect urban vs rural
   - Temporal features capture regional patterns

---

## Summary Table

| Aspect | suburb_std Usage |
|--------|------------------|
| **Loaded from CSV?** | ✅ Yes |
| **Used for data cleaning?** | ✅ Yes (imputing missing values) |
| **Used as a predictor?** | ❌ No |
| **Included in model training?** | ❌ No |
| **Needed for predictions?** | ❌ No |
| **Creates suburb-specific models?** | ❌ No |
| **Purpose** | Data quality improvement only |

---

## Key Takeaway

**suburb_std is used ONLY for improving data quality** (filling missing air quality values with location-appropriate medians), **NOT for making predictions**.

The model is **location-agnostic** - it predicts congestion class based on environmental and temporal features, regardless of which suburb the observation comes from.

This design choice prioritizes **generalization** and **simplicity** over **location-specific customization**.
