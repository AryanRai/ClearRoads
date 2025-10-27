# ClearRoads V2: Enhanced Traffic Congestion Prediction with Location Features

## Overview
This version implements **Strategy 3: Hybrid Approach** from the location feature strategies, incorporating regional grouping, distance to CBD, and urban classification alongside environmental predictors.

## Dataset
- **File**: `datasets/TrafficWeather_Beuro_AQ_withSuburb/complete_traffic_environment_data.csv`
- **Total Records**: 3,925,503
- **Date Range**: 2006-2025
- **Unique Suburbs**: 354
- **Environmental Data Coverage**: ~8.5% of records

## Q1 — Import and Descriptives (Enhanced)

### Data Loading
- Import complete merged dataset with traffic, air quality, BOM weather, and location features
- Drop unnecessary columns: `the_geom`, `cartodb_id`, `record_id`, `md5`, `updated_on`, `suburb`, `lga`
- Keep location features: `suburb_std`, `rms_region`, `distance_to_cbd_km`, coordinates

### Feature Categories
1. **Traffic Features** (25): `daily_total`, `hour_00` to `hour_23`
2. **Air Quality Features** (5): `PM10`, `PM2_5`, `NO2`, `NO`, `CO`
3. **Weather Features** (4): `rainfall_mm`, `solar_exposure_mj`, `min_temp_c`, `max_temp_c`
4. **Location Features** (5): `suburb_std`, `rms_region`, `distance_to_cbd_km`, `wgs84_latitude`, `wgs84_longitude`
5. **Temporal Features** (6): `year`, `month`, `day_of_week`, `public_holiday`, `school_holiday`, `date`
6. **Infrastructure** (4): `station_key`, `traffic_direction_seq`, `cardinal_direction_seq`, `classification_seq`

### Descriptive Analysis
- Show data completeness for environmental features
- Display regional distribution (Sydney, Hunter, Southern, Western, Northern, South West)
- Analyze distance to CBD statistics
- Identify records with environmental data

## Q2 — Data Cleaning & Preprocessing (Enhanced)

### Missing Values Strategy
1. **Filter to environmental data**: Keep only records with ANY environmental features (~333,705 records, 8.5%)
2. **Remove missing targets**: Drop rows where `daily_total` is missing
3. **Remove missing location**: Drop rows missing `suburb_std`, `rms_region`, or `distance_to_cbd_km`
4. **Impute remaining**: Use suburb-specific medians for air quality and weather features

### Feature Engineering

#### Temporal Features
- `is_weekend`: Binary indicator for Saturday/Sunday
- `season`: Categorical (Summer, Autumn, Winter, Spring) based on month

#### Traffic Pattern Features
- `morning_rush`: Sum of traffic from hours 6-9
- `evening_rush`: Sum of traffic from hours 16-19
- `peak_hour_traffic`: Maximum hourly traffic value

#### Air Quality Features
- `AQI_composite`: Weighted composite index
  - PM2.5: 30%
  - PM10: 25%
  - NO2: 25%
  - CO: 10%
  - NO: 10%

#### Location Features (Strategy 3: Hybrid Approach)

**1. Regional Grouping** (`rms_region`)
- Already in dataset from RMS classification
- Categories: Sydney, Hunter, Southern, Western, Northern, South West
- One-hot encoded for modeling

**2. Distance to CBD** (`distance_to_cbd_km`)
- Continuous feature (0.6 km to 928 km)
- Already calculated in dataset
- Captures urban-rural gradient

**3. Urban Classification** (`urban_type`)
- Created from suburb mapping
- Categories:
  - **Urban**: Inner Sydney, high-density areas (CBD, Alexandria, Bondi, etc.)
  - **Suburban**: Outer Sydney suburbs (Parramatta, Blacktown, Penrith, etc.)
  - **Regional_City**: Major regional centers (Newcastle, Wollongong, etc.)
  - **Regional**: Rural and remote areas
- One-hot encoded for modeling

### Data Quality
- Remove outliers: < 1st percentile or > 99th percentile of `daily_total`
- Validate data types and conversions
- Handle boolean columns (`public_holiday`, `school_holiday`)

## Q3 — Create Target: Traffic Congestion Class

### Target Definition (4-Class System)
Based on `daily_total` percentiles:
- **Very Low**: < 25th percentile
- **Low**: 25th-50th percentile
- **High**: 50th-75th percentile
- **Very High**: ≥ 75th percentile

### Analysis
- Show class distribution (counts and percentages)
- Calculate majority-class baseline accuracy
- Plot bar chart of class frequencies
- Ensure balanced representation

## Q4 — Model Development (Enhanced)

### Feature Selection

**Total Features: ~30-40** (depending on one-hot encoding)

1. **Air Quality** (6): PM10, PM2.5, NO2, NO, CO, AQI_composite
2. **Weather** (4): rainfall_mm, solar_exposure_mj, min_temp_c, max_temp_c
3. **Traffic Patterns** (3): morning_rush, evening_rush, peak_hour_traffic
4. **Temporal** (7): month, day_of_week, year, public_holiday, school_holiday, is_weekend, season (one-hot)
5. **Location - Continuous** (1): distance_to_cbd_km
6. **Location - Categorical** (10-15): 
   - rms_region (one-hot: 6 categories)
   - urban_type (one-hot: 4 categories)

### Models
1. **k-Nearest Neighbors** (k=5): Baseline
2. **Decision Tree** (max_depth=10): Interpretable
3. **Random Forest** (n_estimators=100, max_depth=15): Ensemble
4. **Neural Network** (MLP: 100-50 hidden layers): Non-linear
5. **XGBoost** (n_estimators=100, max_depth=6): High-performance

### Pipeline
```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', Model())
])
```

### Evaluation
- 5-fold Stratified Cross-Validation
- Test set: 20% (stratified)
- Metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrices for all models

## Q5 — Performance Analysis (Enhanced)

### Model Comparison
- Compare all models vs baseline
- Identify best-performing model
- Analyze class-specific performance

### Feature Importance Analysis
For tree-based models (Random Forest, XGBoost):

**Feature Categories:**
1. **Traffic Patterns**: Expected to dominate (70-85%)
2. **Location Features**: NEW - Expected contribution (5-10%)
3. **Air Quality**: Expected (3-8%)
4. **Weather**: Expected (2-5%)
5. **Temporal**: Expected (2-5%)

### Location Feature Impact
- Analyze importance of:
  - Regional grouping (rms_region)
  - Distance to CBD
  - Urban classification
- Compare performance across regions
- Identify location-specific patterns

### Expected Results

#### Before (Original Model - V1)
- Features: 19
- Best Accuracy: ~96.35% (Random Forest)
- No location information

#### After (Enhanced Model - V2)
- Features: 30-40
- Expected Accuracy: 96.8-97.5%
- Expected Improvement: +0.5-1.2%
- Location features contribute 5-10% to predictions

### Visualizations
1. Congestion class distribution
2. Confusion matrices (all models)
3. Feature importance plots (Random Forest, XGBoost)
4. Category-wise importance breakdown

## Implementation Notes

### Advantages of This Approach
✅ Captures regional traffic patterns  
✅ Generalizes to new suburbs (via distance/region)  
✅ Interpretable location features  
✅ Balances complexity and performance  
✅ Demonstrates sophisticated feature engineering  

### Computational Considerations
- Dataset filtered to ~333k records (with environmental data)
- Training time: 5-15 minutes depending on model
- Memory usage: ~2-3 GB
- Parallel processing enabled (n_jobs=-1)

### Output Files
- `congestion_class_distribution_v2.png`
- `confusion_matrices_v2.png`
- `feature_importance_random_forest_v2.png`
- `feature_importance_xgboost_v2.png`

## Next Steps
1. Run `python traffic_analysis_v2.py`
2. Analyze feature importance results
3. Compare V2 vs V1 performance
4. Document findings in report
5. Consider hyperparameter tuning if needed
