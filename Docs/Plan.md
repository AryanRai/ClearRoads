# ClearRoads: Traffic Congestion Prediction - Analysis Plan

## Q1 — Import and Descriptives 
- Import required Python packages: pandas, numpy, sklearn, matplotlib, seaborn
- Load the merged traffic-air quality dataset from `output_merge.csv` (71,039 records)
- Drop unnecessary columns: `the_geom`, `the_geom_webmercator`, `cartodb_id`, `record_id`, `md5`, `updated_on`, `suburb` (keeping `suburb_std`)
- Show final column list and provide descriptive statistics for:
  - Traffic features: `daily_total`, hourly columns (`hour_00` to `hour_23`)
  - Air quality features: `PM10`, `PM2_5`, `NO2`, `NO`, `CO`
  - Temporal features: `year`, `month`, `day_of_week`, `public_holiday`, `school_holiday`
  - Categorical: `station_key`, `traffic_direction_seq`, `cardinal_direction_seq`, `classification_seq`, `suburb_std`

## Q2 — Data Cleaning & Preprocessing Plan 

**Missing Values Strategy:**
- Air quality columns (`PM10`, `PM2_5`, `NO2`, `NO`, `CO`) have significant missing values due to spatial mismatch between traffic and air quality stations
- Options: (1) Drop rows with any missing air quality data, (2) Impute with mean/median by suburb, (3) Forward-fill by date within suburb
- Hourly traffic columns (`hour_00` to `hour_23`) have some missing values - impute with 0 or drop rows where `daily_total` is missing
- **Recommended approach**: Drop rows where all air quality features are missing; impute remaining missing air quality values with suburb-specific medians

**Feature Engineering:**
- Create composite Air Quality Index (AQI) from PM2.5, PM10, NO2, CO
- Extract temporal features: `is_weekend`, `season`, `is_peak_hour` (based on hourly distribution)
- Calculate traffic metrics: `peak_hour_traffic` (max of hourly values), `morning_rush` (sum of hours 6-9), `evening_rush` (sum of hours 16-19)
- Normalize/standardize continuous features for model training

**Categorical Encoding:**
- One-hot encode: `day_of_week`, `month`, `suburb_std` (if using as feature)
- Binary encode: `public_holiday`, `school_holiday` (already boolean)
- Keep numeric: `station_key`, direction sequences (treat as identifiers or encode if needed)

**Data Quality Checks:**
- Remove outliers in `daily_total` (e.g., values > 99th percentile or < 1st percentile)
- Validate that hourly sums approximately match `daily_total`
- Check for duplicate records based on `station_key` + `date` + `traffic_direction_seq`

## Q3 — Create Target: Traffic Congestion Class

**Target Definition:**
Create a multiclass target `Congestion_Class` based on `daily_total` traffic volume:
- **Low**: daily_total < 25th percentile
- **Medium**: 25th percentile ≤ daily_total < 75th percentile  
- **High**: daily_total ≥ 75th percentile

Alternative 4-class system:
- **Very Low**: < 25th percentile
- **Low**: 25th-50th percentile
- **High**: 50th-75th percentile
- **Very High**: ≥ 75th percentile

**Analysis:**
- Show class distribution (counts and percentages)
- Compute majority-class baseline accuracy
- Plot bar chart of class frequencies
- Analyze class distribution across suburbs and time periods

## Q4 — Model Development

**Feature Selection:**
- **Air Quality Features**: `PM10`, `PM2_5`, `NO2`, `NO`, `CO` (primary predictors per proposal)
- **Temporal Features**: `month`, `day_of_week`, `public_holiday`, `school_holiday`, `year`
- **Engineered Features**: `is_weekend`, `season`, composite AQI
- **Optional**: `station_key` (as categorical), `suburb_std` (location proxy)

**Data Preparation:**
- Handle missing values as per Q2 strategy
- Split data: 80% train, 20% test (stratified by `Congestion_Class`)
- Consider temporal split: train on earlier years, test on recent data to validate temporal generalization

**Models to Implement (per proposal):**
1. **k-Nearest Neighbors (kNN)**: Simple baseline, test k=3,5,7,9
2. **Decision Tree**: Interpretable, shows feature importance
3. **Random Forest**: Robust ensemble method, handles missing data well
4. **Neural Network (MLP)**: Captures non-linear relationships
5. **XGBoost**: High-performance gradient boosting

**Pipeline Structure:**
```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', Model())
])
```

**Evaluation:**
- Use 5-fold Stratified Cross-Validation on training set
- Report mean ± std accuracy for each model
- Evaluate on test set: accuracy, confusion matrix, classification report
- Compare feature importance across tree-based models
- Analyze performance by suburb and season

**Hyperparameter Tuning:**
- Random Forest: n_estimators, max_depth, min_samples_split
- XGBoost: learning_rate, max_depth, n_estimators
- Neural Network: hidden_layer_sizes, activation, alpha
- kNN: n_neighbors, weights, metric

## Q5 — Performance Analysis & Environmental Impact

**Class-Specific Metrics:**
- Extract precision, recall, F1-score, and support for each congestion class
- Compare to majority baseline accuracy from Q3
- Identify which congestion levels are easiest/hardest to predict

**Environmental Correlation Analysis:**
- Analyze feature importance: which air quality factors most influence predictions?
- Correlation matrix between air quality features and traffic congestion
- Temporal analysis: how do predictions vary by season, holiday periods?
- Spatial analysis: performance differences across suburbs

**Model Comparison:**
- Create comparison table of all models with accuracy, precision, recall, F1
- Identify best-performing model for deployment
- Discuss trade-offs: accuracy vs interpretability vs computational cost

**Insights for Policy Recommendations:**
- Which environmental conditions most strongly predict high congestion?
- Are there actionable patterns (e.g., high PM2.5 → reduced traffic)?
- Validate the traffic-pollution feedback loop hypothesis from proposal
