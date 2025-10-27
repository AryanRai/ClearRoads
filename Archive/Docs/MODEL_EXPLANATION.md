# ClearRoads Model Explanation

## What Does the Model Predict?

### 🎯 Target Variable (Dependent Variable)

**Congestion_Class** - A 4-category classification of traffic congestion levels:

| Class | Definition | Traffic Volume Range |
|-------|------------|---------------------|
| **Very Low** | < 25th percentile | < 533 vehicles/day |
| **Low** | 25th-50th percentile | 533-1,443 vehicles/day |
| **High** | 50th-75th percentile | 1,443-6,154 vehicles/day |
| **Very High** | ≥ 75th percentile | ≥ 6,154 vehicles/day |

**Important**: The model predicts the **congestion class**, NOT the exact number of vehicles. It's a **classification problem**, not a regression problem.

---

## 📊 Independent Variables (Predictors/Features)

The model uses **19 features** across 3 categories:

### 1. Air Quality Features (6 features)
- `PM10` - Particulate matter (10 micrometers) in μg/m³
- `PM2_5` - Fine particulate matter (2.5 micrometers) in μg/m³
- `NO2` - Nitrogen dioxide in ppm
- `NO` - Nitrogen oxide in ppm
- `CO` - Carbon monoxide in ppm
- `AQI_composite` - Weighted composite air quality index

**Source**: NSW EPA air quality monitoring stations, matched to traffic locations by suburb

### 2. Temporal Features (6 features)
- `month` - Month of year (1-12)
- `day_of_week` - Day of week (1-7)
- `public_holiday` - Binary flag (0/1)
- `school_holiday` - Binary flag (0/1)
- `is_weekend` - Binary flag (0/1)
- `year` - Year of observation

**Purpose**: Capture seasonal patterns, weekly cycles, and holiday effects

### 3. Traffic Pattern Features (3 features)
- `morning_rush` - Sum of traffic during hours 6-9 AM
- `evening_rush` - Sum of traffic during hours 4-7 PM
- `peak_hour_traffic` - Maximum hourly traffic in the day

**Source**: Derived from hourly traffic counts (hour_00 to hour_23)

### 4. Seasonal Features (4 features - one-hot encoded)
- `season_Autumn` - Binary flag
- `season_Spring` - Binary flag
- `season_Summer` - Binary flag
- `season_Winter` - Binary flag

**Purpose**: Capture seasonal variations in traffic and air quality

---

## 🗺️ Is This Suburb-Based?

### Short Answer: **NO, it's NOT suburb-specific predictions**

### Detailed Explanation:

#### What the Model Does:
- **Trains on data from ALL suburbs combined** (69,617 observations across multiple suburbs)
- **Learns general patterns** that apply across different locations
- **Predicts congestion class** for ANY observation with the required features

#### What the Model Does NOT Do:
- Does NOT predict "congestion in Suburb X"
- Does NOT create separate models for each suburb
- Does NOT use suburb as a predictor feature

#### How Suburb Information is Used:
1. **During Data Cleaning**: Suburb is used to impute missing air quality values
   ```python
   df_clean[col] = df_clean.groupby('suburb_std')[col].transform(
       lambda x: x.fillna(x.median())
   )
   ```
   This fills missing PM10 in "Parramatta" with Parramatta's median PM10.

2. **For Air Quality Matching**: The original data merging process matched traffic stations with nearby air quality stations by suburb

3. **NOT Used in Prediction**: Suburb is dropped before model training

---

## 🔄 How the Prediction Works

### Example Prediction Scenario:

**Input** (a single observation):
```python
{
    'PM10': 15.3,
    'PM2_5': 6.2,
    'NO2': 0.11,
    'NO': 0.47,
    'CO': 0.11,
    'AQI_composite': 5.8,
    'month': 4,
    'day_of_week': 3,
    'public_holiday': 0,
    'school_holiday': 1,
    'is_weekend': 0,
    'year': 2020,
    'morning_rush': 1200,
    'evening_rush': 1500,
    'peak_hour_traffic': 450,
    'season_Autumn': 1,
    'season_Spring': 0,
    'season_Summer': 0,
    'season_Winter': 0
}
```

**Output** (prediction):
```python
Congestion_Class = "High"  # or "Very Low", "Low", "Very High"
```

**Interpretation**: 
Given these air quality conditions, temporal factors, and traffic patterns, the model predicts **High congestion** (1,443-6,154 vehicles/day).

---

## 📈 What Drives the Predictions?

### Feature Importance (Random Forest):

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | peak_hour_traffic | 37.80% | **Strongest predictor** - if peak hour is busy, day is busy |
| 2 | evening_rush | 30.04% | Evening traffic volume highly predictive |
| 3 | morning_rush | 23.61% | Morning traffic volume highly predictive |
| 4 | CO | 1.21% | Carbon monoxide levels matter |
| 5 | NO2 | 1.14% | Nitrogen dioxide levels matter |
| 6 | PM10 | 0.91% | Particulate matter matters |
| 7 | AQI_composite | 0.85% | Overall air quality index |
| 8 | PM2_5 | 0.85% | Fine particulate matter |
| 9 | NO | 0.83% | Nitrogen oxide |
| 10 | day_of_week | 0.82% | Day of week patterns |

### Key Insights:

1. **Traffic patterns dominate** (91.45% combined importance)
   - The model primarily learns from historical traffic patterns
   - Peak hour traffic is the single strongest predictor

2. **Air quality contributes** (5.79% combined importance)
   - While smaller, air quality features are consistently important
   - CO and NO2 are the most predictive pollutants
   - Validates the project hypothesis about environmental-traffic relationships

3. **Temporal factors matter** (0.82% for day_of_week, plus seasonal effects)
   - Weekday vs weekend patterns
   - Seasonal variations
   - Holiday effects

---

## 🤔 Why Not Use Suburb as a Predictor?

### Reasons Suburb is NOT Included:

1. **High Cardinality**: Too many unique suburbs (50+ categories)
   - Would require 50+ one-hot encoded features
   - Risk of overfitting to specific locations
   - Poor generalization to new suburbs

2. **Data Sparsity**: Not all suburbs have equal representation
   - Some suburbs have 1000s of observations
   - Others have only a few dozen
   - Model would be biased toward well-represented suburbs

3. **Generalization Goal**: We want a model that works anywhere
   - Should predict congestion for ANY location with the required features
   - Not limited to suburbs in the training data
   - Can be applied to new traffic stations

4. **Location Information is Implicit**: 
   - Air quality values already capture location characteristics
   - Urban areas have different pollution profiles than rural areas
   - Traffic patterns reflect location type (urban vs suburban vs rural)

---

## 🎯 What Questions Can the Model Answer?

### ✅ Questions the Model CAN Answer:

1. **"Given these air quality conditions and traffic patterns, what congestion level should we expect?"**
   - Input: PM10=20, PM2.5=8, morning_rush=1500, etc.
   - Output: "High" congestion

2. **"On a weekday in autumn with moderate air quality, what's the likely congestion?"**
   - Input: day_of_week=3, season_Autumn=1, AQI_composite=6, etc.
   - Output: Congestion class prediction

3. **"How does air quality affect traffic congestion predictions?"**
   - Answer: Feature importance shows CO, NO2, PM10 contribute 1-6%
   - Higher pollution correlates with certain congestion patterns

4. **"What are the most important factors for predicting congestion?"**
   - Answer: Peak hour traffic (38%), evening rush (30%), morning rush (24%)

### ❌ Questions the Model CANNOT Answer:

1. **"What will traffic be like in Parramatta tomorrow?"**
   - Model doesn't predict for specific suburbs
   - Would need suburb-specific model or include suburb as feature

2. **"How many vehicles will be on the road?"**
   - Model predicts congestion CLASS, not exact vehicle count
   - Would need regression model for exact numbers

3. **"Which suburb has the worst traffic?"**
   - Model doesn't compare suburbs
   - Would need separate analysis of suburb-level data

4. **"What will traffic be like at 3 PM specifically?"**
   - Model predicts daily congestion class, not hourly traffic
   - Would need time-of-day specific model

---

## 🔮 How to Use the Model in Practice

### Scenario 1: Daily Congestion Forecasting
```python
# Tomorrow's forecast
input_features = {
    'PM10': 18.5,  # Forecasted air quality
    'PM2_5': 7.2,
    'NO2': 0.15,
    'month': 10,  # October
    'day_of_week': 2,  # Tuesday
    'public_holiday': 0,
    'school_holiday': 0,
    # ... other features
}

prediction = model.predict(input_features)
# Output: "High" congestion expected
```

### Scenario 2: Environmental Impact Analysis
```python
# Compare normal vs high pollution day
normal_pollution = {'PM2_5': 5, 'PM10': 12, 'NO2': 0.08, ...}
high_pollution = {'PM2_5': 25, 'PM10': 45, 'NO2': 0.35, ...}

pred_normal = model.predict(normal_pollution)
pred_high = model.predict(high_pollution)

# Analyze if pollution affects congestion predictions
```

### Scenario 3: Policy Planning
```python
# Test different scenarios
weekday_school = {'day_of_week': 3, 'school_holiday': 0, ...}
weekday_no_school = {'day_of_week': 3, 'school_holiday': 1, ...}

# See how school holidays affect congestion predictions
```

---

## 📊 Model Type Summary

| Aspect | Description |
|--------|-------------|
| **Problem Type** | Multi-class Classification |
| **Target** | Congestion_Class (4 categories) |
| **Predictors** | 19 features (air quality + temporal + traffic patterns) |
| **Scope** | General model across all locations |
| **Granularity** | Daily congestion level |
| **Location** | Not suburb-specific |
| **Best Model** | Random Forest (96.35% accuracy) |

---

## 🎓 Key Takeaways

1. **The model predicts congestion CLASS** (Very Low/Low/High/Very High), not exact vehicle counts

2. **It's a GENERAL model** that works across all locations, not suburb-specific

3. **Traffic patterns are the strongest predictors** (91%), but air quality contributes meaningfully (6%)

4. **Suburb information is used for data cleaning** but not as a predictor feature

5. **The model achieves 96.35% accuracy**, meaning it correctly classifies congestion level 96% of the time

6. **It validates the hypothesis** that environmental factors (air quality) can help predict traffic patterns

---

## 💡 Future Enhancements

To make the model suburb-specific, you could:

1. **Add suburb as a feature** (one-hot encoded)
   - Pro: Captures location-specific patterns
   - Con: Doesn't generalize to new suburbs

2. **Train separate models per suburb**
   - Pro: Highly customized predictions
   - Con: Requires sufficient data per suburb

3. **Use hierarchical modeling**
   - Pro: Shares information across suburbs while allowing customization
   - Con: More complex implementation

4. **Add geographic features**
   - Distance to CBD, road type, nearby amenities
   - Captures location characteristics without explicit suburb labels

Currently, the model prioritizes **generalization** over **location-specificity**, making it applicable to any traffic station with the required features.
