# Adding Suburb/Location Features to the Model

## Problem Statement

Currently, the model doesn't use location information (suburb) as a predictor. We want to incorporate location while avoiding:
- ❌ High cardinality (50+ unique suburbs)
- ❌ Overfitting to specific locations
- ❌ Poor generalization to new suburbs

## Recommended Strategies

---

## Strategy 1: Regional Grouping (RECOMMENDED)

### Approach
Group suburbs into 5-10 regions based on geographic/demographic similarity, then one-hot encode.

### Advantages
✅ Reduces features from 50+ to 5-10  
✅ Captures regional patterns  
✅ Generalizes better than individual suburbs  
✅ Interpretable (e.g., "Inner West", "Northern Beaches")

### Implementation Steps

1. **Get unique suburbs from your data**
2. **Group into regions** (see AI prompt below)
3. **Create mapping dictionary**
4. **Add to model as one-hot encoded features**

### Example Grouping
```python
region_mapping = {
    # Inner Sydney
    'ROZELLE': 'Inner_Sydney',
    'ALEXANDRIA': 'Inner_Sydney',
    'ULTIMO': 'Inner_Sydney',
    
    # Western Sydney
    'PARRAMATTA': 'Western_Sydney',
    'BLACKTOWN': 'Western_Sydney',
    'PENRITH': 'Western_Sydney',
    
    # Northern Sydney
    'CHATSWOOD': 'Northern_Sydney',
    'HORNSBY': 'Northern_Sydney',
    
    # Southern Sydney
    'SUTHERLAND': 'Southern_Sydney',
    'HURSTVILLE': 'Southern_Sydney',
    
    # Eastern Sydney
    'BONDI': 'Eastern_Sydney',
    'RANDWICK': 'Eastern_Sydney',
    
    # Hunter Region
    'NEWCASTLE': 'Hunter_Region',
    'MAITLAND': 'Hunter_Region',
    
    # Illawarra
    'WOLLONGONG': 'Illawarra',
    'SHELLHARBOUR': 'Illawarra',
    
    # Regional NSW
    'ALBURY': 'Regional_NSW',
    'WAGGA WAGGA': 'Regional_NSW',
}
```

### Code Implementation
```python
# In prepare_features() function, add:

# Create region feature
df['region'] = df['suburb_std'].map(region_mapping)

# One-hot encode regions
region_dummies = pd.get_dummies(df['region'], prefix='region')
df = pd.concat([df, region_dummies], axis=1)

# Add to feature list
feature_cols.extend(region_dummies.columns.tolist())
```

### Expected Impact
- **Features added**: 5-10 (one per region)
- **Accuracy improvement**: +0.5-2%
- **Interpretability**: High (can see which regions have different patterns)

---

## Strategy 2: Distance to CBD

### Approach
Calculate distance from each suburb to Sydney CBD, use as continuous feature.

### Advantages
✅ Single feature (no high cardinality)  
✅ Captures urban vs suburban vs rural gradient  
✅ Generalizes to new suburbs (just calculate distance)  
✅ Continuous variable (more information than categories)

### Implementation Steps

1. **Get CBD coordinates**: Sydney CBD ≈ (-33.8688, 151.2093)
2. **Get suburb coordinates** from your station reference data
3. **Calculate Haversine distance**
4. **Add as feature**

### Code Implementation
```python
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points"""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

# Sydney CBD coordinates
CBD_LAT = -33.8688
CBD_LON = 151.2093

# Calculate distance for each suburb
# (You'll need to get lat/lon from station reference data)
suburb_coords = {
    'PARRAMATTA': (-33.8150, 151.0000),
    'NEWCASTLE': (-32.9283, 151.7817),
    # ... add all suburbs
}

df['distance_to_cbd'] = df['suburb_std'].map(
    lambda suburb: haversine_distance(
        CBD_LAT, CBD_LON, 
        suburb_coords[suburb][0], 
        suburb_coords[suburb][1]
    )
)

# Add to features
feature_cols.append('distance_to_cbd')
```

### Expected Impact
- **Features added**: 1
- **Accuracy improvement**: +0.3-1%
- **Interpretability**: High (distance effect on traffic)

---

## Strategy 3: Hybrid Approach (BEST)

### Approach
Combine regional grouping + distance to CBD + urban/rural classification.

### Features Created
1. **Region** (5-10 categories, one-hot encoded)
2. **Distance to CBD** (continuous, km)
3. **Urban Type** (3 categories: Urban/Suburban/Regional, one-hot encoded)

### Advantages
✅ Captures multiple aspects of location  
✅ Redundancy helps model learn robust patterns  
✅ Best of both categorical and continuous  
✅ Most likely to improve accuracy

### Code Implementation
```python
# 1. Regional grouping (as in Strategy 1)
df['region'] = df['suburb_std'].map(region_mapping)
region_dummies = pd.get_dummies(df['region'], prefix='region')

# 2. Distance to CBD (as in Strategy 2)
df['distance_to_cbd'] = df['suburb_std'].map(
    lambda s: haversine_distance(CBD_LAT, CBD_LON, suburb_coords[s][0], suburb_coords[s][1])
)

# 3. Urban classification
urban_classification = {
    'SYDNEY': 'Urban',
    'PARRAMATTA': 'Urban',
    'NEWCASTLE': 'Urban',
    'PENRITH': 'Suburban',
    'CAMPBELLTOWN': 'Suburban',
    'ALBURY': 'Regional',
    'WAGGA WAGGA': 'Regional',
    # ... classify all suburbs
}

df['urban_type'] = df['suburb_std'].map(urban_classification)
urban_dummies = pd.get_dummies(df['urban_type'], prefix='urban')

# Combine all
df = pd.concat([df, region_dummies, urban_dummies], axis=1)
feature_cols.extend(region_dummies.columns.tolist())
feature_cols.append('distance_to_cbd')
feature_cols.extend(urban_dummies.columns.tolist())
```

### Expected Impact
- **Features added**: 8-15 total
- **Accuracy improvement**: +1-3%
- **Interpretability**: Very high

---

## Strategy 4: Suburb Embeddings (ADVANCED)

### Approach
Use target encoding or embeddings to represent suburbs as continuous values.

### Target Encoding
```python
# Calculate mean congestion for each suburb
suburb_encoding = df.groupby('suburb_std')['daily_total'].mean()

# Map to dataframe
df['suburb_avg_traffic'] = df['suburb_std'].map(suburb_encoding)

# Add to features
feature_cols.append('suburb_avg_traffic')
```

### Advantages
✅ Single continuous feature  
✅ Captures suburb-specific traffic levels  
✅ Works for new suburbs (use global mean)

### Disadvantages
⚠️ Risk of data leakage (using target in features)  
⚠️ Requires careful cross-validation

---

## Comparison Table

| Strategy | Features Added | Complexity | Generalization | Expected Improvement | Interpretability |
|----------|----------------|------------|----------------|---------------------|------------------|
| **Regional Grouping** | 5-10 | Low | High | +0.5-2% | High |
| **Distance to CBD** | 1 | Very Low | Very High | +0.3-1% | High |
| **Hybrid** | 8-15 | Medium | High | +1-3% | Very High |
| **Target Encoding** | 1 | Low | Medium | +0.5-1.5% | Medium |

---

## Recommended Approach

### For Your Project: **Hybrid Approach (Strategy 3)**

**Why?**
1. Balances complexity and performance
2. Multiple location features provide redundancy
3. Highly interpretable for your report
4. Most likely to show measurable improvement
5. Demonstrates sophisticated feature engineering

### Implementation Priority
1. **Start with Regional Grouping** (easiest, good impact)
2. **Add Distance to CBD** (simple, continuous feature)
3. **Add Urban Classification** (completes the picture)

---

## Getting Your Suburb List

First, let's see what suburbs you have:

```python
# Run this to get your unique suburbs
import pandas as pd

df = pd.read_csv("datasets/TrafficWeatherwithSuburb/roadandweathermerged-20251020T083319Z-1-001/roadandweathermerged/output_merge.csv")

suburbs = df['suburb_std'].unique()
print(f"Total unique suburbs: {len(suburbs)}")
print("\nSuburbs:")
for suburb in sorted(suburbs):
    print(f"  - {suburb}")

# Get counts
suburb_counts = df['suburb_std'].value_counts()
print("\nTop 10 suburbs by observation count:")
print(suburb_counts.head(10))
```

---

## Next Steps

1. **Run the code above** to get your suburb list
2. **Use the AI prompt below** to get regional groupings
3. **Implement Strategy 3 (Hybrid)** in your code
4. **Compare model performance** before and after
5. **Document improvement** in your report

---

## Expected Results

### Before (Current Model)
- Features: 19
- Best Accuracy: 96.35% (Random Forest)

### After (With Location Features)
- Features: 27-34 (19 original + 8-15 location)
- Expected Accuracy: 96.8-97.5%
- Improvement: +0.5-1.2%

### Feature Importance Changes
- Traffic patterns: 85-88% (down from 91%)
- Location features: 5-8% (new)
- Air quality: 4-6% (similar)
- Temporal: 2-3% (similar)

This shows location matters, but traffic patterns still dominate!
