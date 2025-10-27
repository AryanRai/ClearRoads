# Exact Code Modifications to Add Location Features

## Files You'll Create/Modify

1. ✅ `get_suburbs.py` - Already created
2. 🆕 `suburb_mappings.py` - You'll create after AI response
3. ✏️ `traffic_analysis.py` - Modify existing file

---

## Step 1: Run get_suburbs.py

```bash
python get_suburbs.py
```

Copy the output for the AI prompt.

---

## Step 2: Create suburb_mappings.py

After getting AI response, create this file:

```python
"""
Suburb mappings for location features
Generated using AI grouping based on geographic and demographic similarity
"""

# Regional grouping (5-10 regions)
# PASTE AI OUTPUT HERE
region_mapping = {
    'PARRAMATTA': 'Western_Sydney',
    'NEWCASTLE': 'Hunter_Region',
    'WOLLONGONG': 'Illawarra',
    # ... add all your suburbs
}

# Urban classification (3 categories)
# PASTE AI OUTPUT HERE
urban_classification = {
    'PARRAMATTA': 'Urban',
    'PENRITH': 'Suburban',
    'ALBURY': 'Regional',
    # ... add all your suburbs
}

# Sydney CBD coordinates (for distance calculation)
CBD_LAT = -33.8688
CBD_LON = 151.2093

# Optional: Suburb coordinates (if you want distance feature)
# PASTE AI OUTPUT HERE IF YOU GOT IT
suburb_coords = {
    'PARRAMATTA': (-33.8150, 151.0000),
    'NEWCASTLE': (-32.9283, 151.7817),
    # ... add all your suburbs
}
```

---

## Step 3: Modify traffic_analysis.py

### 3A. Add Imports at the Top

```python
# Add these imports at the top of the file (after existing imports)
from suburb_mappings import region_mapping, urban_classification, CBD_LAT, CBD_LON
from math import radians, sin, cos, sqrt, atan2
```

### 3B. Add Haversine Function (Optional - only if using distance)

Add this function after the imports, before `load_and_explore_data()`:

```python
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance in km between two points using Haversine formula
    
    Parameters:
    -----------
    lat1, lon1 : float
        Latitude and longitude of first point in decimal degrees
    lat2, lon2 : float
        Latitude and longitude of second point in decimal degrees
    
    Returns:
    --------
    float : Distance in kilometers
    """
    R = 6371  # Earth radius in kilometers
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c
```

### 3C. Modify prepare_features() Function

Find the `prepare_features()` function and modify it:

```python
def prepare_features(df):
    """Prepare features for modeling"""
    # Select features
    air_quality_features = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO', 'AQI_composite']
    temporal_features = ['month', 'day_of_week', 'public_holiday', 'school_holiday', 
                        'is_weekend', 'year']
    traffic_features = ['morning_rush', 'evening_rush', 'peak_hour_traffic']
    
    # Combine all features
    feature_cols = []
    for col in air_quality_features + temporal_features + traffic_features:
        if col in df.columns:
            feature_cols.append(col)
    
    # Encode season
    if 'season' in df.columns:
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        feature_cols.extend(season_dummies.columns.tolist())
    
    # ========== NEW: ADD LOCATION FEATURES ==========
    
    # 1. Regional grouping
    print("\n🗺️ Adding location features...")
    df['region'] = df['suburb_std'].map(region_mapping)
    
    # Check for unmapped suburbs
    unmapped = df[df['region'].isna()]['suburb_std'].unique()
    if len(unmapped) > 0:
        print(f"⚠️ Warning: {len(unmapped)} suburbs not in region_mapping:")
        for suburb in unmapped:
            print(f"  - {suburb}")
        # Fill unmapped with 'Unknown' or most common region
        df['region'] = df['region'].fillna('Unknown')
    
    region_dummies = pd.get_dummies(df['region'], prefix='region')
    df = pd.concat([df, region_dummies], axis=1)
    feature_cols.extend(region_dummies.columns.tolist())
    print(f"✓ Added {len(region_dummies.columns)} regional features")
    
    # 2. Urban classification
    df['urban_type'] = df['suburb_std'].map(urban_classification)
    
    # Check for unmapped suburbs
    unmapped = df[df['urban_type'].isna()]['suburb_std'].unique()
    if len(unmapped) > 0:
        print(f"⚠️ Warning: {len(unmapped)} suburbs not in urban_classification")
        # Fill unmapped with 'Unknown' or most common type
        df['urban_type'] = df['urban_type'].fillna('Suburban')
    
    urban_dummies = pd.get_dummies(df['urban_type'], prefix='urban')
    df = pd.concat([df, urban_dummies], axis=1)
    feature_cols.extend(urban_dummies.columns.tolist())
    print(f"✓ Added {len(urban_dummies.columns)} urban type features")
    
    # 3. Distance to CBD (OPTIONAL - uncomment if you have suburb_coords)
    # try:
    #     from suburb_mappings import suburb_coords
    #     df['distance_to_cbd'] = df['suburb_std'].map(
    #         lambda s: haversine_distance(
    #             CBD_LAT, CBD_LON, 
    #             suburb_coords.get(s, (CBD_LAT, CBD_LON))[0],
    #             suburb_coords.get(s, (CBD_LAT, CBD_LON))[1]
    #         ) if s in suburb_coords else 0
    #     )
    #     feature_cols.append('distance_to_cbd')
    #     print(f"✓ Added distance_to_cbd feature")
    # except ImportError:
    #     print("ℹ️ suburb_coords not available, skipping distance feature")
    
    print(f"✓ Total location features added: {len([f for f in feature_cols if 'region_' in f or 'urban_' in f])}")
    
    # ========== END NEW CODE ==========
    
    X = df[feature_cols].copy()
    y = df['Congestion_Class'].copy()
    
    # Encode target labels for XGBoost compatibility
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y, y_encoded, feature_cols, label_encoder
```

---

## Complete Modified prepare_features() Function

Here's the complete function with location features:

```python
def prepare_features(df):
    """Prepare features for modeling including location features"""
    # Select features
    air_quality_features = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO', 'AQI_composite']
    temporal_features = ['month', 'day_of_week', 'public_holiday', 'school_holiday', 
                        'is_weekend', 'year']
    traffic_features = ['morning_rush', 'evening_rush', 'peak_hour_traffic']
    
    # Combine all features
    feature_cols = []
    for col in air_quality_features + temporal_features + traffic_features:
        if col in df.columns:
            feature_cols.append(col)
    
    # Encode season
    if 'season' in df.columns:
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        feature_cols.extend(season_dummies.columns.tolist())
    
    # Add location features
    print("\n🗺️ Adding location features...")
    
    # Regional grouping
    df['region'] = df['suburb_std'].map(region_mapping)
    unmapped = df[df['region'].isna()]['suburb_std'].unique()
    if len(unmapped) > 0:
        print(f"⚠️ Warning: {len(unmapped)} suburbs not in region_mapping")
        df['region'] = df['region'].fillna('Unknown')
    
    region_dummies = pd.get_dummies(df['region'], prefix='region')
    df = pd.concat([df, region_dummies], axis=1)
    feature_cols.extend(region_dummies.columns.tolist())
    print(f"✓ Added {len(region_dummies.columns)} regional features")
    
    # Urban classification
    df['urban_type'] = df['suburb_std'].map(urban_classification)
    unmapped = df[df['urban_type'].isna()]['suburb_std'].unique()
    if len(unmapped) > 0:
        print(f"⚠️ Warning: {len(unmapped)} suburbs not in urban_classification")
        df['urban_type'] = df['urban_type'].fillna('Suburban')
    
    urban_dummies = pd.get_dummies(df['urban_type'], prefix='urban')
    df = pd.concat([df, urban_dummies], axis=1)
    feature_cols.extend(urban_dummies.columns.tolist())
    print(f"✓ Added {len(urban_dummies.columns)} urban type features")
    
    print(f"✓ Total location features: {len([f for f in feature_cols if 'region_' in f or 'urban_' in f])}")
    
    X = df[feature_cols].copy()
    y = df['Congestion_Class'].copy()
    
    # Encode target labels for XGBoost compatibility
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y, y_encoded, feature_cols, label_encoder
```

---

## Testing Your Changes

### Before Running

1. ✅ Created `suburb_mappings.py` with AI output
2. ✅ Modified `traffic_analysis.py` with new code
3. ✅ Saved all files

### Run the Analysis

```bash
python traffic_analysis.py
```

### Expected Output

You should see:
```
🗺️ Adding location features...
✓ Added 8 regional features
✓ Added 3 urban type features
✓ Total location features: 11

📋 Selected Features (30):
  PM10, PM2_5, NO2, NO, CO, AQI_composite, month, day_of_week, 
  public_holiday, school_holiday, is_weekend, year, morning_rush, 
  evening_rush, peak_hour_traffic, season_Autumn, season_Spring, 
  season_Summer, season_Winter, region_Eastern_Sydney, 
  region_Hunter_Region, region_Illawarra, region_Inner_Sydney, 
  region_Northern_Sydney, region_Regional_NSW, region_Southern_Sydney, 
  region_Western_Sydney, urban_Regional, urban_Suburban, urban_Urban
```

### Compare Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Features | 19 | 30 | +11 |
| Random Forest Accuracy | 96.35% | 96.8-97.5% | +0.5-1.2% |

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'suburb_mappings'"

**Fix**: Create `suburb_mappings.py` in the same directory as `traffic_analysis.py`

### Error: "KeyError: 'SUBURB_NAME'"

**Fix**: Some suburb in your data isn't in the mapping. The code handles this with:
```python
df['region'] = df['region'].fillna('Unknown')
```

### Warning: "X suburbs not in region_mapping"

**Fix**: Add missing suburbs to your `suburb_mappings.py` or let them default to 'Unknown'

### No Accuracy Improvement

**Possible reasons**:
1. Regions too broad (all suburbs in one region)
2. Regions too specific (each suburb its own region)
3. Location doesn't matter much for your data
4. Need to tune model hyperparameters with new features

---

## Verification Checklist

- [ ] `suburb_mappings.py` exists and has all dictionaries
- [ ] Imports added to `traffic_analysis.py`
- [ ] `prepare_features()` modified correctly
- [ ] Script runs without errors
- [ ] Feature count increased (19 → 30+)
- [ ] Accuracy improved or stayed similar
- [ ] Feature importance shows location features

---

## Next Steps After Implementation

1. **Check feature importance**: See if location features matter
2. **Compare models**: Which model benefits most from location?
3. **Analyze by region**: Do some regions have better predictions?
4. **Document in report**: Show before/after comparison
5. **Consider distance feature**: If you have coordinates

---

## Quick Copy-Paste Summary

### 1. Create suburb_mappings.py
```python
region_mapping = {
    # PASTE AI OUTPUT
}

urban_classification = {
    # PASTE AI OUTPUT
}

CBD_LAT = -33.8688
CBD_LON = 151.2093
```

### 2. Add to traffic_analysis.py imports
```python
from suburb_mappings import region_mapping, urban_classification, CBD_LAT, CBD_LON
```

### 3. Modify prepare_features()
See complete function above

### 4. Run
```bash
python traffic_analysis.py
```

Done! 🎉
