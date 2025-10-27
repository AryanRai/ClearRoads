# Quick Guide: Adding Location Features to Your Model

## 🎯 Goal
Add suburb/location information to improve model accuracy from 96.35% to ~97%+

## 📋 Step-by-Step Process

### Step 1: Get Your Suburb List (5 minutes)

```bash
python get_suburbs.py
```

This will:
- Print all unique suburbs in your dataset
- Show observation counts per suburb
- Save output to `suburb_list.txt`

### Step 2: Get AI to Group Suburbs (10 minutes)

1. Open `AI_PROMPT_FOR_SUBURB_GROUPING.md`
2. Copy the prompt template
3. Paste your suburb list from Step 1
4. Send to ChatGPT/Claude
5. Save the output dictionaries

### Step 3: Create Mapping File (5 minutes)

Create `suburb_mappings.py`:

```python
# Regional grouping (from AI)
region_mapping = {
    'PARRAMATTA': 'Western_Sydney',
    'NEWCASTLE': 'Hunter_Region',
    # ... paste AI output
}

# Urban classification (from AI)
urban_classification = {
    'PARRAMATTA': 'Urban',
    'PENRITH': 'Suburban',
    # ... paste AI output
}

# Sydney CBD coordinates
CBD_LAT = -33.8688
CBD_LON = 151.2093
```

### Step 4: Modify Your Analysis Script (15 minutes)

Add to `traffic_analysis.py`:

```python
# At the top, import mappings
from suburb_mappings import region_mapping, urban_classification, CBD_LAT, CBD_LON
from math import radians, sin, cos, sqrt, atan2

# Add haversine function
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# In prepare_features() function, add:
def prepare_features(df):
    # ... existing code ...
    
    # NEW: Add location features
    # 1. Regional grouping
    df['region'] = df['suburb_std'].map(region_mapping)
    region_dummies = pd.get_dummies(df['region'], prefix='region')
    df = pd.concat([df, region_dummies], axis=1)
    feature_cols.extend(region_dummies.columns.tolist())
    
    # 2. Urban classification
    df['urban_type'] = df['suburb_std'].map(urban_classification)
    urban_dummies = pd.get_dummies(df['urban_type'], prefix='urban')
    df = pd.concat([df, urban_dummies], axis=1)
    feature_cols.extend(urban_dummies.columns.tolist())
    
    # 3. Distance to CBD (if you have coordinates)
    # df['distance_to_cbd'] = df['suburb_std'].map(
    #     lambda s: haversine_distance(CBD_LAT, CBD_LON, coords[s][0], coords[s][1])
    # )
    # feature_cols.append('distance_to_cbd')
    
    # ... rest of existing code ...
```

### Step 5: Run and Compare (5 minutes)

```bash
python traffic_analysis.py
```

Compare results:
- **Before**: 19 features, 96.35% accuracy
- **After**: 27-34 features, 96.8-97.5% accuracy (expected)

### Step 6: Document Results (10 minutes)

Note in your report:
- How many regions you created
- Rationale for grouping
- Accuracy improvement
- Feature importance changes

---

## 🚀 Quick Start (Copy-Paste)

### 1. Run this first:
```bash
python get_suburbs.py
```

### 2. Use this prompt with ChatGPT:
```
I have a traffic congestion prediction project for NSW, Australia. I need to group suburbs into 5-10 geographic regions.

Here are my unique suburbs:
[PASTE OUTPUT FROM STEP 1]

Please provide:
1. Regional grouping as Python dictionary
2. Urban/Suburban/Regional classification
3. Rationale for groupings

Format:
region_mapping = {'SUBURB': 'Region_Name', ...}
urban_classification = {'SUBURB': 'Urban', ...}
```

### 3. Save AI output to `suburb_mappings.py`

### 4. Add to `traffic_analysis.py` (see Step 4 above)

### 5. Run and enjoy improved accuracy! 🎉

---

## 📊 Expected Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Features | 19 | 27-34 | +8-15 |
| Accuracy | 96.35% | 96.8-97.5% | +0.5-1.2% |
| Location Importance | 0% | 5-8% | New |
| Traffic Importance | 91% | 85-88% | -3-6% |

---

## 📚 Reference Documents

- **SUBURB_FEATURE_STRATEGIES.md** - Detailed explanation of all strategies
- **AI_PROMPT_FOR_SUBURB_GROUPING.md** - Ready-to-use AI prompts
- **get_suburbs.py** - Script to extract suburb list
- **suburb_mappings.py** - Your mapping file (create after AI response)

---

## ⚠️ Common Issues

**Issue**: Some suburbs not in mapping  
**Fix**: Add them manually or use a default region

**Issue**: Too many regions (15+)  
**Fix**: Ask AI to consolidate into 5-10 regions

**Issue**: Accuracy doesn't improve  
**Fix**: Check if regions are too broad or too specific

**Issue**: Model takes longer to train  
**Fix**: Normal with more features, still fast (<5 min)

---

## 💡 Pro Tips

1. **Start simple**: Just regional grouping first
2. **Validate groupings**: Use your local knowledge
3. **Check feature importance**: See if location features matter
4. **Compare models**: Random Forest vs XGBoost with location features
5. **Document everything**: Show before/after in your report

---

## ✅ Checklist

- [ ] Run `get_suburbs.py`
- [ ] Get AI grouping (ChatGPT/Claude)
- [ ] Create `suburb_mappings.py`
- [ ] Modify `traffic_analysis.py`
- [ ] Run and verify improvement
- [ ] Document in report

**Estimated Time**: 50 minutes total

Good luck! 🚀
