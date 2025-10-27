# ClearRoads Data Pipeline Guide

**Project**: Traffic Congestion Prediction using Environmental Intelligence  
**Course**: ENGG2112 - The University of Sydney  
**Last Updated**: October 28, 2025

---

## 📊 Data Pipeline Overview

This document describes the complete data processing pipeline from raw data to final ML-ready dataset.

```
Raw Data Sources
    ↓
Step 1: Extract Traffic + Location Features
    ↓
Step 2: Merge Traffic + Air Quality
    ↓
Step 3: Combine BoM Weather Data
    ↓
Step 4: Final Merge
    ↓
Complete Dataset for ML
```

---

## 🗂️ Directory Structure

```
ProjectProposal/
├── datasets/
│   ├── Traffic_TimesOfDay/
│   │   ├── road_traffic_counts_hourly_permanent/
│   │   │   ├── road_traffic_counts_hourly_permanent0.csv (50+ MB)
│   │   │   ├── road_traffic_counts_hourly_permanent1.csv (50+ MB)
│   │   │   ├── road_traffic_counts_hourly_permanent2.csv (50+ MB)
│   │   │   ├── road_traffic_counts_hourly_permanent3.csv (50+ MB)
│   │   │   └── road_traffic_counts_hourly_permanent_all_with_location.csv ✅ OUTPUT
│   │   └── road_traffic_counts_station_reference.csv
│   │
│   ├── Weather_AQ/
│   │   ├── XLS-file_Daily_Averages-PM10_Time_Range_01012008_0000_to_02012025_0000.csv
│   │   ├── XLS-file_Daily_Averages-PM2-5_Time_Range_01012008_0000_to_02012025_0000.csv
│   │   ├── XLS-file_Daily_Averages-NO2_Time_Range_01012008_0000_to_02012025_0000.csv
│   │   ├── XLS-file_Daily_Averages-NO_Time_Range_01012008_0000_to_02012025_0000.csv
│   │   └── XLS-file_Daily_Averages-CO_Time_Range_01012008_0000_to_02012025_0000.csv
│   │
│   ├── Weather_Beuro_Meterology_PerDay/
│   │   ├── Substations_1/Substations_1/[34 suburb folders]/
│   │   ├── Substations_2/Substations_2/[34 suburb folders]/
│   │   ├── substations_34/my substations/[34 suburb folders]/
│   │   └── bom_weather_combined.csv ✅ OUTPUT
│   │
│   ├── TrafficWeatherwithSuburb/
│   │   └── traffic_weather_merged_full.csv ✅ OUTPUT
│   │
│   └── TrafficWeather_Beuro_AQ_withSuburb/
│       └── complete_traffic_environment_data.csv ✅ FINAL OUTPUT
│
├── datasets/TrafficwithSuburb/
│   └── cleanData.py 📜 SCRIPT 1
│
├── merge_traffic_weather.py 📜 SCRIPT 2
├── combine_bom_weather.py 📜 SCRIPT 3
└── datasets/TrafficWeather_Beuro_AQ_withSuburb/
    └── merge_all_data.py 📜 SCRIPT 4
```

---

## 📜 Processing Scripts

### Script 1: `datasets/TrafficwithSuburb/cleanData.py`

**Purpose**: Extract traffic data and add location features

**Input**:
- `road_traffic_counts_hourly_permanent0-3.csv` (4 files, ~200 MB total)
- `road_traffic_counts_station_reference.csv`

**Output**:
- `road_traffic_counts_hourly_permanent_all_with_location.csv` (~200 MB)

**Features Added**:
- Suburb names (standardized)
- GPS coordinates (latitude, longitude)
- Distance to Sydney CBD (km)
- RMS region (Sydney, Hunter, Western, etc.)
- Road classification
- Lane count

**Run Command**:
```bash
python datasets/TrafficwithSuburb/cleanData.py
```

**Runtime**: ~3-5 minutes

**Output Stats**:
- Records: 3,925,503
- Unique suburbs: 354
- Date range: 2006-2025

---

### Script 2: `merge_traffic_weather.py`

**Purpose**: Merge traffic data with air quality data

**Input**:
- `road_traffic_counts_hourly_permanent_all_with_location.csv`
- Air quality CSVs (PM10, PM2.5, NO2, NO, CO)

**Output**:
- `traffic_weather_merged_full.csv` (~1.3 GB)

**Features Added**:
- PM10 (particulate matter 10μm)
- PM2.5 (fine particulate matter 2.5μm)
- NO2 (nitrogen dioxide)
- NO (nitrogen oxide)
- CO (carbon monoxide)

**Run Command**:
```bash
python merge_traffic_weather.py
```

**Runtime**: ~3-6 minutes

**Output Stats**:
- Records: 3,925,503
- Air quality coverage: ~5% (200k records)
- Matched suburbs: 37

---

### Script 3: `combine_bom_weather.py`

**Purpose**: Combine Bureau of Meteorology weather data from multiple suburb folders

**Input**:
- `Substations_1/Substations_1/[suburb folders]/1-4.csv`
- `Substations_2/Substations_2/[suburb folders]/1-4.csv`
- `substations_34/my substations/[suburb folders]/1-4.csv`

**Output**:
- `bom_weather_combined.csv` (~68 MB)

**Features Added**:
- Rainfall (mm)
- Solar exposure (MJ/m²)
- Minimum temperature (°C)
- Maximum temperature (°C)

**Run Command**:
```bash
python combine_bom_weather.py
```

**Runtime**: ~30 seconds

**Output Stats**:
- Records: ~1M (after deduplication)
- Unique suburbs: 34
- Date range: 1862-2025

---

### Script 4: `datasets/TrafficWeather_Beuro_AQ_withSuburb/merge_all_data.py`

**Purpose**: Final merge of all data sources

**Input**:
- `traffic_weather_merged_full.csv` (Traffic + Air Quality)
- `bom_weather_combined.csv` (BoM Weather)

**Output**:
- `complete_traffic_environment_data.csv` (~1.4 GB) ✅ **FINAL DATASET**

**Features Added**:
- All BoM weather features merged by suburb + date

**Run Command**:
```bash
python datasets/TrafficWeather_Beuro_AQ_withSuburb/merge_all_data.py
```

**Runtime**: ~5-10 minutes

**Output Stats**:
- Records: 3,925,503
- Complete environmental data: ~5-10%
- All features combined

---

## 📊 Final Datasets

### 1. Intermediate Dataset: `traffic_weather_merged_full.csv`

**Size**: ~1.3 GB  
**Records**: 3,925,503  
**Date Range**: 2006-2025

**Features** (48 columns):
- **Traffic**: daily_total, hour_00 to hour_23
- **Location**: suburb, suburb_std, distance_to_cbd_km, rms_region, wgs84_latitude, wgs84_longitude, lga, road_classification_type
- **Air Quality**: PM10, PM2.5, NO2, NO, CO
- **Temporal**: date, year, month, day_of_week, public_holiday, school_holiday

**Coverage**:
- PM10: 200,420 records (5.1%)
- PM2.5: 147,746 records (3.8%)
- NO2: 73,938 records (1.9%)
- NO: 167,205 records (4.3%)
- CO: 73,938 records (1.9%)

---

### 2. Final Dataset: `complete_traffic_environment_data.csv` ✅

**Size**: ~1.4 GB  
**Records**: 3,925,503  
**Date Range**: 2006-2025

**Features** (52 columns):
- **Traffic**: daily_total, hour_00 to hour_23
- **Location**: suburb, suburb_std, distance_to_cbd_km, rms_region, wgs84_latitude, wgs84_longitude, lga, road_classification_type, lane_count, road_functional_hierarchy
- **Air Quality**: PM10, PM2.5, NO2, NO, CO
- **BoM Weather**: rainfall_mm, solar_exposure_mj, min_temp_c, max_temp_c
- **Temporal**: date, year, month, day_of_week, public_holiday, school_holiday

**Coverage**:
- Air Quality: ~5% (200k records)
- BoM Weather: ~5-10% (varies by feature)
- Complete environmental data: ~200k-400k records

**Regional Distribution**:
- Sydney: 50.6%
- Southern: 11.6%
- Hunter: 10.3%
- Western: 7.8%
- Northern: 5.0%
- South West: 4.0%

---

## 🚀 Quick Start Guide

### Full Pipeline Execution

Run all scripts in order:

```bash
# Step 1: Extract traffic + location features (~3-5 min)
python datasets/TrafficwithSuburb/cleanData.py

# Step 2: Merge traffic + air quality (~3-6 min)
python merge_traffic_weather.py

# Step 3: Combine BoM weather data (~30 sec)
python combine_bom_weather.py

# Step 4: Final merge (~5-10 min)
python datasets/TrafficWeather_Beuro_AQ_withSuburb/merge_all_data.py
```

**Total Runtime**: ~12-22 minutes

---

## 📈 Using the Final Dataset

### For ML Modeling

Use `complete_traffic_environment_data.csv` with your `traffic_analysis.py` script:

```python
# Update file path in traffic_analysis.py
filepath = "datasets/TrafficWeather_Beuro_AQ_withSuburb/complete_traffic_environment_data.csv"
```

### Filtering Recommendations

For best ML results, filter to records with environmental data:

```python
# Filter to records with air quality data
df_with_aq = df.dropna(subset=['PM10', 'PM2_5'], how='all')

# Filter to records with complete environmental data
env_cols = ['PM10', 'PM2_5', 'rainfall_mm', 'min_temp_c', 'max_temp_c']
df_complete = df.dropna(subset=env_cols)
```

---

## 🔍 Data Quality Notes

### Spatial Coverage

- **Traffic stations**: 354 suburbs across NSW
- **Air quality stations**: 67 suburbs (mostly urban)
- **BoM weather stations**: 34 suburbs

### Temporal Coverage

- **Traffic data**: 2006-2025 (19 years)
- **Air quality data**: 2008-2025 (17 years)
- **BoM weather data**: 1862-2025 (163 years, varies by suburb)

### Missing Data

- **Air quality**: ~95% missing (spatial mismatch - only 67 monitoring stations)
- **BoM weather**: ~90-95% missing (only 34 weather stations)
- **Traffic data**: ~10-15% missing (hourly columns)

**This is normal and expected!** The ~200k-400k records with complete environmental data is still plenty for ML modeling.

---

## 📝 Key Insights

### Why Low Environmental Coverage?

1. **Sparse monitoring network**: Only 67 air quality + 34 weather stations vs 354 traffic stations
2. **Spatial mismatch**: Most traffic stations don't have nearby environmental monitors
3. **Temporal mismatch**: Different data collection periods

### Why This is Still Good

1. **200k+ records** with environmental data (vs 71k in original analysis)
2. **Full regional representation** (all RMS regions included)
3. **Rich feature set** (52 features total)
4. **High-quality data** (validated and cleaned)

---

## 🎯 Next Steps

1. ✅ Run all 4 scripts to generate final dataset
2. ✅ Update `traffic_analysis.py` to use new dataset
3. ✅ Add location features (regions, distance to CBD)
4. ✅ Train models with improved feature set
5. ✅ Compare results with original analysis

---

## 📚 Related Documentation

- `proposal.md` - Project proposal
- `Plan.md` - Analysis plan
- `RESULTS_SUMMARY.md` - Analysis results
- `MODEL_PARAMETERS_EXPLAINED.md` - Model parameter details
- `SUBURB_FEATURE_STRATEGIES.md` - Location feature strategies

---

**Last Updated**: October 28, 2025  
**Pipeline Version**: 1.0  
**Status**: ✅ Complete and Ready for ML
