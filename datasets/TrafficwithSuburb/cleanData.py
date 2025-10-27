import pandas as pd
from pathlib import Path
import re
from math import radians, sin, cos, sqrt, atan2

# --- your constants ---
# Get the script's directory and construct paths relative to ProjectProposal root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up to ProjectProposal directory

FILE1 = PROJECT_ROOT / "datasets/Traffic_TimesOfDay/road_traffic_counts_hourly_permanent/road_traffic_counts_hourly_permanent0.csv"
FILE2 = PROJECT_ROOT / "datasets/Traffic_TimesOfDay/road_traffic_counts_station_reference.csv"
KEY   = "station_key"
SUBURB_COL = "suburb"

# Sydney CBD coordinates for distance calculation
CBD_LAT = -33.8708
CBD_LON = 151.2073
# ----------------------

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df

def as_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points using Haversine formula"""
    R = 6371  # Earth radius in kilometers
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

# 1) Load reference file (station_key + suburb + location features)
print("Loading station reference data...")
print(f"Reference file: {FILE2}")
ref = pd.read_csv(str(FILE2))
ref = normalize_cols(ref)

# Check for required columns
required_cols = {KEY, SUBURB_COL}
missing = required_cols - set(ref.columns)
if missing:
    raise KeyError(f"{FILE2} missing columns: {missing}")

# Location features to extract
location_features = [
    KEY,
    SUBURB_COL,
    'wgs84_latitude',
    'wgs84_longitude', 
    'rms_region',
    'lga',
    'road_classification_type',
    'road_functional_hierarchy',
    'lane_count'
]

# Keep only columns that exist
available_features = [col for col in location_features if col in ref.columns]
print(f"Extracting {len(available_features)} location features: {', '.join(available_features)}")

ref[KEY] = as_key(ref[KEY])
ref_dedup = ref.drop_duplicates(subset=[KEY], keep="first")[available_features]

# Calculate distance to CBD if coordinates are available
if 'wgs84_latitude' in ref_dedup.columns and 'wgs84_longitude' in ref_dedup.columns:
    print("Calculating distance to Sydney CBD...")
    ref_dedup['distance_to_cbd_km'] = ref_dedup.apply(
        lambda row: haversine_distance(
            CBD_LAT, CBD_LON,
            row['wgs84_latitude'], row['wgs84_longitude']
        ) if pd.notna(row['wgs84_latitude']) and pd.notna(row['wgs84_longitude']) else None,
        axis=1
    )
    print(f"✓ Distance to CBD calculated (range: {ref_dedup['distance_to_cbd_km'].min():.1f} - {ref_dedup['distance_to_cbd_km'].max():.1f} km)")

# Standardize suburb names
ref_dedup['suburb_std'] = ref_dedup[SUBURB_COL].str.upper().str.strip()

# 2) Find all hourly CSVs 0–3
file1_str = str(FILE1)
m = re.search(r"(.*?)(\d)(\.csv)$", file1_str)
if m:
    prefix, _, suffix = m.groups()
    file1_paths = [Path(f"{prefix}{i}{suffix}") for i in range(0, 4)]
    file1_paths = [p for p in file1_paths if p.is_file()]
else:
    file1_paths = [FILE1]

print("Will process:")
for p in file1_paths:
    print(" -", p)

# 3) Merge and stack all four
merged_parts = []
for p in file1_paths:
    print(f"Processing: {p.name}")
    df1 = pd.read_csv(str(p))
    df1 = normalize_cols(df1)

    if KEY not in df1.columns:
        raise KeyError(f"{p} is missing key column '{KEY}'\nColumns: {list(df1.columns)}")

    df1[KEY] = as_key(df1[KEY])
    merged = df1.merge(ref_dedup, on=KEY, how="left")
    merged_parts.append(merged)
    print(f"  ✓ Merged {len(merged):,} rows")

# 4) Combine and export one CSV
combined = pd.concat(merged_parts, ignore_index=True)
out_path = file1_paths[0].with_name("road_traffic_counts_hourly_permanent_all_with_location.csv")
combined.to_csv(str(out_path), index=False)

print("\n" + "="*80)
print("✓ PROCESSING COMPLETE")
print("="*80)
print(f"Output file: {Path(out_path).name}")
print(f"Total rows: {len(combined):,}")
print(f"Rows with suburbs: {combined[SUBURB_COL].notna().sum():,} ({combined[SUBURB_COL].notna().sum()/len(combined)*100:.1f}%)")

# Summary statistics
print("\n" + "="*80)
print("LOCATION FEATURES SUMMARY")
print("="*80)

if 'suburb_std' in combined.columns:
    print(f"Unique suburbs: {combined['suburb_std'].nunique()}")
    print(f"Top 5 suburbs by traffic count:")
    top_suburbs = combined['suburb_std'].value_counts().head(5)
    for suburb, count in top_suburbs.items():
        print(f"  - {suburb}: {count:,} observations")

if 'rms_region' in combined.columns:
    print(f"\nRMS Regions: {combined['rms_region'].nunique()}")
    print("Distribution:")
    for region, count in combined['rms_region'].value_counts().items():
        print(f"  - {region}: {count:,} ({count/len(combined)*100:.1f}%)")

if 'distance_to_cbd_km' in combined.columns:
    print(f"\nDistance to CBD:")
    print(f"  Min: {combined['distance_to_cbd_km'].min():.1f} km")
    print(f"  Max: {combined['distance_to_cbd_km'].max():.1f} km")
    print(f"  Mean: {combined['distance_to_cbd_km'].mean():.1f} km")
    print(f"  Median: {combined['distance_to_cbd_km'].median():.1f} km")

if 'road_classification_type' in combined.columns:
    print(f"\nRoad Classifications:")
    for road_type, count in combined['road_classification_type'].value_counts().head(5).items():
        print(f"  - {road_type}: {count:,} ({count/len(combined)*100:.1f}%)")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Use this file to merge with weather/air quality data")
print("2. Location features are ready for ML model:")
print("   - suburb_std (standardized suburb names)")
print("   - distance_to_cbd_km (continuous feature)")
print("   - rms_region (categorical - can be used for regional grouping)")
print("   - road_classification_type (road type)")
print("   - wgs84_latitude, wgs84_longitude (coordinates)")
print("="*80)