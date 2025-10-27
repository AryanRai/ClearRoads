"""
Quick data summary script to check the complete dataset
"""

import pandas as pd
import numpy as np

print("="*80)
print("DATA SUMMARY: Complete Traffic Environment Dataset")
print("="*80)

# Load sample of data
filepath = "datasets/TrafficWeather_Beuro_AQ_withSuburb/complete_traffic_environment_data.csv"
print(f"\nLoading sample (100,000 rows)...")
df = pd.read_csv(filepath, nrows=100000, low_memory=False)

print(f"\n✓ Loaded {len(df):,} rows × {df.shape[1]} columns")

# Show columns
print(f"\n📋 All Columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# Check key features
print(f"\n🗺️ Location Features:")
if 'suburb_std' in df.columns:
    unique_suburbs = df['suburb_std'].nunique()
    print(f"  Unique suburbs: {unique_suburbs}")
    print(f"  Top 10 suburbs:")
    for suburb, count in df['suburb_std'].value_counts().head(10).items():
        print(f"    {suburb:30s}: {count:>6,}")

if 'rms_region' in df.columns:
    print(f"\n  RMS Regions:")
    for region, count in df['rms_region'].value_counts().items():
        print(f"    {region:15s}: {count:>6,} ({count/len(df)*100:>5.1f}%)")

if 'distance_to_cbd_km' in df.columns:
    print(f"\n  Distance to CBD:")
    print(f"    Min:    {df['distance_to_cbd_km'].min():.1f} km")
    print(f"    Max:    {df['distance_to_cbd_km'].max():.1f} km")
    print(f"    Mean:   {df['distance_to_cbd_km'].mean():.1f} km")
    print(f"    Median: {df['distance_to_cbd_km'].median():.1f} km")

# Environmental features
print(f"\n🌫️ Environmental Features Coverage:")
env_features = {
    'Air Quality': ['PM10', 'PM2_5', 'NO2', 'NO', 'CO'],
    'Weather': ['rainfall_mm', 'solar_exposure_mj', 'min_temp_c', 'max_temp_c']
}

for category, features in env_features.items():
    print(f"\n  {category}:")
    for feat in features:
        if feat in df.columns:
            count = df[feat].notna().sum()
            pct = count / len(df) * 100
            print(f"    {feat:20s}: {count:>6,} ({pct:>5.1f}%)")

# Records with environmental data
env_cols = []
for features in env_features.values():
    env_cols.extend([f for f in features if f in df.columns])

any_env = df[env_cols].notna().any(axis=1).sum()
print(f"\n  Records with ANY environmental data: {any_env:,} ({any_env/len(df)*100:.1f}%)")

# Traffic data
print(f"\n🚗 Traffic Data:")
if 'daily_total' in df.columns:
    print(f"  Daily total statistics:")
    print(f"    Min:    {df['daily_total'].min():>10,.0f}")
    print(f"    Max:    {df['daily_total'].max():>10,.0f}")
    print(f"    Mean:   {df['daily_total'].mean():>10,.0f}")
    print(f"    Median: {df['daily_total'].median():>10,.0f}")
    print(f"    Missing: {df['daily_total'].isna().sum():>10,}")

# Temporal coverage
print(f"\n📅 Temporal Coverage:")
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Years: {sorted(df['year'].unique()) if 'year' in df.columns else 'N/A'}")

print(f"\n✅ Summary complete!")
print("="*80)
