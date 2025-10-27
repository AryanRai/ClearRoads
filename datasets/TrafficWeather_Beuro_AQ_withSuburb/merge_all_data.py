"""
Final Data Merger: Traffic + Air Quality + BoM Weather
Combines all three data sources into one comprehensive dataset

Input files:
1. traffic_weather_merged_full.csv - Traffic + Air Quality (PM10, PM2.5, NO2, NO, CO)
2. bom_weather_combined.csv - BoM Weather (rainfall, solar, min/max temp)

Output:
- complete_traffic_environment_data.csv - All features combined
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from fuzzywuzzy import fuzz, process
import warnings
warnings.filterwarnings('ignore')

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

TRAFFIC_AQ_FILE = PROJECT_ROOT / "datasets/TrafficWeatherwithSuburb/traffic_weather_merged_full.csv"
BOM_WEATHER_FILE = PROJECT_ROOT / "datasets/Weather_Beuro_Meterology_PerDay/bom_weather_combined.csv"
OUTPUT_FILE = SCRIPT_DIR / "complete_traffic_environment_data.csv"

def fuzzy_match_suburbs(traffic_suburbs, bom_suburbs, threshold=80):
    """Match traffic suburbs to BoM weather suburbs using fuzzy matching"""
    matches = {}
    near_misses = []
    
    for traffic_suburb in traffic_suburbs:
        best_match = process.extractOne(traffic_suburb, bom_suburbs, scorer=fuzz.token_sort_ratio)
        
        if best_match:
            if best_match[1] >= threshold:
                matches[traffic_suburb] = best_match[0]
            elif best_match[1] >= 60:
                near_misses.append((traffic_suburb, best_match[0], best_match[1]))
    
    return matches, near_misses

def main():
    print("="*80)
    print("FINAL DATA MERGER: TRAFFIC + AIR QUALITY + BOM WEATHER")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load Traffic + Air Quality data
    print("\n" + "="*80)
    print("STEP 1: LOADING TRAFFIC + AIR QUALITY DATA")
    print("="*80)
    
    print(f"Loading: {TRAFFIC_AQ_FILE}")
    traffic_df = pd.read_csv(str(TRAFFIC_AQ_FILE), low_memory=False)
    
    # Convert date column
    traffic_df['date'] = pd.to_datetime(traffic_df['date'], errors='coerce')
    
    # Standardize suburb column
    if 'suburb_std' not in traffic_df.columns:
        traffic_df['suburb_std'] = traffic_df['suburb'].str.upper().str.strip()
    
    print(f"✓ Loaded {len(traffic_df):,} traffic records")
    print(f"  Date range: {traffic_df['date'].min()} to {traffic_df['date'].max()}")
    print(f"  Unique suburbs: {traffic_df['suburb_std'].nunique()}")
    
    # Get unique traffic suburbs
    traffic_suburbs = sorted(traffic_df['suburb_std'].dropna().unique())
    print(f"  Sample suburbs: {', '.join(traffic_suburbs[:5])}")
    
    # Step 2: Load BoM Weather data
    print("\n" + "="*80)
    print("STEP 2: LOADING BOM WEATHER DATA")
    print("="*80)
    
    print(f"Loading: {BOM_WEATHER_FILE}")
    bom_df = pd.read_csv(str(BOM_WEATHER_FILE))
    
    # Convert date column
    bom_df['date'] = pd.to_datetime(bom_df['date'], errors='coerce')
    
    # Standardize suburb column
    bom_df['suburb'] = bom_df['suburb'].str.upper().str.strip()
    
    print(f"✓ Loaded {len(bom_df):,} weather records")
    print(f"  Date range: {bom_df['date'].min()} to {bom_df['date'].max()}")
    print(f"  Unique suburbs: {bom_df['suburb'].nunique()}")
    
    # Get unique BoM suburbs
    bom_suburbs = sorted(bom_df['suburb'].dropna().unique())
    print(f"  Sample suburbs: {', '.join(bom_suburbs[:5])}")
    
    # Step 3: Match suburbs
    print("\n" + "="*80)
    print("STEP 3: MATCHING SUBURBS")
    print("="*80)
    
    suburb_matches, near_misses = fuzzy_match_suburbs(traffic_suburbs, bom_suburbs, threshold=80)
    
    print(f"\n✓ Matched {len(suburb_matches)} suburbs (threshold: 80)")
    
    if len(suburb_matches) > 0:
        print(f"\nTop 10 matches:")
        for i, (traffic_suburb, bom_suburb) in enumerate(sorted(suburb_matches.items())[:10], 1):
            print(f"  {i:2}. {traffic_suburb:30} → {bom_suburb}")
        
        if len(suburb_matches) > 10:
            print(f"  ... and {len(suburb_matches)-10} more")
    
    if len(near_misses) > 0 and len(suburb_matches) < 10:
        print(f"\n⚠️ Near misses (score 60-79):")
        for traffic, bom, score in near_misses[:5]:
            print(f"  {traffic:30} → {bom:30} (score: {score})")
    
    # Step 4: Merge BoM weather data
    print("\n" + "="*80)
    print("STEP 4: MERGING BOM WEATHER DATA")
    print("="*80)
    
    # Create reverse mapping (bom suburb -> traffic suburb)
    reverse_matches = {v: k for k, v in suburb_matches.items()}
    
    # Add traffic suburb column to BoM data
    bom_df['traffic_suburb'] = bom_df['suburb'].map(reverse_matches)
    
    # Filter to only matched suburbs
    bom_matched = bom_df[bom_df['traffic_suburb'].notna()].copy()
    print(f"BoM records with matched suburbs: {len(bom_matched):,}")
    
    # Merge with traffic data
    print("\nMerging on suburb + date...")
    merged_df = traffic_df.merge(
        bom_matched[['date', 'traffic_suburb', 'rainfall_mm', 'solar_exposure_mj', 'min_temp_c', 'max_temp_c']],
        left_on=['date', 'suburb_std'],
        right_on=['date', 'traffic_suburb'],
        how='left'
    )
    
    # Drop the extra traffic_suburb column
    if 'traffic_suburb' in merged_df.columns:
        merged_df = merged_df.drop(columns=['traffic_suburb'])
    
    print(f"✓ Merge complete: {len(merged_df):,} records")
    
    # Check BoM weather data coverage
    print(f"\nBoM Weather Data Coverage:")
    for col in ['rainfall_mm', 'solar_exposure_mj', 'min_temp_c', 'max_temp_c']:
        non_null = merged_df[col].notna().sum()
        print(f"  {col:20}: {non_null:,} ({non_null/len(merged_df)*100:.1f}%)")
    
    # Step 5: Save output
    print("\n" + "="*80)
    print("STEP 5: SAVING OUTPUT")
    print("="*80)
    
    output_path = OUTPUT_FILE
    merged_df.to_csv(str(OUTPUT_FILE), index=False)
    
    print(f"✓ Saved to: {OUTPUT_FILE}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Step 6: Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\nTotal records: {len(merged_df):,}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    
    print(f"\nComplete Feature List:")
    print(f"  Traffic: daily_total, hourly counts (hour_00 to hour_23)")
    print(f"  Location: suburb, distance_to_cbd_km, rms_region, coordinates")
    print(f"  Air Quality: PM10, PM2_5, NO2, NO, CO")
    print(f"  BoM Weather: rainfall_mm, solar_exposure_mj, min_temp_c, max_temp_c")
    print(f"  Temporal: year, month, day_of_week, public_holiday, school_holiday")
    
    print(f"\nData Completeness:")
    
    # Air Quality
    print(f"\n  Air Quality:")
    for col in ['PM10', 'PM2_5', 'NO2', 'NO', 'CO']:
        if col in merged_df.columns:
            non_null = merged_df[col].notna().sum()
            print(f"    {col:6}: {non_null:,} ({non_null/len(merged_df)*100:.1f}%)")
    
    # BoM Weather
    print(f"\n  BoM Weather:")
    for col in ['rainfall_mm', 'solar_exposure_mj', 'min_temp_c', 'max_temp_c']:
        if col in merged_df.columns:
            non_null = merged_df[col].notna().sum()
            print(f"    {col:20}: {non_null:,} ({non_null/len(merged_df)*100:.1f}%)")
    
    # Records with complete environmental data
    env_cols = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO', 'rainfall_mm', 'min_temp_c', 'max_temp_c']
    available_env_cols = [col for col in env_cols if col in merged_df.columns]
    
    complete_records = merged_df[available_env_cols].notna().all(axis=1).sum()
    print(f"\n  Records with ALL environmental features: {complete_records:,} ({complete_records/len(merged_df)*100:.1f}%)")
    
    # Records with at least some environmental data
    any_env_data = merged_df[available_env_cols].notna().any(axis=1).sum()
    print(f"  Records with ANY environmental features: {any_env_data:,} ({any_env_data/len(merged_df)*100:.1f}%)")
    
    print(f"\nRegional Distribution:")
    if 'rms_region' in merged_df.columns:
        region_counts = merged_df['rms_region'].value_counts()
        for region, count in region_counts.items():
            print(f"  {region:15}: {count:,} ({count/len(merged_df)*100:.1f}%)")
    
    print(f"\n✓ Processing complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print(f"\n🎉 SUCCESS! Your complete dataset is ready for ML modeling!")
    print(f"\nNext steps:")
    print(f"  1. Use this file in your traffic_analysis.py script")
    print(f"  2. Filter to records with environmental data")
    print(f"  3. Add location features (regions, distance to CBD)")
    print(f"  4. Train models with improved accuracy!")
    print("="*80)

if __name__ == "__main__":
    main()
