"""
Merge Traffic Data with Weather/Air Quality Data
Improved version that:
1. Uses ALL available data (not just 2020+)
2. Better handles regional distribution
3. More efficient merging process
4. Adds location features from station reference
"""

import pandas as pd
import numpy as np
from pathlib import Path
from fuzzywuzzy import fuzz, process
import re
from datetime import datetime

# Configuration
TRAFFIC_FILE = "datasets/Traffic_TimesOfDay/road_traffic_counts_hourly_permanent/road_traffic_counts_hourly_permanent_all_with_location.csv"
WEATHER_DIR = "datasets/Weather_AQ"
OUTPUT_FILE = "datasets/TrafficWeatherwithSuburb/traffic_weather_merged_full.csv"

# Weather files
WEATHER_FILES = {
    'PM10': 'XLS-file_Daily_Averages-PM10_Time_Range_01012008_0000_to_02012025_0000.csv',
    'PM2_5': 'XLS-file_Daily_Averages-PM2-5_Time_Range_01012008_0000_to_02012025_0000.csv',
    'NO2': 'XLS-file_Daily_Averages-NO2_Time_Range_01012008_0000_to_02012025_0000.csv',
    'NO': 'XLS-file_Daily_Averages-NO_Time_Range_01012008_0000_to_02012025_0000.csv',
    'CO': 'XLS-file_Daily_Averages-CO_Time_Range_01012008_0000_to_02012025_0000.csv'
}

def load_weather_file(filepath, pollutant_name):
    """Load and process weather/air quality file"""
    print(f"\nLoading {pollutant_name} data from {filepath}...")
    
    # Read CSV, skip first 2 header rows
    df = pd.read_csv(filepath, skiprows=2, encoding='latin-1')
    df.columns = df.columns.str.strip()
    
    # Rename first column to 'Date'
    if df.columns[0] != 'Date':
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Extract suburb names from column headers
    new_cols = ['Date']
    for col in df.columns[1:]:
        # Extract suburb name (everything before pollutant info)
        # Fixed: Use single backslash for regex
        match = re.search(r'^(.+?)\s+(?:PM10|PM2\.5|PM2-5|NO2|NO|CO)\s+', col, re.IGNORECASE)
        if match:
            suburb = match.group(1).strip().upper()
        else:
            # Fallback: take everything before numbers or special patterns
            suburb = re.split(r'\s+\d', col)[0].strip().upper()
        new_cols.append(suburb)
    
    df.columns = new_cols
    
    # Remove columns with all NaN
    df = df.dropna(axis=1, how='all')
    
    # Debug: Show sample suburbs
    sample_suburbs = [col for col in df.columns[1:6] if col != 'Date']
    print(f"  ✓ Loaded {len(df)} dates, {len(df.columns)-1} suburbs")
    print(f"  Sample suburbs: {', '.join(sample_suburbs)}")
    
    return df

def fuzzy_match_suburbs(road_suburbs, weather_suburbs, threshold=75):
    """Match road suburbs to weather suburbs using fuzzy matching"""
    matches = {}
    near_misses = []
    
    for road_suburb in road_suburbs:
        best_match = process.extractOne(road_suburb, weather_suburbs, scorer=fuzz.token_sort_ratio)
        
        if best_match:
            if best_match[1] >= threshold:
                matches[road_suburb] = best_match[0]
            elif best_match[1] >= 60:  # Track near misses
                near_misses.append((road_suburb, best_match[0], best_match[1]))
    
    # Show near misses if no matches found
    if len(matches) == 0 and len(near_misses) > 0:
        print("\n⚠️ No matches found! Near misses (score 60-74):")
        for road, weather, score in near_misses[:10]:
            print(f"  {road:30} → {weather:30} (score: {score})")
    
    return matches

def merge_weather_data_efficient(road_df, weather_data, suburb_matches):
    """Efficiently merge all weather data at once"""
    print("\n" + "="*80)
    print("MERGING WEATHER DATA")
    print("="*80)
    
    # Prepare road data
    road_df = road_df.copy()
    road_df['date'] = pd.to_datetime(road_df['date'], errors='coerce')
    
    # Remove timezone if present
    if road_df['date'].dt.tz is not None:
        road_df['date'] = road_df['date'].dt.tz_localize(None)
    
    # Standardize suburb names
    road_df['suburb_std'] = road_df['suburb'].str.upper().str.strip()
    
    # Initialize pollutant columns
    for pollutant in weather_data.keys():
        road_df[pollutant] = np.nan
    
    # Create reverse mapping (weather suburb -> road suburb)
    reverse_matches = {v: k for k, v in suburb_matches.items()}
    
    # For each pollutant
    for pollutant_name, weather_df in weather_data.items():
        print(f"\nProcessing {pollutant_name}...")
        
        # Melt weather data to long format for easier merging
        weather_long = weather_df.melt(
            id_vars=['Date'],
            var_name='weather_suburb',
            value_name=pollutant_name
        )
        
        # Map weather suburbs to road suburbs
        weather_long['road_suburb'] = weather_long['weather_suburb'].map(reverse_matches)
        
        # Drop rows without matches
        weather_long = weather_long.dropna(subset=['road_suburb'])
        
        # Rename Date to date for merging
        weather_long = weather_long.rename(columns={'Date': 'date'})
        
        # Merge with road data
        road_df = road_df.merge(
            weather_long[['date', 'road_suburb', pollutant_name]],
            left_on=['date', 'suburb_std'],
            right_on=['date', 'road_suburb'],
            how='left',
            suffixes=('', '_new')
        )
        
        # Update the pollutant column with new values
        if f'{pollutant_name}_new' in road_df.columns:
            road_df[pollutant_name] = road_df[f'{pollutant_name}_new'].combine_first(road_df[pollutant_name])
            road_df = road_df.drop(columns=[f'{pollutant_name}_new', 'road_suburb'])
        
        non_null = road_df[pollutant_name].notna().sum()
        print(f"  ✓ {non_null:,} records with {pollutant_name} data ({non_null/len(road_df)*100:.1f}%)")
    
    return road_df

def main():
    print("="*80)
    print("TRAFFIC + WEATHER DATA MERGER")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load traffic data
    print("\n" + "="*80)
    print("STEP 1: LOADING TRAFFIC DATA")
    print("="*80)
    
    print(f"Loading: {TRAFFIC_FILE}")
    road_df = pd.read_csv(TRAFFIC_FILE, low_memory=False)
    print(f"✓ Loaded {len(road_df):,} traffic records")
    print(f"  Date range: {road_df['year'].min()} - {road_df['year'].max()}")
    print(f"  Unique suburbs: {road_df['suburb'].nunique()}")
    
    # Get unique road suburbs
    road_suburbs = road_df['suburb'].dropna().unique()
    road_suburbs = sorted([str(s).upper().strip() for s in road_suburbs if pd.notna(s)])
    print(f"  Standardized suburbs: {len(road_suburbs)}")
    print(f"  Sample road suburbs: {', '.join(road_suburbs[:10])}")
    
    # Step 2: Load weather data
    print("\n" + "="*80)
    print("STEP 2: LOADING WEATHER/AIR QUALITY DATA")
    print("="*80)
    
    weather_data = {}
    for pollutant, filename in WEATHER_FILES.items():
        filepath = Path(WEATHER_DIR) / filename
        weather_data[pollutant] = load_weather_file(str(filepath), pollutant)
    
    # Get all unique weather suburbs
    weather_suburbs = set()
    for df in weather_data.values():
        weather_suburbs.update([col for col in df.columns if col != 'Date'])
    weather_suburbs = sorted(list(weather_suburbs))
    print(f"\n✓ Total unique weather suburbs: {len(weather_suburbs)}")
    print(f"  Sample weather suburbs: {', '.join(weather_suburbs[:10])}")
    
    # Step 3: Match suburbs
    print("\n" + "="*80)
    print("STEP 3: MATCHING SUBURBS (FUZZY MATCHING)")
    print("="*80)
    
    suburb_matches = fuzzy_match_suburbs(road_suburbs, weather_suburbs, threshold=75)
    
    print(f"\n✓ Matched {len(suburb_matches)} suburbs:")
    for road_suburb, weather_suburb in sorted(suburb_matches.items())[:10]:
        print(f"  {road_suburb:30} → {weather_suburb}")
    if len(suburb_matches) > 10:
        print(f"  ... and {len(suburb_matches)-10} more")
    
    # Step 4: Merge data
    print("\n" + "="*80)
    print("STEP 4: MERGING DATA")
    print("="*80)
    
    merged_df = merge_weather_data_efficient(road_df, weather_data, suburb_matches)
    
    # Step 5: Save output
    print("\n" + "="*80)
    print("STEP 5: SAVING OUTPUT")
    print("="*80)
    
    # Create output directory if needed
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved to: {OUTPUT_FILE}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Step 6: Summary statistics
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\nTotal records: {len(merged_df):,}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    
    print(f"\nAir Quality Data Coverage:")
    for pollutant in ['PM10', 'PM2_5', 'NO2', 'NO', 'CO']:
        non_null = merged_df[pollutant].notna().sum()
        print(f"  {pollutant:6}: {non_null:,} records ({non_null/len(merged_df)*100:.1f}%)")
    
    print(f"\nRecords by RMS Region:")
    if 'rms_region' in merged_df.columns:
        region_counts = merged_df['rms_region'].value_counts()
        for region, count in region_counts.items():
            print(f"  {region:15}: {count:,} ({count/len(merged_df)*100:.1f}%)")
    
    print(f"\nRecords by Year:")
    year_counts = merged_df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:,}")
    
    print(f"\n✓ Processing complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
