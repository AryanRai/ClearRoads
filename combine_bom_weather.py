"""
Combine Bureau of Meteorology Weather Data
Combines weather data from multiple suburb folders into a single CSV

Structure:
- Each suburb has a folder with 4 CSV files:
  1.csv: Rainfall data
  2.csv: Solar exposure data
  3.csv: Minimum temperature data
  4.csv: Maximum temperature data
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
BOM_BASE_DIRS = [
    "datasets/Weather_Beuro_Meterology_PerDay/Substations_1/Substations_1",
    "datasets/Weather_Beuro_Meterology_PerDay/Substations_2/Substations_2",
    "datasets/Weather_Beuro_Meterology_PerDay/substations_34/my substations"
]

OUTPUT_FILE = "datasets/Weather_Beuro_Meterology_PerDay/bom_weather_combined.csv"

# File mapping
FILE_MAPPING = {
    '1.csv': 'rainfall_mm',
    '2.csv': 'solar_exposure_mj',
    '3.csv': 'min_temp_c',
    '4.csv': 'max_temp_c'
}

def load_bom_file(filepath, data_type):
    """Load a single BOM CSV file"""
    try:
        df = pd.read_csv(filepath)
        
        # Standardize column names
        df.columns = df.columns.str.strip()
        
        # Create date column properly
        df['date'] = pd.to_datetime(
            df['Year'].astype(str) + '-' + 
            df['Month'].astype(str).str.zfill(2) + '-' + 
            df['Day'].astype(str).str.zfill(2),
            format='%Y-%m-%d',
            errors='coerce'
        )
        
        # Identify the measurement column based on keywords
        measurement_col = None
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['rainfall', 'solar', 'temperature', 'temp']):
                measurement_col = col
                break
        
        if measurement_col is None:
            # Fallback: use column 5 (index 5) which is typically the measurement
            if len(df.columns) > 5:
                measurement_col = df.columns[5]
            else:
                print(f"  ⚠️ Could not identify measurement column in {filepath}")
                return None
        
        # Create result dataframe
        result = df[['date', measurement_col]].copy()
        result.columns = ['date', data_type]
        
        # Convert to numeric
        result[data_type] = pd.to_numeric(result[data_type], errors='coerce')
        
        # Remove rows with invalid dates
        result = result.dropna(subset=['date'])
        
        return result
        
    except Exception as e:
        print(f"  ⚠️ Error loading {filepath}: {e}")
        return None

def process_suburb(suburb_path):
    """Process all weather files for a single suburb"""
    suburb_name = suburb_path.name
    
    # Load all 4 files
    dfs = {}
    for filename, data_type in FILE_MAPPING.items():
        filepath = suburb_path / filename
        
        if filepath.exists():
            df = load_bom_file(filepath, data_type)
            if df is not None:
                dfs[data_type] = df
    
    if not dfs:
        return None
    
    # Merge all dataframes on date
    combined = None
    for data_type, df in dfs.items():
        if combined is None:
            combined = df
        else:
            combined = combined.merge(df, on='date', how='outer')
    
    # Add suburb column
    combined['suburb'] = suburb_name.upper().strip()
    
    return combined

def main():
    print("="*80)
    print("BUREAU OF METEOROLOGY WEATHER DATA COMBINER")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_data = []
    total_suburbs = 0
    processed_suburbs = 0
    
    # Process each base directory
    for base_dir in BOM_BASE_DIRS:
        base_path = Path(base_dir)
        
        if not base_path.exists():
            print(f"\n⚠️ Directory not found: {base_dir}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing: {base_dir}")
        print(f"{'='*80}")
        
        # Get all suburb folders
        suburb_folders = [d for d in base_path.iterdir() if d.is_dir()]
        print(f"Found {len(suburb_folders)} suburb folders")
        
        # Process each suburb
        for suburb_folder in sorted(suburb_folders):
            total_suburbs += 1
            suburb_name = suburb_folder.name
            
            print(f"\n  Processing: {suburb_name}")
            
            # Check if all 4 files exist
            files_exist = [
                (suburb_folder / filename).exists() 
                for filename in FILE_MAPPING.keys()
            ]
            print(f"    Files found: {sum(files_exist)}/4")
            
            # Process suburb data
            suburb_data = process_suburb(suburb_folder)
            
            if suburb_data is not None:
                all_data.append(suburb_data)
                processed_suburbs += 1
                print(f"    ✓ Loaded {len(suburb_data):,} records")
                print(f"      Date range: {suburb_data['date'].min()} to {suburb_data['date'].max()}")
            else:
                print(f"    ✗ Failed to process")
    
    # Combine all data
    print(f"\n{'='*80}")
    print("COMBINING ALL DATA")
    print(f"{'='*80}")
    
    if not all_data:
        print("❌ No data to combine!")
        return
    
    print(f"Combining data from {len(all_data)} suburbs...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by suburb and date
    combined_df = combined_df.sort_values(['suburb', 'date'])
    
    # Remove rows with no data
    data_cols = list(FILE_MAPPING.values())
    combined_df = combined_df.dropna(subset=data_cols, how='all')
    
    print(f"✓ Combined {len(combined_df):,} total records")
    
    # Save to CSV
    print(f"\n{'='*80}")
    print("SAVING OUTPUT")
    print(f"{'='*80}")
    
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved to: {OUTPUT_FILE}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nTotal suburbs processed: {processed_suburbs}/{total_suburbs}")
    print(f"Total records: {len(combined_df):,}")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    print(f"\nUnique suburbs: {combined_df['suburb'].nunique()}")
    print(f"Top 10 suburbs by record count:")
    top_suburbs = combined_df['suburb'].value_counts().head(10)
    for suburb, count in top_suburbs.items():
        print(f"  {suburb:30}: {count:,} records")
    
    print(f"\nData completeness:")
    for col in data_cols:
        non_null = combined_df[col].notna().sum()
        print(f"  {col:20}: {non_null:,} ({non_null/len(combined_df)*100:.1f}%)")
    
    print(f"\nSample statistics:")
    print(combined_df[data_cols].describe())
    
    print(f"\n✓ Processing complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Show sample data
    print(f"\nSample data (first 10 rows):")
    print(combined_df.head(10).to_string())

if __name__ == "__main__":
    main()
