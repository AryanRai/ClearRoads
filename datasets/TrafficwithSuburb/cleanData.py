import pandas as pd
from pathlib import Path
import re

# --- your constants ---
FILE1 = "ProjectProposal/datasets/Traffic_TimesOfDay/road_traffic_counts_hourly_permanent/road_traffic_counts_hourly_permanent0.csv"
FILE2 = "ProjectProposal/datasets/Traffic_TimesOfDay/road_traffic_counts_station_reference.csv"
KEY   = "station_key"
SUBURB_COL = "suburb"
# ----------------------

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df

def as_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

# 1) Load reference file (station_key + suburb)
ref = pd.read_csv(FILE2)
ref = normalize_cols(ref)

missing = {KEY, SUBURB_COL} - set(ref.columns)
if missing:
    raise KeyError(f"{FILE2} missing columns: {missing}")

ref[KEY] = as_key(ref[KEY])
ref_dedup = ref.drop_duplicates(subset=[KEY], keep="first")[[KEY, SUBURB_COL]]

# 2) Find all hourly CSVs 0–3
m = re.search(r"(.*?)(\d)(\.csv)$", FILE1)
if m:
    prefix, _, suffix = m.groups()
    file1_paths = [f"{prefix}{i}{suffix}" for i in range(0, 4)]
    file1_paths = [p for p in file1_paths if Path(p).is_file()]
else:
    file1_paths = [FILE1]

print("Will process:")
for p in file1_paths:
    print(" -", p)

# 3) Merge and stack all four
merged_parts = []
for p in file1_paths:
    df1 = pd.read_csv(p)
    df1 = normalize_cols(df1)

    if KEY not in df1.columns:
        raise KeyError(f"{p} is missing key column '{KEY}'\nColumns: {list(df1.columns)}")

    df1[KEY] = as_key(df1[KEY])
    merged = df1.merge(ref_dedup, on=KEY, how="left")
    merged_parts.append(merged)

# 4) Combine and export one CSV
combined = pd.concat(merged_parts, ignore_index=True)
out_path = str(Path(file1_paths[0]).with_name("road_traffic_counts_hourly_permanent_all_with_suburb.csv"))
combined.to_csv(out_path, index=False)

print(f"✓ Combined {len(file1_paths)} files → {Path(out_path).name}")
print(f"Total rows: {len(combined)}, with suburbs matched: {combined[SUBURB_COL].notna().sum()}")