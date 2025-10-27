"""
Helper script to extract suburb information from your dataset
Run this to get the suburb list for the AI prompt
"""

import pandas as pd

# Load data
print("Loading data...")
df = pd.read_csv("datasets/TrafficWeatherwithSuburb/roadandweathermerged-20251020T083319Z-1-001/roadandweathermerged/output_merge.csv")

# Get unique suburbs
suburbs = sorted(df['suburb_std'].unique())

print("\n" + "="*80)
print("COPY THIS LIST FOR AI PROMPT")
print("="*80)
print("\nSuburbs in my dataset:")
print(", ".join(suburbs))

print("\n" + "="*80)
print("SUBURB COUNTS (for AI context)")
print("="*80)
suburb_counts = df['suburb_std'].value_counts().sort_values(ascending=False)
for suburb, count in suburb_counts.items():
    print(f"{suburb}: {count} observations")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total unique suburbs: {len(suburbs)}")
print(f"Total observations: {len(df)}")
print(f"Average observations per suburb: {len(df) / len(suburbs):.0f}")
print(f"Most common suburb: {suburb_counts.index[0]} ({suburb_counts.iloc[0]} obs)")
print(f"Least common suburb: {suburb_counts.index[-1]} ({suburb_counts.iloc[-1]} obs)")
print("="*80)

# Save to file for easy reference
with open('suburb_list.txt', 'w') as f:
    f.write("SUBURB LIST FOR AI PROMPT\n")
    f.write("="*80 + "\n\n")
    f.write(", ".join(suburbs))
    f.write("\n\n" + "="*80 + "\n")
    f.write("SUBURB COUNTS\n")
    f.write("="*80 + "\n")
    for suburb, count in suburb_counts.items():
        f.write(f"{suburb}: {count} observations\n")

print("\n✅ Suburb list also saved to 'suburb_list.txt'")
