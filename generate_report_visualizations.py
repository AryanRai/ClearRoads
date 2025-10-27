"""
Generate additional visualizations and analysis for the report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("GENERATING ADDITIONAL REPORT VISUALIZATIONS")
print("="*80)

# Load and prepare data (using same preprocessing as main script)
print("\n⏳ Loading and preparing data...")
filepath = "datasets/TrafficWeather_Beuro_AQ_withSuburb/complete_traffic_environment_data.csv"
df = pd.read_csv(filepath, low_memory=False)

# Quick preprocessing
df['date'] = pd.to_datetime(df['date'])
bool_cols = ['public_holiday', 'school_holiday']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0, 1: 1, 0: 0})
        df[col] = df[col].fillna(0).astype(int)

# Filter to environmental data
env_cols = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO', 'rainfall_mm', 
            'solar_exposure_mj', 'min_temp_c', 'max_temp_c']
env_present = [col for col in env_cols if col in df.columns]
df = df[df[env_present].notna().any(axis=1)]
df = df.dropna(subset=['daily_total'])

# Remove outliers
q1 = df['daily_total'].quantile(0.01)
q99 = df['daily_total'].quantile(0.99)
df = df[(df['daily_total'] >= q1) & (df['daily_total'] <= q99)]

print(f"✓ Data prepared: {len(df):,} records")

# ============================================================================
# 1. TRAFFIC PATTERNS BY REGION
# ============================================================================

print("\n📊 Creating regional traffic analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1a. Average traffic by region
if 'rms_region' in df.columns:
    region_traffic = df.groupby('rms_region')['daily_total'].agg(['mean', 'median', 'std'])
    region_traffic = region_traffic.sort_values('mean', ascending=False)
    
    ax = axes[0, 0]
    x = np.arange(len(region_traffic))
    ax.bar(x, region_traffic['mean'], alpha=0.7, label='Mean', color='steelblue')
    ax.errorbar(x, region_traffic['mean'], yerr=region_traffic['std'], 
                fmt='none', color='black', capsize=5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(region_traffic.index, rotation=45, ha='right')
    ax.set_ylabel('Daily Traffic Volume', fontsize=12)
    ax.set_title('Average Daily Traffic by Region', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

# 1b. Traffic distribution by distance to CBD
if 'distance_to_cbd_km' in df.columns:
    ax = axes[0, 1]
    # Create distance bins
    df['distance_bin'] = pd.cut(df['distance_to_cbd_km'], 
                                 bins=[0, 10, 25, 50, 100, 1000],
                                 labels=['0-10km', '10-25km', '25-50km', '50-100km', '>100km'])
    
    distance_traffic = df.groupby('distance_bin')['daily_total'].mean()
    ax.bar(range(len(distance_traffic)), distance_traffic.values, color='coral', alpha=0.7)
    ax.set_xticks(range(len(distance_traffic)))
    ax.set_xticklabels(distance_traffic.index, rotation=45, ha='right')
    ax.set_ylabel('Average Daily Traffic', fontsize=12)
    ax.set_title('Traffic Volume vs Distance from CBD', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

# 1c. Seasonal traffic patterns
df['season'] = df['month'].map({
    12: 'Summer', 1: 'Summer', 2: 'Summer',
    3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
    6: 'Winter', 7: 'Winter', 8: 'Winter',
    9: 'Spring', 10: 'Spring', 11: 'Spring'
})

ax = axes[1, 0]
season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
season_data = [df[df['season'] == s]['daily_total'].mean() for s in season_order]
colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#95E1D3']
ax.bar(season_order, season_data, color=colors, alpha=0.7)
ax.set_ylabel('Average Daily Traffic', fontsize=12)
ax.set_title('Seasonal Traffic Patterns', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 1d. Weekend vs Weekday
df['is_weekend'] = df['day_of_week'].isin([6, 7])
ax = axes[1, 1]
weekend_data = df.groupby('is_weekend')['daily_total'].mean()
ax.bar(['Weekday', 'Weekend'], weekend_data.values, color=['steelblue', 'coral'], alpha=0.7)
ax.set_ylabel('Average Daily Traffic', fontsize=12)
ax.set_title('Weekday vs Weekend Traffic', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report_traffic_patterns_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: report_traffic_patterns_analysis.png")
plt.close()

# ============================================================================
# 2. ENVIRONMENTAL CORRELATIONS
# ============================================================================

print("\n🌫️ Creating environmental correlation analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Create congestion classes
p25 = df['daily_total'].quantile(0.25)
p50 = df['daily_total'].quantile(0.50)
p75 = df['daily_total'].quantile(0.75)

def classify_congestion(value):
    if value < p25:
        return 'Very Low'
    elif value < p50:
        return 'Low'
    elif value < p75:
        return 'High'
    else:
        return 'Very High'

df['Congestion_Class'] = df['daily_total'].apply(classify_congestion)

# Plot environmental factors by congestion level
env_features = ['PM2_5', 'PM10', 'NO2', 'rainfall_mm', 'min_temp_c', 'max_temp_c']
class_order = ['Very Low', 'Low', 'High', 'Very High']

for idx, feature in enumerate(env_features):
    if feature in df.columns:
        ax = axes[idx // 3, idx % 3]
        
        # Calculate means by congestion class
        feature_by_class = df.groupby('Congestion_Class')[feature].mean().reindex(class_order)
        
        ax.bar(class_order, feature_by_class.values, color='teal', alpha=0.7)
        ax.set_xlabel('Congestion Level', fontsize=11)
        ax.set_ylabel(f'Average {feature}', fontsize=11)
        ax.set_title(f'{feature} by Congestion Level', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report_environmental_correlations.png', dpi=300, bbox_inches='tight')
print("✓ Saved: report_environmental_correlations.png")
plt.close()

# ============================================================================
# 3. MODEL COMPARISON CHART
# ============================================================================

print("\n🤖 Creating model comparison visualization...")

models_data = {
    'Model': ['kNN', 'Decision\nTree', 'Random\nForest', 'Neural\nNetwork', 'XGBoost'],
    'Accuracy': [0.8713, 0.9717, 0.9809, 0.9791, 0.9830],
    'CV_Mean': [0.8497, 0.9714, 0.9809, 0.9750, 0.9826],
    'CV_Std': [0.0015, 0.0009, 0.0005, 0.0013, 0.0004]
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 3a. Test Accuracy Comparison
ax = axes[0]
colors_map = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
bars = ax.bar(models_data['Model'], models_data['Accuracy'], color=colors_map, alpha=0.8)
ax.axhline(y=0.25, color='red', linestyle='--', label='Baseline (25%)', linewidth=2)
ax.set_ylabel('Test Accuracy', fontsize=13)
ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2%}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3b. Cross-Validation Stability
ax = axes[1]
ax.bar(models_data['Model'], models_data['CV_Mean'], 
       yerr=models_data['CV_Std'], capsize=5, color=colors_map, alpha=0.8)
ax.set_ylabel('Cross-Validation Accuracy', fontsize=13)
ax.set_title('Model Stability (5-Fold CV)', fontsize=15, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: report_model_comparison.png")
plt.close()

# ============================================================================
# 4. LOCATION FEATURE IMPACT
# ============================================================================

print("\n🗺️ Creating location feature impact analysis...")

# Train a quick model to get feature importances
print("  Training model for feature analysis...")

# Prepare features
air_quality_features = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO']
weather_features = ['rainfall_mm', 'solar_exposure_mj', 'min_temp_c', 'max_temp_c']
temporal_features = ['month', 'day_of_week', 'public_holiday', 'school_holiday', 'is_weekend', 'year']

# Create engineered features
hour_cols = [f'hour_{i:02d}' for i in range(24) if f'hour_{i:02d}' in df.columns]
if hour_cols:
    df['morning_rush'] = df[[f'hour_{i:02d}' for i in range(6, 10) if f'hour_{i:02d}' in df.columns]].sum(axis=1)
    df['evening_rush'] = df[[f'hour_{i:02d}' for i in range(16, 20) if f'hour_{i:02d}' in df.columns]].sum(axis=1)
    df['peak_hour_traffic'] = df[hour_cols].max(axis=1)

traffic_features = ['morning_rush', 'evening_rush', 'peak_hour_traffic']
location_continuous = ['distance_to_cbd_km']

# Impute missing values
for col in air_quality_features + weather_features:
    if col in df.columns:
        df[col] = df.groupby('suburb_std')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

feature_cols = []
for col in air_quality_features + weather_features + temporal_features + traffic_features + location_continuous:
    if col in df.columns:
        feature_cols.append(col)

# One-hot encode
season_dummies = pd.get_dummies(df['season'], prefix='season')
df = pd.concat([df, season_dummies], axis=1)
feature_cols.extend(season_dummies.columns.tolist())

if 'rms_region' in df.columns:
    region_dummies = pd.get_dummies(df['rms_region'], prefix='region')
    df = pd.concat([df, region_dummies], axis=1)
    feature_cols.extend(region_dummies.columns.tolist())

X = df[feature_cols].copy()
y = df['Congestion_Class'].copy()

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
])

pipeline.fit(X_train, y_train)
classifier = pipeline.named_steps['classifier']
importances = classifier.feature_importances_

# Categorize features
feat_imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
})

location_features = [f for f in feature_cols if any(x in f for x in ['region_', 'distance_to_cbd'])]
traffic_features_list = [f for f in feature_cols if any(x in f for x in ['morning_rush', 'evening_rush', 'peak_hour'])]
air_quality_features_list = [f for f in feature_cols if any(x in f for x in ['PM', 'NO', 'CO'])]
weather_features_list = [f for f in feature_cols if any(x in f for x in ['rainfall', 'solar', 'temp'])]
temporal_features_list = [f for f in feature_cols if any(x in f for x in ['month', 'day_of_week', 'year', 'weekend', 'holiday', 'season'])]

categories = {
    'Traffic Patterns': traffic_features_list,
    'Location': location_features,
    'Air Quality': air_quality_features_list,
    'Weather': weather_features_list,
    'Temporal': temporal_features_list
}

category_importance = {}
for cat, features in categories.items():
    category_importance[cat] = feat_imp_df[feat_imp_df['Feature'].isin(features)]['Importance'].sum()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 4a. Category importance pie chart
ax = axes[0]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
wedges, texts, autotexts = ax.pie(category_importance.values(), labels=category_importance.keys(),
                                    autopct='%1.1f%%', colors=colors, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
ax.set_title('Feature Category Importance', fontsize=15, fontweight='bold')

# 4b. Location features breakdown
ax = axes[1]
location_imp = feat_imp_df[feat_imp_df['Feature'].isin(location_features)].sort_values('Importance', ascending=True)
if len(location_imp) > 0:
    ax.barh(location_imp['Feature'], location_imp['Importance'], color='teal', alpha=0.7)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Location Feature Importance Breakdown', fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('report_location_feature_impact.png', dpi=300, bbox_inches='tight')
print("✓ Saved: report_location_feature_impact.png")
plt.close()

# ============================================================================
# 5. PERFORMANCE METRICS TABLE
# ============================================================================

print("\n📋 Creating performance metrics summary...")

# Create detailed metrics table
metrics_data = [
    ['kNN (k=5)', '87.13%', '84.97 ± 0.15%', 'Fast', 'Medium'],
    ['Decision Tree', '97.17%', '97.14 ± 0.09%', 'Fast', 'High'],
    ['Random Forest', '98.09%', '98.09 ± 0.05%', 'Medium', 'Medium'],
    ['Neural Network', '97.91%', '97.50 ± 0.13%', 'Slow', 'Low'],
    ['XGBoost', '98.30%', '98.26 ± 0.04%', 'Medium', 'Medium']
]

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=metrics_data,
                colLabels=['Model', 'Test Accuracy', 'CV Accuracy', 'Training Time', 'Interpretability'],
                cellLoc='center',
                loc='center',
                colWidths=[0.2, 0.15, 0.25, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for j in range(5):
    table[(0, j)].set_facecolor('#4ECDC4')
    table[(0, j)].set_text_props(weight='bold', color='white')

# Color code the best model (XGBoost - row 5)
for j in range(5):
    table[(5, j)].set_facecolor('#E8F5E9')
    table[(5, j)].set_text_props(weight='bold')

plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig('report_performance_metrics_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: report_performance_metrics_table.png")
plt.close()

print("\n✅ All visualizations generated successfully!")
print("\n📁 Generated files:")
print("  1. report_traffic_patterns_analysis.png")
print("  2. report_environmental_correlations.png")
print("  3. report_model_comparison.png")
print("  4. report_location_feature_impact.png")
print("  5. report_performance_metrics_table.png")
print("="*80)
