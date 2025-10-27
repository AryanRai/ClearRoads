"""
ClearRoads: Enhanced Traffic Congestion Prediction with Location Features
ENGG2112 - Multidisciplinary Engineering Project
Authors: Aryan Rai, Nixie Nassar, Nell Nesci

Version 2.0 - Includes:
- Regional grouping (Strategy 3: Hybrid Approach)
- Distance to CBD
- Urban classification
- BOM weather data
- Enhanced air quality features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             ConfusionMatrixDisplay)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Urban classification based on population density and distance from CBD
URBAN_CLASSIFICATION = {
    # Sydney Metro - Inner
    'SYDNEY': 'Urban', 'ALEXANDRIA': 'Urban', 'ULTIMO': 'Urban', 'ROZELLE': 'Urban',
    'DRUMMOYNE': 'Urban', 'BALMAIN': 'Urban', 'GLEBE': 'Urban', 'REDFERN': 'Urban',
    'PYRMONT': 'Urban', 'SURRY HILLS': 'Urban', 'DARLINGHURST': 'Urban',
    
    # Sydney Metro - Eastern
    'BONDI': 'Urban', 'RANDWICK': 'Urban', 'KENSINGTON': 'Urban', 'MAROUBRA': 'Urban',
    'COOGEE': 'Urban', 'MASCOT': 'Urban', 'BOTANY': 'Urban',
    
    # Sydney Metro - Inner West
    'ASHFIELD': 'Urban', 'BURWOOD': 'Urban', 'STRATHFIELD': 'Urban', 'HOMEBUSH': 'Urban',
    'CONCORD': 'Urban', 'FIVE DOCK': 'Urban',
    
    # Sydney Metro - North Shore
    'CHATSWOOD': 'Urban', 'NORTH SYDNEY': 'Urban', 'ST LEONARDS': 'Urban',
    'WILLOUGHBY': 'Urban', 'LANE COVE': 'Urban', 'ARTARMON': 'Urban',
    
    # Sydney Metro - South
    'HURSTVILLE': 'Urban', 'KOGARAH': 'Urban', 'ROCKDALE': 'Urban', 'ARNCLIFFE': 'Urban',
    
    # Sydney - Suburban
    'PARRAMATTA': 'Suburban', 'BANKSTOWN': 'Suburban', 'LIVERPOOL': 'Suburban',
    'BLACKTOWN': 'Suburban', 'PENRITH': 'Suburban', 'CAMPBELLTOWN': 'Suburban',
    'HORNSBY': 'Suburban', 'RYDE': 'Suburban', 'EPPING': 'Suburban',
    'SUTHERLAND': 'Suburban', 'ENGADINE': 'Suburban', 'CARINGBAH': 'Suburban',
    'MIRANDA': 'Suburban', 'CRONULLA': 'Suburban', 'MANLY': 'Suburban',
    'BROOKVALE': 'Suburban', 'DEE WHY': 'Suburban', 'MONA VALE': 'Suburban',
    'CASTLE HILL': 'Suburban', 'BAULKHAM HILLS': 'Suburban', 'CARLINGFORD': 'Suburban',
    'GREENACRE': 'Suburban', 'LAKEMBA': 'Suburban', 'PUNCHBOWL': 'Suburban',
    
    # Hunter Region
    'NEWCASTLE': 'Regional_City', 'MAITLAND': 'Regional_City', 'CESSNOCK': 'Regional_City',
    'CHARLESTOWN': 'Regional_City', 'ADAMSTOWN': 'Regional_City', 'BROADMEADOW': 'Regional_City',
    'HAMILTON': 'Regional_City', 'MAYFIELD': 'Regional_City', 'WALLSEND': 'Regional_City',
    'BELMONT': 'Regional_City', 'TORONTO': 'Regional_City', 'ARGENTON': 'Regional_City',
    
    # Illawarra
    'WOLLONGONG': 'Regional_City', 'SHELLHARBOUR': 'Regional_City', 'PORT KEMBLA': 'Regional_City',
    'DAPTO': 'Regional_City', 'WARRAWONG': 'Regional_City',
    
    # Regional NSW
    'ALBURY': 'Regional', 'WAGGA WAGGA': 'Regional', 'DUBBO': 'Regional',
    'TAMWORTH': 'Regional', 'ORANGE': 'Regional', 'BATHURST': 'Regional',
    'NOWRA': 'Regional', 'QUEANBEYAN': 'Regional', 'GOULBURN': 'Regional',
    'LISMORE': 'Regional', 'COFFS HARBOUR': 'Regional', 'PORT MACQUARIE': 'Regional',
    'GRAFTON': 'Regional', 'ARMIDALE': 'Regional', 'BROKEN HILL': 'Regional',
}

# ============================================================================
# Q1: IMPORT AND DESCRIPTIVES
# ============================================================================

def load_and_explore_data(filepath):
    """Load dataset and perform initial exploration"""
    print("="*80)
    print("Q1: IMPORT AND DESCRIPTIVES - ENHANCED WITH LOCATION FEATURES")
    print("="*80)
    
    # Load data with low_memory=False to handle mixed types
    print("\n⏳ Loading dataset (this may take a moment)...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"\n✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Drop unnecessary columns
    cols_to_drop = ['the_geom', 'the_geom_webmercator', 'cartodb_id', 
                    'record_id', 'md5', 'updated_on', 'suburb', 'lga']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    print(f"✓ Dropped {len([c for c in cols_to_drop if c in df.columns])} unnecessary columns")
    
    # Show column categories
    print(f"\n📋 Final Column List ({len(df.columns)} columns):")
    print("-" * 80)
    
    traffic_cols = ['daily_total'] + [f'hour_{i:02d}' for i in range(24)]
    air_quality_cols = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO']
    weather_cols = ['rainfall_mm', 'solar_exposure_mj', 'min_temp_c', 'max_temp_c']
    location_cols = ['suburb_std', 'rms_region', 'distance_to_cbd_km', 
                     'wgs84_latitude', 'wgs84_longitude']
    temporal_cols = ['date', 'year', 'month', 'day_of_week', 'public_holiday', 'school_holiday']
    
    print(f"Traffic Features: {len([c for c in traffic_cols if c in df.columns])}")
    print(f"Air Quality Features: {len([c for c in air_quality_cols if c in df.columns])}")
    print(f"Weather Features: {len([c for c in weather_cols if c in df.columns])}")
    print(f"Location Features: {len([c for c in location_cols if c in df.columns])}")
    print(f"Temporal Features: {len([c for c in temporal_cols if c in df.columns])}")
    
    # Data completeness
    print("\n📊 Data Completeness:")
    print("-" * 80)
    
    total_records = len(df)
    
    # Environmental features
    env_cols = air_quality_cols + weather_cols
    env_present = [col for col in env_cols if col in df.columns]
    
    print("\nEnvironmental Features:")
    for col in env_present:
        count = df[col].notna().sum()
        pct = (count / total_records) * 100
        print(f"  {col:20s}: {count:>10,} ({pct:>5.1f}%)")
    
    # Records with ANY environmental data
    any_env = df[env_present].notna().any(axis=1).sum()
    print(f"\nRecords with ANY environmental data: {any_env:,} ({any_env/total_records*100:.1f}%)")
    
    # Location features
    print("\nLocation Features:")
    if 'rms_region' in df.columns:
        print(f"\nRegional Distribution:")
        region_dist = df['rms_region'].value_counts()
        for region, count in region_dist.items():
            print(f"  {region:15s}: {count:>10,} ({count/total_records*100:>5.1f}%)")
    
    if 'distance_to_cbd_km' in df.columns:
        print(f"\nDistance to CBD Statistics:")
        print(df['distance_to_cbd_km'].describe())
    
    return df

# ============================================================================
# Q2: DATA CLEANING & PREPROCESSING WITH LOCATION FEATURES
# ============================================================================

def clean_and_preprocess(df):
    """Clean and preprocess the dataset with location features"""
    print("\n" + "="*80)
    print("Q2: DATA CLEANING & PREPROCESSING WITH LOCATION FEATURES")
    print("="*80)
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # Convert date to datetime
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    # Handle boolean columns
    bool_cols = ['public_holiday', 'school_holiday']
    for col in bool_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map({'True': 1, 'False': 0, True: 1, False: 0, 1: 1, 0: 0})
            df_clean[col] = df_clean[col].fillna(0).astype(int)
    
    print("\n🧹 Data Cleaning Steps:")
    print("-" * 80)
    
    # Step 1: Filter to records with environmental data
    env_cols = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO', 'rainfall_mm', 
                'solar_exposure_mj', 'min_temp_c', 'max_temp_c']
    env_present = [col for col in env_cols if col in df_clean.columns]
    
    before = len(df_clean)
    df_clean = df_clean[df_clean[env_present].notna().any(axis=1)]
    print(f"✓ Filtered to records with environmental data: {len(df_clean):,} ({len(df_clean)/before*100:.1f}%)")
    
    # Step 2: Remove rows with missing daily_total
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=['daily_total'])
    print(f"✓ Removed {before - len(df_clean):,} rows with missing daily_total")
    
    # Step 3: Remove rows with missing location features
    location_required = ['suburb_std', 'rms_region', 'distance_to_cbd_km']
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=[col for col in location_required if col in df_clean.columns])
    print(f"✓ Removed {before - len(df_clean):,} rows with missing location data")
    
    # Step 4: Impute environmental features
    print("\n🔧 Feature Engineering:")
    print("-" * 80)
    
    # Impute air quality with suburb-specific medians
    air_quality_cols = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO']
    for col in air_quality_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean.groupby('suburb_std')[col].transform(
                lambda x: x.fillna(x.median())
            )
            # If still missing, use global median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Impute weather with suburb-specific medians
    weather_cols = ['rainfall_mm', 'solar_exposure_mj', 'min_temp_c', 'max_temp_c']
    for col in weather_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean.groupby('suburb_std')[col].transform(
                lambda x: x.fillna(x.median())
            )
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    print(f"✓ Imputed environmental features with suburb-specific medians")
    
    # Temporal features
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([6, 7]).astype(int)
    df_clean['season'] = df_clean['month'].map({
        12: 'Summer', 1: 'Summer', 2: 'Summer',
        3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
        6: 'Winter', 7: 'Winter', 8: 'Winter',
        9: 'Spring', 10: 'Spring', 11: 'Spring'
    })
    print(f"✓ Created temporal features: is_weekend, season")
    
    # Composite Air Quality Index
    if all(col in df_clean.columns for col in air_quality_cols):
        df_clean['AQI_composite'] = (
            df_clean['PM2_5'].fillna(0) * 0.3 +
            df_clean['PM10'].fillna(0) * 0.25 +
            df_clean['NO2'].fillna(0) * 0.25 +
            df_clean['CO'].fillna(0) * 0.1 +
            df_clean['NO'].fillna(0) * 0.1
        )
        print(f"✓ Created composite Air Quality Index (AQI)")
    
    # Traffic pattern features
    hour_cols = [f'hour_{i:02d}' for i in range(24) if f'hour_{i:02d}' in df_clean.columns]
    if hour_cols:
        df_clean['morning_rush'] = df_clean[[f'hour_{i:02d}' for i in range(6, 10) if f'hour_{i:02d}' in df_clean.columns]].sum(axis=1)
        df_clean['evening_rush'] = df_clean[[f'hour_{i:02d}' for i in range(16, 20) if f'hour_{i:02d}' in df_clean.columns]].sum(axis=1)
        df_clean['peak_hour_traffic'] = df_clean[hour_cols].max(axis=1)
        print(f"✓ Created traffic pattern features: morning_rush, evening_rush, peak_hour_traffic")
    
    # LOCATION FEATURES (Strategy 3: Hybrid Approach)
    print("\n🗺️ Location Features (Hybrid Approach):")
    print("-" * 80)
    
    # 1. Urban classification
    df_clean['urban_type'] = df_clean['suburb_std'].map(URBAN_CLASSIFICATION)
    df_clean['urban_type'] = df_clean['urban_type'].fillna('Suburban')  # Default for unmapped
    print(f"✓ Created urban_type classification")
    print(f"  Distribution: {df_clean['urban_type'].value_counts().to_dict()}")
    
    # 2. Distance to CBD (already in dataset)
    print(f"✓ Using distance_to_cbd_km (already in dataset)")
    
    # 3. Regional grouping (already in dataset as rms_region)
    print(f"✓ Using rms_region (already in dataset)")
    
    # Remove outliers in daily_total
    q1 = df_clean['daily_total'].quantile(0.01)
    q99 = df_clean['daily_total'].quantile(0.99)
    before = len(df_clean)
    df_clean = df_clean[(df_clean['daily_total'] >= q1) & (df_clean['daily_total'] <= q99)]
    print(f"\n✓ Removed {before - len(df_clean):,} outliers (< 1st or > 99th percentile)")
    
    print(f"\n📈 Final dataset: {len(df_clean):,} rows ({(len(df_clean)/initial_rows)*100:.1f}% of original)")
    
    return df_clean

# ============================================================================
# Q3: CREATE TARGET - TRAFFIC CONGESTION CLASS
# ============================================================================

def create_target_classes(df):
    """Create multiclass target for traffic congestion"""
    print("\n" + "="*80)
    print("Q3: CREATE TARGET - TRAFFIC CONGESTION CLASS")
    print("="*80)
    
    df_target = df.copy()
    
    # Calculate percentiles
    p25 = df_target['daily_total'].quantile(0.25)
    p50 = df_target['daily_total'].quantile(0.50)
    p75 = df_target['daily_total'].quantile(0.75)
    
    print(f"\n📊 Traffic Volume Percentiles:")
    print(f"  25th percentile: {p25:,.0f}")
    print(f"  50th percentile: {p50:,.0f}")
    print(f"  75th percentile: {p75:,.0f}")
    
    # Create 4-class target
    def classify_congestion(value):
        if value < p25:
            return 'Very Low'
        elif value < p50:
            return 'Low'
        elif value < p75:
            return 'High'
        else:
            return 'Very High'
    
    df_target['Congestion_Class'] = df_target['daily_total'].apply(classify_congestion)
    
    # Class distribution
    print(f"\n🎯 Congestion Class Distribution:")
    print("-" * 80)
    class_dist = df_target['Congestion_Class'].value_counts().sort_index()
    class_pct = (class_dist / len(df_target) * 100).round(2)
    
    dist_df = pd.DataFrame({
        'Count': class_dist,
        'Percentage': class_pct
    })
    print(dist_df)
    
    # Majority baseline
    majority_class = class_dist.idxmax()
    baseline_accuracy = class_dist.max() / len(df_target)
    print(f"\n📌 Majority Class Baseline:")
    print(f"  Majority class: {majority_class}")
    print(f"  Baseline accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    class_order = ['Very Low', 'Low', 'High', 'Very High']
    sns.barplot(x=class_order, y=[class_dist.get(c, 0) for c in class_order], palette='viridis')
    plt.title('Traffic Congestion Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Congestion Class', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=0)
    for i, v in enumerate([class_dist.get(c, 0) for c in class_order]):
        plt.text(i, v + 500, f'{v:,}\n({v/len(df_target)*100:.1f}%)', 
                ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('congestion_class_distribution_v2.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved as 'congestion_class_distribution_v2.png'")
    plt.close()
    
    return df_target, baseline_accuracy

# ============================================================================
# Q4: MODEL DEVELOPMENT WITH LOCATION FEATURES
# ============================================================================

def prepare_features(df):
    """Prepare features for modeling with location features"""
    # Air quality features
    air_quality_features = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO', 'AQI_composite']
    
    # Weather features
    weather_features = ['rainfall_mm', 'solar_exposure_mj', 'min_temp_c', 'max_temp_c']
    
    # Temporal features
    temporal_features = ['month', 'day_of_week', 'public_holiday', 'school_holiday', 
                        'is_weekend', 'year']
    
    # Traffic features
    traffic_features = ['morning_rush', 'evening_rush', 'peak_hour_traffic']
    
    # Location features (continuous)
    location_continuous = ['distance_to_cbd_km']
    
    # Combine all continuous features
    feature_cols = []
    for col in air_quality_features + weather_features + temporal_features + traffic_features + location_continuous:
        if col in df.columns:
            feature_cols.append(col)
    
    # Encode season (categorical)
    if 'season' in df.columns:
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        feature_cols.extend(season_dummies.columns.tolist())
    
    # Encode urban_type (categorical) - LOCATION FEATURE
    if 'urban_type' in df.columns:
        urban_dummies = pd.get_dummies(df['urban_type'], prefix='urban')
        df = pd.concat([df, urban_dummies], axis=1)
        feature_cols.extend(urban_dummies.columns.tolist())
    
    # Encode rms_region (categorical) - LOCATION FEATURE
    if 'rms_region' in df.columns:
        region_dummies = pd.get_dummies(df['rms_region'], prefix='region')
        df = pd.concat([df, region_dummies], axis=1)
        feature_cols.extend(region_dummies.columns.tolist())
    
    X = df[feature_cols].copy()
    y = df['Congestion_Class'].copy()
    
    # Encode target labels for XGBoost compatibility
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y, y_encoded, feature_cols, label_encoder

def train_and_evaluate_models(X, y, y_encoded, label_encoder, baseline_accuracy):
    """Train and evaluate multiple models"""
    print("\n" + "="*80)
    print("Q4: MODEL DEVELOPMENT WITH LOCATION FEATURES")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
        X, y, y_encoded, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Data Split:")
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Total features: {X.shape[1]}")
    
    # Define models
    models = {
        'kNN (k=5)': (KNeighborsClassifier(n_neighbors=5), False),
        'Decision Tree': (DecisionTreeClassifier(max_depth=10, random_state=42), False),
        'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1), False),
        'Neural Network': (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42, early_stopping=True, validation_fraction=0.1), False),
        'XGBoost': (xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                      random_state=42, eval_metric='mlogloss', n_jobs=-1, verbosity=0), True)
    }
    
    results = {}
    
    print(f"\n🤖 Training and Evaluating Models:")
    print("-" * 80)
    
    for name, (model, use_encoded) in models.items():
        print(f"\n{name}:")
        
        # Select appropriate target labels
        y_tr = y_train_enc if use_encoded else y_train
        y_te = y_test_enc if use_encoded else y_test
        
        # Create pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_tr, cv=cv, scoring='accuracy')
        print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train on full training set
        pipeline.fit(X_train, y_tr)
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        
        # Convert predictions back to string labels if using encoded
        if use_encoded:
            y_pred_labels = label_encoder.inverse_transform(y_pred)
            y_test_labels = label_encoder.inverse_transform(y_te)
        else:
            y_pred_labels = y_pred
            y_test_labels = y_te
        
        test_accuracy = accuracy_score(y_test_labels, y_pred_labels)
        print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Store results
        results[name] = {
            'pipeline': pipeline,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'y_pred': y_pred_labels
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels, 
                            labels=['Very Low', 'Low', 'High', 'Very High'])
        
        # Classification report
        report = classification_report(y_test_labels, y_pred_labels, 
                                      labels=['Very Low', 'Low', 'High', 'Very High'],
                                      output_dict=True)
        results[name]['classification_report'] = report
        results[name]['confusion_matrix'] = cm
    
    # Plot confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                      display_labels=['Very Low', 'Low', 'High', 'Very High'])
        disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
        axes[idx].set_title(f'{name}\nAccuracy: {result["test_accuracy"]:.4f}', 
                           fontsize=12, fontweight='bold')
    
    # Hide the last subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_v2.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrices saved as 'confusion_matrices_v2.png'")
    plt.close()
    
    return results, X_test, y_test

# ============================================================================
# Q5: PERFORMANCE ANALYSIS WITH LOCATION IMPACT
# ============================================================================

def analyze_performance(results, baseline_accuracy, feature_cols, X):
    """Analyze model performance and location feature impact"""
    print("\n" + "="*80)
    print("Q5: PERFORMANCE ANALYSIS & LOCATION FEATURE IMPACT")
    print("="*80)
    
    # Model comparison table
    print(f"\n📊 Model Comparison:")
    print("-" * 80)
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'CV Accuracy': f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}",
            'Test Accuracy': f"{result['test_accuracy']:.4f}",
            'vs Baseline': f"+{(result['test_accuracy'] - baseline_accuracy):.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Best model analysis
    best_model_name = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    best_result = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"  Test Accuracy: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
    print(f"  Improvement over baseline: +{(best_result['test_accuracy'] - baseline_accuracy)*100:.2f}%")
    
    # Class-specific performance
    print(f"\n📈 Class-Specific Performance ({best_model_name}):")
    print("-" * 80)
    
    report = best_result['classification_report']
    class_metrics = []
    for class_name in ['Very Low', 'Low', 'High', 'Very High']:
        if class_name in report:
            metrics = report[class_name]
            class_metrics.append({
                'Class': class_name,
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1-score']:.4f}",
                'Support': int(metrics['support'])
            })
    
    metrics_df = pd.DataFrame(class_metrics)
    print(metrics_df.to_string(index=False))
    
    # Feature importance analysis
    print(f"\n🔍 Feature Importance Analysis:")
    print("-" * 80)
    
    for name in ['Random Forest', 'XGBoost']:
        if name in results:
            pipeline = results[name]['pipeline']
            classifier = pipeline.named_steps['classifier']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Create feature importance dataframe
                feat_imp_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                print(f"\n{name} - Top 15 Features:")
                for i, row in feat_imp_df.head(15).iterrows():
                    print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")
                
                # Categorize features
                location_features = [f for f in feature_cols if any(x in f for x in ['region_', 'urban_', 'distance_to_cbd'])]
                traffic_features = [f for f in feature_cols if any(x in f for x in ['morning_rush', 'evening_rush', 'peak_hour'])]
                air_quality_features = [f for f in feature_cols if any(x in f for x in ['PM', 'NO', 'CO', 'AQI'])]
                weather_features = [f for f in feature_cols if any(x in f for x in ['rainfall', 'solar', 'temp'])]
                temporal_features = [f for f in feature_cols if any(x in f for x in ['month', 'day_of_week', 'year', 'weekend', 'holiday', 'season'])]
                
                # Calculate category importance
                location_imp = feat_imp_df[feat_imp_df['Feature'].isin(location_features)]['Importance'].sum()
                traffic_imp = feat_imp_df[feat_imp_df['Feature'].isin(traffic_features)]['Importance'].sum()
                air_quality_imp = feat_imp_df[feat_imp_df['Feature'].isin(air_quality_features)]['Importance'].sum()
                weather_imp = feat_imp_df[feat_imp_df['Feature'].isin(weather_features)]['Importance'].sum()
                temporal_imp = feat_imp_df[feat_imp_df['Feature'].isin(temporal_features)]['Importance'].sum()
                
                print(f"\n{name} - Feature Category Importance:")
                print(f"  Traffic Patterns:  {traffic_imp:.4f} ({traffic_imp*100:.1f}%)")
                print(f"  Location Features: {location_imp:.4f} ({location_imp*100:.1f}%)")
                print(f"  Air Quality:       {air_quality_imp:.4f} ({air_quality_imp*100:.1f}%)")
                print(f"  Weather:           {weather_imp:.4f} ({weather_imp*100:.1f}%)")
                print(f"  Temporal:          {temporal_imp:.4f} ({temporal_imp*100:.1f}%)")
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                top_features = feat_imp_df.head(20)
                sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')
                plt.title(f'{name} - Top 20 Feature Importances', fontsize=14, fontweight='bold')
                plt.xlabel('Importance', fontsize=12)
                plt.ylabel('Feature', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'feature_importance_{name.replace(" ", "_").lower()}_v2.png', dpi=300, bbox_inches='tight')
                print(f"\n✓ Feature importance plot saved")
                plt.close()
    
    print(f"\n✅ Analysis Complete!")
    print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # File path
    filepath = "datasets/TrafficWeather_Beuro_AQ_withSuburb/complete_traffic_environment_data.csv"
    
    print("\n" + "="*80)
    print("CLEARROADS V2: ENHANCED TRAFFIC PREDICTION WITH LOCATION FEATURES")
    print("="*80)
    print("\nStrategy 3: Hybrid Approach")
    print("  ✓ Regional grouping (rms_region)")
    print("  ✓ Distance to CBD (continuous)")
    print("  ✓ Urban classification (Urban/Suburban/Regional)")
    print("="*80)
    
    # Q1: Load and explore
    df = load_and_explore_data(filepath)
    
    # Q2: Clean and preprocess
    df_clean = clean_and_preprocess(df)
    
    # Q3: Create target classes
    df_target, baseline_accuracy = create_target_classes(df_clean)
    
    # Q4: Prepare features and train models
    X, y, y_encoded, feature_cols, label_encoder = prepare_features(df_target)
    print(f"\n📋 Selected Features ({len(feature_cols)}):")
    
    # Categorize and display features
    location_features = [f for f in feature_cols if any(x in f for x in ['region_', 'urban_', 'distance_to_cbd'])]
    traffic_features = [f for f in feature_cols if any(x in f for x in ['morning_rush', 'evening_rush', 'peak_hour'])]
    air_quality_features = [f for f in feature_cols if any(x in f for x in ['PM', 'NO', 'CO', 'AQI'])]
    weather_features = [f for f in feature_cols if any(x in f for x in ['rainfall', 'solar', 'temp'])]
    temporal_features = [f for f in feature_cols if any(x in f for x in ['month', 'day_of_week', 'year', 'weekend', 'holiday', 'season'])]
    
    print(f"  Location Features ({len(location_features)}): {', '.join(location_features)}")
    print(f"  Traffic Features ({len(traffic_features)}): {', '.join(traffic_features)}")
    print(f"  Air Quality Features ({len(air_quality_features)}): {', '.join(air_quality_features)}")
    print(f"  Weather Features ({len(weather_features)}): {', '.join(weather_features)}")
    print(f"  Temporal Features ({len(temporal_features)}): {', '.join(temporal_features)}")
    
    results, X_test, y_test = train_and_evaluate_models(X, y, y_encoded, label_encoder, baseline_accuracy)
    
    # Q5: Analyze performance
    analyze_performance(results, baseline_accuracy, feature_cols, X)
    
    print(f"\n🎉 All tasks completed successfully!")
    print(f"\n📁 Output files generated:")
    print(f"  - congestion_class_distribution_v2.png")
    print(f"  - confusion_matrices_v2.png")
    print(f"  - feature_importance_random_forest_v2.png")
    print(f"  - feature_importance_xgboost_v2.png")
