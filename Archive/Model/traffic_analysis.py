"""
ClearRoads: Traffic Congestion Prediction using Air Quality Data
ENGG2112 - Multidisciplinary Engineering Project
Authors: Aryan Rai, Nixie Nassar, Nell Nesci
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
# Q1: IMPORT AND DESCRIPTIVES
# ============================================================================

def load_and_explore_data(filepath):
    """Load dataset and perform initial exploration"""
    print("="*80)
    print("Q1: IMPORT AND DESCRIPTIVES")
    print("="*80)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\n✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Drop unnecessary columns
    cols_to_drop = ['the_geom', 'the_geom_webmercator', 'cartodb_id', 
                    'record_id', 'md5', 'updated_on', 'suburb']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    print(f"✓ Dropped {len([c for c in cols_to_drop if c in df.columns])} unnecessary columns")
    
    # Show column list
    print(f"\n📋 Final Column List ({len(df.columns)} columns):")
    print("-" * 80)
    
    # Categorize columns
    traffic_cols = ['daily_total'] + [f'hour_{i:02d}' for i in range(24)]
    air_quality_cols = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO']
    temporal_cols = ['date', 'year', 'month', 'day_of_week', 'public_holiday', 'school_holiday']
    categorical_cols = ['station_key', 'traffic_direction_seq', 'cardinal_direction_seq', 
                       'classification_seq', 'suburb_std']
    
    print(f"Traffic Features ({len([c for c in traffic_cols if c in df.columns])}): {', '.join([c for c in traffic_cols if c in df.columns][:5])}...")
    print(f"Air Quality Features ({len([c for c in air_quality_cols if c in df.columns])}): {', '.join([c for c in air_quality_cols if c in df.columns])}")
    print(f"Temporal Features ({len([c for c in temporal_cols if c in df.columns])}): {', '.join([c for c in temporal_cols if c in df.columns])}")
    print(f"Categorical Features ({len([c for c in categorical_cols if c in df.columns])}): {', '.join([c for c in categorical_cols if c in df.columns])}")
    
    # Descriptive statistics
    print("\n📊 Descriptive Statistics:")
    print("-" * 80)
    
    # Traffic statistics
    print("\n🚗 Traffic Volume Statistics:")
    print(df['daily_total'].describe())
    
    # Air quality statistics
    print("\n🌫️ Air Quality Statistics:")
    print(df[air_quality_cols].describe())
    
    # Missing values analysis
    print("\n❓ Missing Values Analysis:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing[missing > 0],
        'Percentage': missing_pct[missing > 0]
    }).sort_values('Percentage', ascending=False)
    print(missing_df)
    
    return df

# ============================================================================
# Q2: DATA CLEANING & PREPROCESSING
# ============================================================================

def clean_and_preprocess(df):
    """Clean and preprocess the dataset"""
    print("\n" + "="*80)
    print("Q2: DATA CLEANING & PREPROCESSING")
    print("="*80)
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # Convert date to datetime
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    # Handle boolean columns
    bool_cols = ['public_holiday', 'school_holiday']
    for col in bool_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    
    print("\n🧹 Data Cleaning Steps:")
    print("-" * 80)
    
    # Step 1: Remove rows where ALL air quality features are missing
    air_quality_cols = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO']
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=air_quality_cols, how='all')
    print(f"✓ Removed {before - len(df_clean):,} rows with all air quality values missing")
    
    # Step 2: Remove rows with missing daily_total
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=['daily_total'])
    print(f"✓ Removed {before - len(df_clean):,} rows with missing daily_total")
    
    # Step 3: Impute remaining air quality missing values with suburb-specific medians
    for col in air_quality_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean.groupby('suburb_std')[col].transform(
                lambda x: x.fillna(x.median())
            )
    print(f"✓ Imputed remaining air quality missing values with suburb-specific medians")
    
    # Step 4: Feature Engineering
    print("\n🔧 Feature Engineering:")
    print("-" * 80)
    
    # Temporal features
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([6, 7]).astype(int)
    df_clean['season'] = df_clean['month'].map({
        12: 'Summer', 1: 'Summer', 2: 'Summer',
        3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
        6: 'Winter', 7: 'Winter', 8: 'Winter',
        9: 'Spring', 10: 'Spring', 11: 'Spring'
    })
    print(f"✓ Created temporal features: is_weekend, season")
    
    # Composite Air Quality Index (simplified)
    # Normalize each pollutant and create weighted average
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
    
    # Remove outliers in daily_total
    q1 = df_clean['daily_total'].quantile(0.01)
    q99 = df_clean['daily_total'].quantile(0.99)
    before = len(df_clean)
    df_clean = df_clean[(df_clean['daily_total'] >= q1) & (df_clean['daily_total'] <= q99)]
    print(f"✓ Removed {before - len(df_clean):,} outliers (< 1st percentile or > 99th percentile)")
    
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
    plt.savefig('congestion_class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved as 'congestion_class_distribution.png'")
    plt.close()
    
    return df_target, baseline_accuracy

# ============================================================================
# Q4: MODEL DEVELOPMENT
# ============================================================================

def prepare_features(df):
    """Prepare features for modeling"""
    # Select features
    air_quality_features = ['PM10', 'PM2_5', 'NO2', 'NO', 'CO', 'AQI_composite']
    temporal_features = ['month', 'day_of_week', 'public_holiday', 'school_holiday', 
                        'is_weekend', 'year']
    traffic_features = ['morning_rush', 'evening_rush', 'peak_hour_traffic']
    
    # Combine all features
    feature_cols = []
    for col in air_quality_features + temporal_features + traffic_features:
        if col in df.columns:
            feature_cols.append(col)
    
    # Encode season
    if 'season' in df.columns:
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        feature_cols.extend(season_dummies.columns.tolist())
    
    X = df[feature_cols].copy()
    y = df['Congestion_Class'].copy()
    
    # Encode target labels for XGBoost compatibility
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y, y_encoded, feature_cols, label_encoder

def train_and_evaluate_models(X, y, y_encoded, label_encoder, baseline_accuracy):
    """Train and evaluate multiple models"""
    print("\n" + "="*80)
    print("Q4: MODEL DEVELOPMENT")
    print("="*80)
    
    # Split data - use string labels for most models, encoded for XGBoost
    X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
        X, y, y_encoded, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Data Split:")
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # Define models
    models = {
        'kNN (k=5)': (KNeighborsClassifier(n_neighbors=5), False),
        'Decision Tree': (DecisionTreeClassifier(max_depth=10, random_state=42), False),
        'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42), False),
        'Neural Network': (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42), False),
        'XGBoost': (xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                      random_state=42, eval_metric='mlogloss'), True)
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
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
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
    
    # Hide the last subplot if odd number of models
    if len(results) < 6:
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrices saved as 'confusion_matrices.png'")
    plt.close()
    
    return results, X_test, y_test

# ============================================================================
# Q5: PERFORMANCE ANALYSIS
# ============================================================================

def analyze_performance(results, baseline_accuracy, feature_cols):
    """Analyze model performance and environmental impact"""
    print("\n" + "="*80)
    print("Q5: PERFORMANCE ANALYSIS & ENVIRONMENTAL IMPACT")
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
    
    # Class-specific performance for best model
    best_model_name = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    best_result = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"  Test Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"  Improvement over baseline: +{(best_result['test_accuracy'] - baseline_accuracy)*100:.2f}%")
    
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
    
    # Feature importance for tree-based models
    print(f"\n🔍 Feature Importance Analysis:")
    print("-" * 80)
    
    for name in ['Random Forest', 'XGBoost']:
        if name in results:
            pipeline = results[name]['pipeline']
            classifier = pipeline.named_steps['classifier']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Get top 10 features
                indices = np.argsort(importances)[::-1][:10]
                
                print(f"\n{name} - Top 10 Features:")
                for i, idx in enumerate(indices, 1):
                    print(f"  {i}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    print(f"\n✅ Analysis Complete!")
    print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # File path
    filepath = "datasets/TrafficWeatherwithSuburb/roadandweathermerged-20251020T083319Z-1-001/roadandweathermerged/output_merge.csv"
    
    # Q1: Load and explore
    df = load_and_explore_data(filepath)
    
    # Q2: Clean and preprocess
    df_clean = clean_and_preprocess(df)
    
    # Q3: Create target classes
    df_target, baseline_accuracy = create_target_classes(df_clean)
    
    # Q4: Prepare features and train models
    X, y, y_encoded, feature_cols, label_encoder = prepare_features(df_target)
    print(f"\n📋 Selected Features ({len(feature_cols)}):")
    print(f"  {', '.join(feature_cols)}")
    
    results, X_test, y_test = train_and_evaluate_models(X, y, y_encoded, label_encoder, baseline_accuracy)
    
    # Q5: Analyze performance
    analyze_performance(results, baseline_accuracy, feature_cols)
    
    print(f"\n🎉 All tasks completed successfully!")
