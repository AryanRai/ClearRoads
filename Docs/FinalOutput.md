(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal> python traffic_analysis_v2.py 2>&1

================================================================================
CLEARROADS V2: ENHANCED TRAFFIC PREDICTION WITH LOCATION FEATURES
================================================================================

Strategy 3: Hybrid Approach
  ✓ Regional grouping (rms_region)
  ✓ Distance to CBD (continuous)
  ✓ Urban classification (Urban/Suburban/Regional)
================================================================================
================================================================================
Q1: IMPORT AND DESCRIPTIVES - ENHANCED WITH LOCATION FEATURES
================================================================================

⏳ Loading dataset (this may take a moment)...

✓ Dataset loaded: 3,925,503 rows × 60 columns
✓ Dropped 0 unnecessary columns

📋 Final Column List (52 columns):
--------------------------------------------------------------------------------
Traffic Features: 25
Air Quality Features: 5
Weather Features: 4
Location Features: 5
Temporal Features: 6

📊 Data Completeness:
--------------------------------------------------------------------------------

Environmental Features:
  PM10                :    200,420 (  5.1%)
  PM2_5               :    147,746 (  3.8%)
  NO2                 :     73,938 (  1.9%)
  NO                  :    167,205 (  4.3%)
  CO                  :     73,938 (  1.9%)
  rainfall_mm         :    125,640 (  3.2%)
  solar_exposure_mj   :    129,276 (  3.3%)
  min_temp_c          :    126,361 (  3.2%)
  max_temp_c          :    120,701 (  3.1%)

Records with ANY environmental data: 333,795 (8.5%)

Location Features:

Regional Distribution:
  Sydney         :  1,986,860 ( 50.6%)
  Southern       :    456,424 ( 11.6%)
  Hunter         :    403,830 ( 10.3%)
  Western        :    307,238 (  7.8%)
  Northern       :    195,619 (  5.0%)
  South West     :    157,556 (  4.0%)

Distance to CBD Statistics:
count    3.508459e+06
mean     1.161617e+02
std      1.601413e+02
min      6.380696e-01
25%      1.574023e+01
50%      3.203948e+01
75%      1.435067e+02
max      9.284024e+02
Name: distance_to_cbd_km, dtype: float64

================================================================================
Q2: DATA CLEANING & PREPROCESSING WITH LOCATION FEATURES
================================================================================

🧹 Data Cleaning Steps:
--------------------------------------------------------------------------------
✓ Filtered to records with environmental data: 333,795 (8.5%)
✓ Removed 0 rows with missing daily_total
✓ Removed 0 rows with missing location data

🔧 Feature Engineering:
--------------------------------------------------------------------------------
✓ Imputed environmental features with suburb-specific medians
✓ Created temporal features: is_weekend, season
✓ Created composite Air Quality Index (AQI)
✓ Created traffic pattern features: morning_rush, evening_rush, peak_hour_traffic

🗺️ Location Features (Hybrid Approach):
--------------------------------------------------------------------------------
✓ Created urban_type classification
  Distribution: {'Suburban': 259744, 'Regional_City': 37119, 'Urban': 36932}
✓ Using distance_to_cbd_km (already in dataset)
✓ Using rms_region (already in dataset)

✓ Removed 6,668 outliers (< 1st or > 99th percentile)

📈 Final dataset: 327,127 rows (8.3% of original)

================================================================================
Q3: CREATE TARGET - TRAFFIC CONGESTION CLASS
================================================================================

📊 Traffic Volume Percentiles:
  25th percentile: 1,334
  50th percentile: 8,473
  75th percentile: 21,639

🎯 Congestion Class Distribution:
--------------------------------------------------------------------------------
                  Count  Percentage
Congestion_Class
High              81782        25.0
Low               81781        25.0
Very High         81785        25.0
Very Low          81779        25.0

📌 Majority Class Baseline:
  Majority class: Very High
  Baseline accuracy: 0.2500 (25.00%)

✓ Plot saved as 'congestion_class_distribution_v2.png'

📋 Selected Features (31):
  Location Features (8): distance_to_cbd_km, urban_Regional_City, urban_Suburban, urban_Urban, region_Hunter, region_Northern, region_Southern, region_Sydney
  Traffic Features (3): morning_rush, evening_rush, peak_hour_traffic
  Air Quality Features (6): PM10, PM2_5, NO2, NO, CO, AQI_composite
  Weather Features (4): rainfall_mm, solar_exposure_mj, min_temp_c, max_temp_c
  Temporal Features (10): month, day_of_week, public_holiday, school_holiday, is_weekend, year, season_Autumn, season_Spring, season_Summer, season_Winter

================================================================================
Q4: MODEL DEVELOPMENT WITH LOCATION FEATURES
================================================================================

📊 Data Split:
  Training set: 261,701 samples
  Test set: 65,426 samples
  Total features: 31

🤖 Training and Evaluating Models:
--------------------------------------------------------------------------------

kNN (k=5):
  5-Fold CV Accuracy: 0.8497 ± 0.0015
  Test Accuracy: 0.8713 (87.13%)

Decision Tree:
  5-Fold CV Accuracy: 0.9714 ± 0.0009
  Test Accuracy: 0.9717 (97.17%)

Random Forest:
  5-Fold CV Accuracy: 0.9809 ± 0.0005
  Test Accuracy: 0.9809 (98.09%)

Neural Network:
  5-Fold CV Accuracy: 0.9750 ± 0.0013
  Test Accuracy: 0.9791 (97.91%)

XGBoost:
  5-Fold CV Accuracy: 0.9826 ± 0.0004
  Test Accuracy: 0.9830 (98.30%)

✓ Confusion matrices saved as 'confusion_matrices_v2.png'

================================================================================
Q5: PERFORMANCE ANALYSIS & LOCATION FEATURE IMPACT
================================================================================

📊 Model Comparison:
--------------------------------------------------------------------------------
         Model     CV Accuracy Test Accuracy vs Baseline
     kNN (k=5) 0.8497 ± 0.0015        0.8713     +0.6213
 Decision Tree 0.9714 ± 0.0009        0.9717     +0.7217
 Random Forest 0.9809 ± 0.0005        0.9809     +0.7309
Neural Network 0.9750 ± 0.0013        0.9791     +0.7291
       XGBoost 0.9826 ± 0.0004        0.9830     +0.7330

🏆 Best Model: XGBoost
  Test Accuracy: 0.9830 (98.30%)
  Improvement over baseline: +73.30%

📈 Class-Specific Performance (XGBoost):
--------------------------------------------------------------------------------
    Class Precision Recall F1-Score  Support
 Very Low    0.9924 0.9922   0.9923    16356
      Low    0.9800 0.9782   0.9791    16356
     High    0.9749 0.9726   0.9737    16357
Very High    0.9849 0.9892   0.9870    16357

🔍 Feature Importance Analysis:
--------------------------------------------------------------------------------

Random Forest - Top 15 Features:
  peak_hour_traffic             : 0.3229
  evening_rush                  : 0.2622
  morning_rush                  : 0.1940
  distance_to_cbd_km            : 0.0771
  year                          : 0.0286
  region_Sydney                 : 0.0221
  urban_Regional_City           : 0.0103
  day_of_week                   : 0.0080
  region_Hunter                 : 0.0075
  region_Southern               : 0.0070
  urban_Urban                   : 0.0064
  NO                            : 0.0058
  max_temp_c                    : 0.0055
  CO                            : 0.0053
  urban_Suburban                : 0.0049

Random Forest - Feature Category Importance:
  Traffic Patterns:  0.7791 (77.9%)
  Location Features: 0.1361 (13.6%)
  Air Quality:       0.0277 (2.8%)
  Weather:           0.0116 (1.2%)
  Temporal:          0.0454 (4.5%)

✓ Feature importance plot saved

XGBoost - Top 15 Features:
  peak_hour_traffic             : 0.6141
  evening_rush                  : 0.2216
  urban_Suburban                : 0.0239
  morning_rush                  : 0.0228
  distance_to_cbd_km            : 0.0210
  day_of_week                   : 0.0197
  urban_Urban                   : 0.0112
  region_Southern               : 0.0088
  urban_Regional_City           : 0.0064
  season_Summer                 : 0.0062
  public_holiday                : 0.0059
  year                          : 0.0058
  region_Hunter                 : 0.0044
  month                         : 0.0041
  season_Winter                 : 0.0035

XGBoost - Feature Category Importance:
  Traffic Patterns:  0.8585 (85.8%)
  Location Features: 0.0757 (7.6%)
  Air Quality:       0.0088 (0.9%)
  Weather:           0.0078 (0.8%)
  Temporal:          0.0491 (4.9%)

✓ Feature importance plot saved

✅ Analysis Complete!
================================================================================

🎉 All tasks completed successfully!

📁 Output files generated:
  - congestion_class_distribution_v2.png
  - confusion_matrices_v2.png
📁 Output files generated:
  - congestion_class_distribution_v2.png
  - confusion_matrices_v2.png
  - feature_importance_random_forest_v2.png
  - feature_importance_xgboost_v2.png
(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal>
(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal>
  - feature_importance_random_forest_v2.png
  - feature_importance_random_forest_v2.png
  - feature_importance_xgboost_v2.png
(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal>

(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal> python generate_report_visualizations.py
================================================================================
GENERATING ADDITIONAL REPORT VISUALIZATIONS
================================================================================

⏳ Loading and preparing data...
✓ Data prepared: 327,127 records

📊 Creating regional traffic analysis...
✓ Saved: report_traffic_patterns_analysis.png

🌫️ Creating environmental correlation analysis...
✓ Saved: report_environmental_correlations.png

🤖 Creating model comparison visualization...
✓ Saved: report_model_comparison.png

🗺️ Creating location feature impact analysis...
  Training model for feature analysis...
✓ Saved: report_location_feature_impact.png

📋 Creating performance metrics summary...
✓ Saved: report_performance_metrics_table.png

✅ All visualizations generated successfully!

📁 Generated files:
  1. report_traffic_patterns_analysis.png
  2. report_environmental_correlations.png
  3. report_model_comparison.png
  4. report_location_feature_impact.png
  5. report_performance_metrics_table.png
================================================================================
(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal> dir report_*.png, *_v2.png | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
  5. report_performance_metrics_table.png
================================================================================
(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal> dir report_*.png, *_v2.png | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
 Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize

Name                                    Length LastWriteTime
----                                    ------ -------------
report_environmental_correlations.png   387090 28-10-2025 03:16:35
report_location_feature_impact.png      221976 28-10-2025 03:16:58
report_model_comparison.png             189474 28-10-2025 03:16:36
report_performance_metrics_table.png    155190 28-10-2025 03:16:59
report_traffic_patterns_analysis.png    384573 28-10-2025 03:16:32
confusion_matrices_v2.png               534509 28-10-2025 03:05:40
congestion_class_distribution_v2.png    122402 28-10-2025 02:40:43
feature_importance_random_forest_v2.png 192247 28-10-2025 03:05:41
feature_importance_xgboost_v2.png       198345 28-10-2025 03:05:42


(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal> ^C
(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal> Write-Output "=== CLEARROADS PROJECT COMPLETE ===" "" " RESULTS:" "- Best Model: XGBoost" "- Accuracy: 98.30%" "- Stability: 0.04%" "- Improvement: +73.30% over baseline" "" " FILES GENERATED:" "- 9 visualization PNG files" "- 3 Python scripts" "- 5 documentation files" "" " READY FOR REPORT!" "" "See Docs/REPORT_SECTIONS_READY_TO_USE.md for copy-paste sections" "See Docs/QUICK_REFERENCE.md for quick stats"
=== CLEARROADS PROJECT COMPLETE ===

 RESULTS:
- Best Model: XGBoost
- Accuracy: 98.30%
- Stability: 0.04%
- Improvement: +73.30% over baseline

 FILES GENERATED:
- 9 visualization PNG files
 FILES GENERATED:
- 9 visualization PNG files
- 3 Python scripts
- 5 documentation files

 READY FOR REPORT!

See Docs/REPORT_SECTIONS_READY_TO_USE.md for copy-paste sections
See Docs/QUICK_REFERENCE.md for quick stats
(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal>
 FILES GENERATED:
- 9 visualization PNG files
- 3 Python scripts
- 5 documentation files
- 9 visualization PNG files
- 3 Python scripts
- 5 documentation files

 READY FOR REPORT!

See Docs/REPORT_SECTIONS_READY_TO_USE.md for copy-paste sections
See Docs/QUICK_REFERENCE.md for quick stats
(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal>








See Docs/QUICK_REFERENCE.md for quick stats
(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal>



See Docs/QUICK_REFERENCE.md for quick stats
(ENGG2112) PS C:\Users\buzza\Desktop\Uni\ENGG2112\ProjectProposal>
See Docs/QUICK_REFERENCE.md for quick stats
See Docs/QUICK_REFERENCE.md for quick stats