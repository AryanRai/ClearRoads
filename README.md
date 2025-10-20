# ClearRoads: Predicting Traffic Congestion Through Environmental Intelligence

**ENGG2112 Multidisciplinary Engineering Project**  
**The University of Sydney | 2025**

## Team Members
- Aryan Rai [530362258] - Project coordination, data integration, geospatial analysis
- Nixie Nassar - Evaluation, validation, and documentation
- Nell Nesci [540701579] - Data preprocessing and exploratory analysis
- [Name 4] - TBD

## Project Overview

ClearRoads develops machine learning models to predict traffic congestion in NSW by integrating air quality data (PM2.5, PM10, NO₂, CO) with historical traffic volumes. Unlike existing systems that rely primarily on historical traffic patterns, our approach incorporates comprehensive air quality metrics as primary predictive features.

### Key Innovation
We recognize that air quality and traffic form a feedback loop: traffic worsens pollution, while pollution and weather alter traffic flows. By incorporating multi-pollutant air quality data as primary predictors, we enable more accurate congestion forecasting during environmental stress events.

## Repository Structure

```
ProjectProposal/
├── traffic_analysis.py          # Main analysis script
├── proposal.md                   # Project proposal
├── Plan.md                       # Detailed analysis plan
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── QUICKSTART.md                 # Quick start guide
├── README_ANALYSIS.md            # Detailed analysis documentation
├── IMPLEMENTATION_SUMMARY.md     # Implementation details
├── FIX_SUMMARY.md               # Technical fixes applied
├── Docs/                        # Documentation
│   ├── sample_proposal1.md
│   ├── report_guidelines.md
│   └── dataset.md
└── datasets/                    # Data files (not tracked in git)
    ├── TrafficWeatherwithSuburb/
    ├── Weather_AQ/
    └── Traffic_TimesOfDay/
```

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ProjectProposal
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Analysis
```bash
python traffic_analysis.py
```

## Dataset

**Merged Traffic-Air Quality Dataset**
- **Records**: 71,038 observations
- **Time Period**: 2008-2025
- **Features**: 
  - Traffic: Daily totals, hourly counts (24 hours)
  - Air Quality: PM10, PM2.5, NO2, NO, CO
  - Temporal: Date, year, month, day of week, holidays
  - Location: Suburb, station identifiers

**Data Sources**:
1. NSW Roads Traffic Volume Counts (2011-2025)
2. NSW EPA Air Quality Data (2008-2025)
3. Bureau of Meteorology Weather Data (planned integration)

## Methodology

### Target Variable
4-class traffic congestion classification based on daily traffic volume:
- **Very Low**: < 25th percentile (< 533 vehicles)
- **Low**: 25th-50th percentile (533-1,443 vehicles)
- **High**: 50th-75th percentile (1,443-6,154 vehicles)
- **Very High**: ≥ 75th percentile (≥ 6,154 vehicles)

### Machine Learning Models
1. **k-Nearest Neighbors (k=5)** - Simple baseline
2. **Decision Tree** - Interpretable model
3. **Random Forest** - Robust ensemble method
4. **Neural Network (MLP)** - Non-linear pattern recognition
5. **XGBoost** - High-performance gradient boosting

### Evaluation
- 5-fold Stratified Cross-Validation
- 80/20 train-test split
- Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Results

### Model Performance
- **Baseline Accuracy**: 25.07% (majority class)
- **Best Model**: Random Forest with **96.35% test accuracy**
- **Improvement**: +71.28% over baseline

### Key Findings
- Air quality features successfully predict traffic congestion
- Random Forest and Neural Network models achieve >95% accuracy
- Feature importance analysis reveals which pollutants most influence traffic patterns

## Outputs

The analysis generates:
1. **Console Output**: Detailed statistics and metrics
2. **congestion_class_distribution.png**: Class distribution visualization
3. **confusion_matrices.png**: Model performance comparison

## Project Timeline

- **Week 9**: Data integration ✅
- **Week 10**: Feature engineering ✅
- **Week 11**: Model development ✅
- **Week 12**: Evaluation & comparison (in progress)
- **Week 13**: Final analysis & presentation

## Dependencies

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- scikit-learn >= 1.2.0
- xgboost >= 1.7.0

## Documentation

- **QUICKSTART.md**: 3-step guide to run the analysis
- **README_ANALYSIS.md**: Comprehensive analysis documentation
- **IMPLEMENTATION_SUMMARY.md**: Technical implementation details
- **Plan.md**: Detailed analysis plan aligned with project requirements

## Ethical Considerations

- **Safety**: Accurate predictions critical during environmental emergencies
- **Privacy**: Aggregated data protects individual privacy
- **Fairness**: Addressing spatial bias in monitoring station distribution
- **Transparency**: Interpretable models support decision-making

## Future Enhancements

1. Integration of BoM weather data (rainfall, temperature, solar radiation)
2. Hyperparameter optimization for improved performance
3. Temporal validation (train on historical, test on recent data)
4. Spatial analysis by region and suburb
5. Real-time prediction system deployment

## References

1. NSW Government. "NSW Roads Traffic Volume Counts API." Data.NSW, 2025.
2. NSW EPA. "Air Quality Data Services." NSW Air Quality, 2025.
3. Bureau of Meteorology. "Climate Data Online." Australian Government, 2025.
4. Infrastructure Australia. "Urban Transport Crowding and Congestion." 2025.

## License

This project is developed for academic purposes as part of ENGG2112 at The University of Sydney.

## Contact

For questions or collaboration:
- Aryan Rai: [530362258]
- Course: ENGG2112 Multidisciplinary Engineering
- Institution: The University of Sydney

---

**Last Updated**: October 2025
