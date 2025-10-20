📑 **The University of Sydney**
**Faculty of Engineering**

---

# ClearRoads: Predicting Traffic Congestion Through Environmental Intelligence

**Authors**: Aryan Rai, Nixie Nassar, [Name 3], [Name 4]
**Course**: ENGG2112 Multidisciplinary Engineering | 2025 | USYD
**Date**: September 28, 2025

---

## 1. Title & Abstract

**Title**: ClearRoads: Predicting Traffic Congestion Through Environmental Intelligence

**Abstract**:
**Objectives**: This project aims to develop a machine learning model that predicts traffic congestion patterns across NSW road networks by integrating environmental factors including air quality (PM2.5, PM10, NO2, CO) and meteorological data (temperature, rainfall, solar radiation) with historical traffic volume data.

**Methods**: We will merge hourly traffic count data from 400+ NSW road monitoring stations with daily air quality measurements from 50+ monitoring sites and Bureau of Meteorology weather data. Multiple regression models (Random Forest, Neural Networks, XGBoost) will be trained to predict daily traffic volumes and congestion levels, with geospatial matching of environmental conditions to traffic stations.

**Novelty**: Unlike existing traffic prediction systems that rely primarily on historical traffic patterns and basic weather data, our approach incorporates comprehensive air quality metrics as predictive features, recognizing that poor air quality days often correlate with altered travel behaviors and traffic patterns.

**Additional dimensions**: The model addresses environmental justice by identifying how air quality impacts traffic patterns in different socioeconomic areas, potentially informing policy decisions about traffic management during pollution events and supporting public health initiatives.

**Deliverables**: A trained ML model capable of predicting traffic congestion levels, a comprehensive analysis of environmental-traffic correlations, and recommendations for traffic management strategies during adverse environmental conditions.

## 2. Justification of the Project

**Why is this project timely and important?**
With increasing urbanization and climate change, NSW faces growing challenges in traffic management and air quality control. Poor air quality events (bushfire smoke, industrial pollution) significantly impact travel patterns, yet current traffic prediction systems don't account for these environmental factors. This project addresses the critical need for intelligent traffic management that considers environmental health impacts.

**Who benefits and how?**
- **Transport authorities**: Better traffic flow prediction during environmental events
- **Public health agencies**: Understanding traffic-pollution feedback loops
- **Commuters**: Improved route planning during poor air quality days
- **Urban planners**: Data-driven insights for sustainable transport infrastructure

**What has been done before (state of the art)?**
Current traffic prediction models primarily use historical traffic data, basic weather (temperature, precipitation), and real-time traffic sensors. Research has explored weather impacts on traffic, but comprehensive integration of air quality data remains limited. Most existing systems treat environmental factors as secondary variables rather than primary predictors.

**What's new/innovative about this approach?**
Our approach uniquely integrates multi-pollutant air quality data (PM2.5, PM10, NO2, CO) as primary predictive features, recognizing that air quality significantly influences travel behavior. We'll use geospatial matching to link environmental conditions with specific road segments, creating a more granular prediction model than existing systems.

**Challenges and how we'll address them**:
- **Data integration complexity**: Different temporal resolutions (hourly traffic vs daily air quality) - addressed through temporal aggregation and interpolation
- **Spatial mismatch**: Air quality stations don't align with traffic counters - solved using nearest neighbor matching and spatial interpolation
- **Missing data**: Handled through multiple imputation techniques and robust model selection

## 3. Objectives

**Sub-problem A**: Data Integration and Preprocessing
Merge traffic count data from 400+ NSW road stations with air quality data from 50+ monitoring sites and meteorological data. This involves temporal alignment (hourly to daily aggregation), spatial matching (linking nearest environmental stations to traffic counters), and handling missing values across datasets spanning 2008-2025.

**Sub-problem B**: Feature Engineering and Environmental Impact Analysis
Develop meaningful features from environmental data including air quality indices, weather patterns, and seasonal variations. Analyze correlations between environmental conditions and traffic patterns to identify key predictive relationships and create composite environmental stress indicators.

**Sub-problem C**: Model Development and Validation
Train and compare multiple ML models (Random Forest, Neural Networks, XGBoost) to predict traffic congestion levels. Implement cross-validation strategies accounting for temporal dependencies and evaluate model performance across different environmental conditions and geographic regions.

**Overall Goal**: Create a comprehensive traffic prediction system that accurately forecasts congestion patterns by leveraging environmental intelligence, enabling proactive traffic management during adverse environmental conditions and supporting sustainable urban mobility planning.

## 4. Proposed Method of Solution & Timeline

**Technical Approach**:

*Data Preprocessing (Weeks 9)*:
- Merge traffic station reference data with hourly count data using station_key
- Spatially match traffic stations with nearest air quality monitoring sites using coordinates
- Aggregate hourly traffic data to daily totals and peak hour metrics
- Interpolate missing environmental data using temporal and spatial methods

*Feature Engineering (Week 10)*:
- Create composite air quality indices from PM2.5, PM10, NO2, CO measurements
- Generate weather-based features (temperature ranges, rainfall categories, solar radiation levels)
- Develop temporal features (day of week, seasonality, holiday indicators)
- Calculate rolling averages for environmental conditions (3-day, 7-day windows)

*Model Development (Weeks 11)*:
- Implement baseline linear regression model
- Train Random Forest for handling non-linear relationships and feature importance
- Develop Neural Network for complex pattern recognition
- Apply XGBoost for gradient boosting performance
- Use time-series cross-validation to prevent data leakage

*Evaluation and Refinement (Week 12)*:
- Compare models using metrics
- Analyze feature importance and environmental impact patterns
- Test model performance across different seasons and pollution events
- Validate predictions against held-out test data from recent months

**Metrics for Success**:
[To Be Filled]

**Novelty vs Prior Methods**:
Traditional models use basic weather data; our approach integrates comprehensive air quality metrics as primary predictors. We employ geospatial matching to create location-specific environmental profiles for each traffic station, enabling more precise predictions than region-wide environmental averages.

**Timeline**:
- **Week 9**: Data collection, cleaning, and integration
- **Week 10**: Feature engineering and exploratory data analysis  
- **Week 11**: Model development and training
- **Week 12**: Model evaluation and comparison
- **Week 13**: Final analysis, documentation, and presentation preparation

## 5. Ethical and Moral Considerations

**Safety Issues**:
Inaccurate traffic predictions during environmental emergencies (bushfires, pollution events) could lead to poor routing decisions, potentially endangering public health. False negatives might direct people through high-pollution areas, while false positives could cause unnecessary traffic diversions.

**Privacy Risks**:
While our data sources are aggregated and anonymized, the combination of traffic patterns with environmental data could potentially reveal sensitive information about community behaviors and health vulnerabilities in specific areas.

**Fairness/Bias Concerns**:
Air quality monitoring stations are not uniformly distributed across NSW, with potential underrepresentation in rural and lower socioeconomic areas. This could lead to biased predictions that favor well-monitored urban areas, potentially exacerbating existing transportation inequities.

**Legal/Moral Responsibility**:
If implemented in real traffic management systems, prediction errors could have significant consequences. Clear accountability frameworks must be established for AI-driven traffic management decisions, particularly during environmental health emergencies.

**Regulatory Needs**:
Integration with existing traffic management systems requires compliance with transport authority standards and environmental health regulations. The model should support, not replace, human decision-making in critical situations.

## 6. Data to Be Used

**Primary Datasets**:

1. **NSW Roads Traffic Volume Counts** (2011-2025)
   - Source: https://data.nsw.gov.au/data/dataset/2-nsw-roads-traffic-volume-counts-api
   - Content: Hourly traffic counts from 400+ permanent stations
   - Features: station_key, date, hourly volumes (00-23), daily totals, location data

2. **Road Traffic Station Reference Data**
   - Content: Geographic coordinates and metadata for traffic stations
   - Features: station_key, latitude, longitude, road classification, suburb, postcode

3. **NSW Air Quality Data** (2008-2025)
   - Source: https://www.airquality.nsw.gov.au/air-quality-data-services
   - Content: Daily averages for PM2.5, PM10, NO2, CO from 50+ monitoring sites
   - Coverage: Major urban centers and industrial areas across NSW

4. **Bureau of Meteorology Weather Data** (1862-2025)
   - Source: http://www.bom.gov.au/climate/data/
   - Content: Daily rainfall, temperature (min/max), solar radiation
   - Spatial coverage: Weather stations matched to traffic monitoring suburb

**Data Integration Strategy**:
- Geospatial matching of environmental monitoring stations to traffic counters using nearest neighbor analysis
- Temporal alignment through daily aggregation of hourly traffic data
- Multiple dataset validation through cross-referencing overlapping time periods
- Quality assessment using data completeness metrics and outlier detection

## 7. Team Composition & Organisation

**Member 1 (Aryan Rai)**: Project coordination, data integration, and geospatial analysis. Responsible for merging traffic and environmental datasets, implementing spatial matching algorithms, and ensuring data quality across all sources.

**Member 2**: Machine learning model development and feature engineering. Focus on implementing Random Forest, Neural Network, and XGBoost models, conducting hyperparameter optimization, and developing environmental feature sets.

**Member 3**: Data preprocessing and exploratory analysis. Handle missing data imputation, temporal aggregation, statistical analysis of traffic-environment correlations, and creation of visualization dashboards.

**Member 4**: Model evaluation, validation, and documentation. Implement cross-validation strategies, performance metrics calculation, result interpretation, and preparation of final report and presentations.

**Team Organization**:
- **Equal contribution** across coding (data processing, model implementation), analysis (statistical evaluation, pattern identification), and documentation (report writing, presentation preparation)
- **Weekly check-ins** with structured agenda covering progress updates, technical challenges, and next steps
- **Collaborative development** using version control for code sharing and documentation
- **Aryan Rai as team coordinator** for scheduling, milestone tracking, and external communication, ensuring balanced workload distribution

**Communication Strategy**:
- Regular team meetings for progress synchronization
- Shared documentation for methodology and findings
- Peer review process for code quality and analysis validation
- Clear task assignment with defined deliverables and deadlines

## References

[1] NSW Government. "NSW Roads Traffic Volume Counts API." Data.NSW, 2025. [Online]. Available: https://data.nsw.gov.au/data/dataset/2-nsw-roads-traffic-volume-counts-api

[2] NSW Environment Protection Authority. "Air Quality Data Services." NSW Air Quality, 2025. [Online]. Available: https://www.airquality.nsw.gov.au/air-quality-data-services/data-download-facility

[3] Bureau of Meteorology. "Climate Data Online." Australian Government, 2025. [Online]. Available: http://www.bom.gov.au/climate/data/

[4] L. Zhang, J. Liu, and M. Chen, "Impact of Air Quality on Urban Traffic Patterns: A Machine Learning Approach," *Transportation Research Part D*, vol. 89, pp. 102-115, 2024.

[5] K. Smith et al., "Environmental Factors in Traffic Flow Prediction: A Comprehensive Review," *IEEE Transactions on Intelligent Transportation Systems*, vol. 25, no. 3, pp. 1245-1260, 2024.

## Appendix

**Data Structure Summary**:

| Dataset | Temporal Resolution | Spatial Coverage | Key Features |
|---------|-------------------|------------------|--------------|
| Traffic Counts | Hourly (2011-2025) | 400+ stations NSW-wide | Hourly volumes, daily totals, station metadata |
| Air Quality | Daily (2008-2025) | 50+ urban/industrial sites | PM2.5, PM10, NO2, CO concentrations |
| Weather Data | Daily (1862-2025) | Bureau stations statewide | Temperature, rainfall, solar radiation |

**Feature Engineering Plan**:
- **Traffic Features**: Daily totals, peak hour ratios, weekend/weekday patterns
- **Environmental Features**: Air quality indices, pollution categories, weather composites
- **Temporal Features**: Seasonality, holidays, day-of-week effects
- **Spatial Features**: Road classification, urban/rural designation, proximity to pollution sources

---