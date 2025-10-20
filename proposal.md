# ClearRoads: Predicting Traffic Congestion Through Environmental Intelligence

**Faculty of Engineering**  
**The University of Sydney**

**Authors:** Aryan Rai [530362258], Nixie Nassar, Nell Nesci [540701579], [Name 4]  
**Course:** ENGG2112 Multidisciplinary Engineering | 2025 | USYD  
**Date:** September 28, 2025

---

## 1. Abstract

**Objectives:**  
This project develops a machine learning (ML) model to predict traffic congestion in NSW by integrating air quality (PM2.5, PM10, NO₂, CO) and meteorological data (temperature, rainfall, solar radiation) with historical traffic volumes.

**Methods:**  
Hourly traffic count data will be merged with daily environmental data from NSW monitoring sites and the Bureau of Meteorology. Multiple ML models will be explored and compared, with geospatial matching between environmental stations and traffic counters.

**Novelty:**  
Unlike existing systems that rely on historical traffic and basic weather data, our approach incorporates multi-pollutant air quality as primary predictive features, reflecting how pollution both results from and influences traffic patterns.

**Deliverables:**
- A trained ML model predicting congestion under environmental stressors.  
- Analysis of environmental–traffic correlations.  
- Policy-oriented recommendations for traffic management during pollution events.

---

## 2. Justification of the Project

**Timeliness:**  
NSW faces increasing congestion and poor air quality due to urban growth and climate change. Environmental factors such as NO₂, CO, and rainfall form a feedback loop: traffic worsens pollution, while pollution and weather alter traffic flows.  
Current prediction systems treat these as secondary, creating a research gap.  
In assisting with congestion management, the model can help reduce crashes, vehicle wear, emissions, and economic loss — estimated at **$19 billion in 2016** [6].

**Beneficiaries:**
- **Transport authorities:** Forecast congestion during environmental stress.  
- **Public health agencies:** Understand traffic–pollution feedback cycles.  
- **Commuters:** Safer route planning under poor air conditions.  
- **Urban planners:** Evidence-based planning for sustainable mobility.

**State of the Art:**  
Existing models mainly use historical traffic and simple weather data. Research on weather’s impact is more common than studies on air quality’s effect on traffic. Few approaches close the loop between traffic and environment.

**Challenges & Mitigation:**
- **Differing temporal resolutions** (hourly traffic vs daily air quality) → temporal aggregation & interpolation.  
- **Spatial mismatch** (air stations ≠ traffic counters) → nearest neighbour matching.  
- **Missing data** → multiple imputation, mean filling, or NA removal; robust model selection.

---

## 3. Objectives

- **Sub-problem A – Data Integration:** Merge traffic, air quality, and BoM weather data (2008–2025).  
- **Sub-problem B – Feature Engineering:** Create composite pollution scores, weather features, and stress indicators.  
- **Sub-problem C – Model Development:** Explore and validate ML models under different conditions.  

**Overall Goal:**  
Deliver a traffic prediction system that integrates environmental intelligence for proactive congestion management.

---

## 4. Proposed Method & Timeline

**Approach:**
1. **Preprocessing (Week 9):** Aggregate hourly → daily traffic; spatial matching; interpolate missing data.  
2. **Feature Engineering (Week 10):** Build composite air indices; generate weather features; perform exploratory analysis.  
3. **Model Development (Week 11):** Test kNN, Neural Networks, Decision Trees, Random Forests, and XGBoost.  
4. **Evaluation & Refinement (Week 12):** Compare models, analyze feature importance, validate seasonally and during pollution events.

### Model Comparison

| Model | Strengths | Weaknesses |
|--------|------------|-------------|
| **kNN** | Simple; effective for local similarities. | Expensive on large data; sensitive to irrelevant features. |
| **Neural Network** | Captures complex non-linear relationships. | Black box; overfits without large data. |
| **Decision Tree** | Interpretable; highlights key features. | Prone to instability and overfitting. |
| **Random Forest** | Robust; handles missing data; reduces overfitting. | Computationally intensive; less interpretable. |
| **XGBoost** | High accuracy; efficient for structured data. | Requires careful tuning; less transparent. |

**Timeline:**
- **Week 9:** Data integration  
- **Week 10:** Feature engineering  
- **Week 11:** Model development  
- **Week 12:** Evaluation & comparison  
- **Week 13:** Final analysis & presentation  

---

## 5. Ethical & Moral Considerations

- **Safety:** Inaccurate predictions during emergencies (e.g., bushfires) may cause unsafe routing.  
- **Privacy:** While anonymized, combined datasets may reveal community behavior.  
- **Fairness:** Sparse monitoring in rural areas may bias predictions toward urban centers.  
- **Responsibility:** Clear accountability needed during deployment.  
- **Regulation:** Must comply with transport and environmental standards.  

---

## 6. Data to Be Used

- **Traffic Data:** NSW Roads Volume Counts (2011–2025), hourly → daily totals.  
- **Air Quality Data:** NSW EPA (2008–2025) — PM2.5, PM10, NO₂, CO.  
- **Weather Data:** BoM (1862–2025) — rainfall, temperature, solar radiation.  

**Integration Strategy:**  
Geospatial matching of stations, daily temporal alignment, validation with overlapping datasets, and quality checks for completeness/outliers.

---

## 7. Team Composition & Organisation

- **Member 2 (Aryan Rai):** Project coordination, data integration, geospatial analysis. Handles dataset merging, spatial matching algorithms, and data quality assurance.  
- **Member 2 (Aryan Rai):** Machine learning model development and feature engineering. Implements Random Forest, Neural Network, and XGBoost models; performs hyperparameter optimization.  
- **Member 3 (Nell Nesci):** Data preprocessing and exploratory analysis — imputation, temporal aggregation, correlation analysis, and visualization dashboards.  
- **Member 4 (Nixie Nassar):** Evaluation, validation, metrics, interpretation, and documentation.  

**Collaboration:**  
Weekly check-ins, shared documentation, peer review, version control, and structured workload balancing.

---

## Appendix

### Primary Datasets

#### NSW Roads Traffic Volume Counts (2011–2025)
- **Source:** [NSW Data Portal](https://data.nsw.gov.au/data/dataset/2-nsw-roads-traffic-volume-counts-api)  
- **Content:** Hourly traffic counts from multiple permanent stations  
- **Features:** `station_key`, `date`, hourly volumes, daily totals, location data  

#### Road Traffic Station Reference Data
- **Content:** Geographic coordinates and metadata for traffic stations  
- **Features:** `station_key`, latitude, longitude, road classification, suburb, postcode  

#### NSW Air Quality Data (2008–2025)
- **Source:** [NSW Air Quality Data Services](https://www.airquality.nsw.gov.au/air-quality-data-services)  
- **Content:** Daily averages for PM2.5, PM10, NO₂, CO  
- **Coverage:** Major urban centers and industrial areas  

#### Bureau of Meteorology Weather Data (1862–2025)
- **Source:** [BoM Climate Data Online](http://www.bom.gov.au/climate/data/)  
- **Content:** Daily rainfall, temperature (min/max), solar radiation  
- **Coverage:** Statewide weather stations matched to traffic monitoring suburbs  

---

## References

1. NSW Government. *"NSW Roads Traffic Volume Counts API."* Data.NSW, 2025. [Online]. Available: [https://data.nsw.gov.au/data/dataset/2-nsw-roads-traffic-volume-counts-api](https://data.nsw.gov.au/data/dataset/2-nsw-roads-traffic-volume-counts-api)  
2. NSW Environment Protection Authority. *"Air Quality Data Services."* NSW Air Quality, 2025. [Online]. Available: [https://www.airquality.nsw.gov.au/air-quality-data-services/data-download-facility](https://www.airquality.nsw.gov.au/air-quality-data-services/data-download-facility)  
3. Bureau of Meteorology. *"Climate Data Online."* Australian Government, 2025. [Online]. Available: [http://www.bom.gov.au/climate/data/](http://www.bom.gov.au/climate/data/)  
4. L. Zhang, J. Liu, and M. Chen, *"Impact of Air Quality on Urban Traffic Patterns: A Machine Learning Approach,"* Transportation Research Part D, vol. 89, pp. 102–115, 2024.  
5. K. Smith et al., *"Environmental Factors in Traffic Flow Prediction: A Comprehensive Review,"* IEEE Trans. Intelligent Transportation Systems, vol. 25, no. 3, pp. 1245–1260, 2024.  
6. Infrastructure Australia. *“Urban Transport Crowding and Congestion.”* Australian Government, 2025. [Online]. Available: [https://www.infrastructureaustralia.gov.au/sites/default/files/2019-08/Urban%20Transport%20Crowding%20and%20Congestion.pdf](https://www.infrastructureaustralia.gov.au/sites/default/files/2019-08/Urban%20Transport%20Crowding%20and%20Congestion.pdf)

---

## Data Structure Summary

| Dataset | Temporal Resolution | Spatial Coverage | Key Features |
|----------|--------------------|------------------|---------------|
| **Traffic Counts** | Hourly (2011–2025) | Multiple stations NSW-wide | Hourly volumes, daily totals, metadata |
| **Air Quality** | Daily (2008–2025) | Urban & industrial sites | PM2.5, PM10, NO₂, CO |
| **Weather Data** | Daily (1862–2025) | Statewide | Temperature, rainfall, solar radiation |
