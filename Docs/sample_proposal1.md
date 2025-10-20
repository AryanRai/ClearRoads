**The University of Sydney**
**Faculty of Engineering**

---

# Learning to Power Sydney – ML Driven Analysis of the Power Grid

**Authors**:
George Ayad
William Kerin
Samuel Pitchforth
Jorge Lara Mino

**SIDs**:
530575836
540808144
540714557
530635347

**Course**: ENGG2112 Multidisciplinary Engineering
**Date**: September 25, 2025

---

## Contents

1. Abstract
2. Justifications
3. Objectives
4. Proposed Method of Solution
5. Ethical and Moral Concerns
6. Data Source
7. Team Structure
8. References
9. Appendix

---

## 1. Abstract

This proposal details the design of a machine learning-based model to predict power demand at substations across the Sydney and Hunter Valley regions. The model will predict power consumption at individual substations based on time of day, date, and weather data.

With the rise of climate change, New South Wales (NSW) faces increasingly severe weather events such as heatwaves and floods, while also transitioning to distributed renewable energy sources. These factors increase the difficulty of predicting substation power demand. This model will allow power utilities to actively predict demand under different conditions and, in turn, plan maintenance and upgrades to existing substations.

---

## 2. Justifications

Due to climate change, NSW’s electricity infrastructure is increasingly vulnerable to stressors such as heatwaves, storms, and flooding. This proposal introduces a machine learning (ML) model designed to forecast power demand across substations with the aim of predicting where upgrades to substation capacity are necessary.

By incorporating weather data, the model adapts to changing climate impacts on the electrical system. It can also identify low-demand periods, enabling scheduled maintenance with minimal disruption.

Societal benefits include:

* Prioritised infrastructure upgrades.
* Reduced unnecessary capital expenditure.
* Improved energy security at lower cost.

Traditional methods rely on static forecasting, whereas this ML model can scale efficiently from a single Local Government Area (LGA) to statewide deployment without extensive rework.

---

## 3. Objectives

1. Predict instantaneous power consumption (regression model) at substations in Sydney and the Hunter Valley, based on features in **Table 2**.

   a. Acquire data ✔
   b. Process data:

   * Transform categorical time data into discrete numerical values.
   * Apply one-hot encoding to categorical features.
   * Handle missing values and normalise data.
     c. Construct multiple regression models.
     d. Select the best-performing model using metrics defined in Section 4.

---

## 4. Proposed Method of Solution

1. **Data Collection (19/04):**
   Gather datasets on weather and power usage. Document sources, collection frequency, and data quality. *(Completed)*

2. **Pre-processing (01/05):**
   Merge power usage and meteorological datasets using tools such as *pandas*. Handle missing values, normalise features, and perform one-hot encoding. Apply KNN clustering for categorical transformation (e.g., rainfall levels).

3. **Exploratory Data Analysis (05/05):**

   * *Univariate:* 5-number summaries, histograms, and density plots.
   * *Multivariate:* Scatter plots and correlation matrices.

4. **Model Development (12/05):**
   Train models listed in **Table 1** with varied hyperparameters.

   **Table 1: Comparison of ML Models**

   | Model         | Strengths                                            | Weaknesses                                                   |
   | ------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
   | kNN           | Simple to implement, effective for localised trends. | Computationally expensive, sensitive to irrelevant features. |
   | Neural Net    | Captures complex non-linear relationships.           | Black box, prone to overfitting.                             |
   | Decision Tree | Easy to interpret, highlights impactful features.    | Slow, overfits with complex data.                            |
   | Random Forest | Handles missing data well, reduces overfitting.      | Computationally intensive, less interpretable.               |

5. **Model Evaluation and Validation (17/05):**
   Establish baseline with linear regression. Discard underperforming models. Apply k=10 cross-validation and generate confusion matrices.

6. **Model Comparison and Selection (18/05):**
   Select final model using classification metrics with emphasis on accuracy, sensitivity, and interpretability.

---

## 5. Ethical and Moral Concerns

The primary concern is reinforcing existing inequalities. Historic data reflects past allocation decisions, often disadvantaging rural or low socio-economic areas. Bias in training data risks amplifying these disparities.

Key risks include:

* **Bias reinforcement:** Under-served regions receive less attention.
* **Data quality issues:** Rural data may reduce performance.
* **Failure modes:**

  * *False positives* → unnecessary expenditure.
  * *False negatives* → prolonged outages, economic loss, safety risks.

Sensitivity must be prioritised to reduce negative outcomes. Context-specific weighting (e.g., prioritising hospitals over industrial zones) should guide deployment.

---

## 6. Data Source

* **Ausgrid Data [2]:** Substation power consumption at 15-minute intervals (2005–2024).
* **Bureau of Meteorology [3]:** Weather data including daily max/min temperature, rainfall, and solar radiation.

---

## 7. Team Structure

1. **George Ayad:** Team organisation, accountability, research into existing predictive models.
2. **William Kerin:** Data collation and preprocessing.
3. **Jorge Lara:** Report writing and formatting.
4. **Samuel Pitchforth:** Model implementation and performance analysis.

---

## 8. References

[1] Y. Han, H. Jia, C. Xu, M. Bockarjova, C. van Westen, and L. Lombardo, *Unveiling spatial inequalities: Exploring county-level disaster damages and social vulnerability on public disaster assistance in contiguous US,* Journal of Environmental Management, vol. 351, p. 119690, 2024. DOI: [https://doi.org/10.1016/j.jenvman.2023.119690](https://doi.org/10.1016/j.jenvman.2023.119690)

[2] Ausgrid. “Ausgrid Distribution Zone Substation Data.” Accessed: Sept. 25, 2025. [https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Distribution-zone-substation-data](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Distribution-zone-substation-data)

[3] Bureau of Meteorology. “Climate Data Online.” Accessed: Sept. 25, 2025. [http://www.bom.gov.au/climate/data/](http://www.bom.gov.au/climate/data/)

---

## 9. Appendix

**Table 2: Features for Substation Power Usage Prediction**

| Feature Name             | Data Type          | Details             |
| ------------------------ | ------------------ | ------------------- |
| Substation Name          | Boolean collection | One-hot encoded     |
| Time of Day              | Integer (0–95)     | 15-minute intervals |
| Day of Week              | Integer (0–6)      | Monday–Sunday       |
| Season                   | Integer (0–3)      | Summer–Spring       |
| Daily Rainfall           | Integer (0–3)      | None → Heavy        |
| Total Daily Solar Energy | Float              | MJ/m²               |
| Minimum Daily Temp.      | Float              | °C                  |
| Maximum Daily Temp.      | Float              | °C                  |
| Public Holiday           | Boolean (0–1)      | Yes/No              |

---
