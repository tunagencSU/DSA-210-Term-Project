# A Decision Support Model for Spatial and Temporal Analysis of Camping Sites in the Black Sea Region

## 1. Motivation and Purpose
The aim of this project is to analyze the 16 camping sites in the black sea region. The main goal is to build a machine learning algorithm which can determine in a specific date by analysing weather condition, transportation condition, human mobility.

## 2. Data Sources and Collection Methodology
Since there is no centralized data set which can contain all the specific parameters and information for analysing these 18 camp sites, the “Unified Data Set” will be achieved by collecting and cleaning the data set obtained by independent sources by the following pipeline:

* **Meteorological Data:** Data weather condition will be extracted from the meteostat Python library. The obtained data will be divided into weekly sections. Spatial interpolation will be used for high-altitude areas that lack direct measurement stations.
* **Digital footprint (Human Density):** for isolated camp sites there is no formal data about human density. Therefore to estimate the weekly digital footprint, the annual number of human visits data will be collected by web scraping and offical formal information sources (news, journal etc.). Afther the annual data collection, Google comments will be analyzed and the human density will be estimated in weekly periods accordingly by comparing the google comment data and the data collected by web scraping and offical formal information sources (news, journal etc.).
* **Accessibility to the healthcare and city site and road conditions:** to correctly mesure the transportation condition and how much the camp site is isolated from the city, the nearest city site and healthcare center's distance and accessibility is estimated with google maps and web scraping. Slope data will be estimated accordingly with google earth data. Also the road surface data will be estimated from OpenStreetMap for each camp site.

## 3. Dataset Characteristics

* **Collection Period & Temporal Resolution:** The collected data will cover a historical period of 5 years (2021-2025). The collected data will be divided into weekly intervals and be used for the estimation of weather conditions. For the human densitiy data all 2022, 2023, 2024 and 2025 Google comments data about camp cites will be analyzed to estimate the monthly human density.
* **Sample Size:** The Weather condition data will be the 16 camp sites' weather condition over a 5 years period (208 week) which will be approximately 3328 primary records. For the human density, Google comment data ıf the years 2022, 2023, 2024 and 2025 will be analysed, in total there is 52.719 google comments data.
* **Fixed Data:** The annual number of human visits data will be collected by web scraping and offical formal information sources (news, journal etc.), the nearest city site and healthcare center's distance and accessibility is estimated with google maps and web scraping. Slope data will be estimated accordingly with google earth data. Also the road surface data will be estimated from OpenStreetMap for each camp site.
* **Key Variables and Expected Units:**
  * *Weather:* Temperature in Celsius (°C), Precipitation in millimeters (mm).
  * *Distance/Access:* Driving time in minutes.
  * *Terrain:* Slope in degrees, Soil composition in percentages (clay/sand).

## 4. Exploratory Data Analysis (EDA)
First, exploratory data analysis is performed on each of the 16 campsites: per-site weekly visitor time-series, dual-axis visitor-versus-temperature comparisons, full Pearson correlation heat-maps over all weather variables, and visitor-distribution histograms along with weather scatter plots. In this context, there are a total of 64 figures under EDA_Grafikleri/, plus clean weekly-visitor bar charts under EDA_GRAPHS_HUMAN_DENSITY/.

## 5. Hypothesis Tests
Second, three statistical hypothesis tests quantify what actually drives weekly visitor numbers: a Pearson correlation between temperature and visitors (significant in 17 of 18 sites), a Welch's t-test comparing rainy and dry weeks (significant in 7 of 18 sites), and a Chi-square test of independence between visitor-volume class and precipitation sensitivity (no association, p = 1.0). The combined result is that temperature is a near-universal driver, while precipitation matters only at highland and outdoor-activity sites whose attraction is the activity itself.

#### A. Relationship Between Temperature and Visitor Numbers 
> **Method:** Pearson Correlation Test <br>
> **Objective:** To determine whether air temperature significantly affects the number of visitors to the camping sites.

* **$H_0$ (Null Hypothesis):** There is no statistically significant relationship between temperature and the number of visitors.
* **$H_1$ (Alternative Hypothesis):** There is a statistically significant relationship between temperature and the number of visitors.

**Result and Interpretation:**
The $H_0$ hypothesis was rejected in **17** of the 18 camping sites analyzed ($p < 0.05$). The relationship was found to be insignificant in only 1 location.

**Key Finding:** Temperature is the strongest and most universal meteorological factor determining visitor density in Black Sea camping sites. As the weather warms up, the demand for camping sites increases in a statistically clear and measurable way. Temperature serves as the primary predictive variable for our modeling.

---

#### B. Relationship Between Precipitation and Visitor Numbers
> **Method:** Independent Samples Welch's T-Test <br>
> **Objective:** To determine if there is a significant difference in the number of visitors between rainy and non-rainy weeks.

* **$H_0$:** The visitor averages for rainy and non-rainy weeks are equal (Precipitation has no effect).
* **$H_1$:** There is a significant difference in the visitor averages between rainy and non-rainy weeks.

**Result and Interpretation:**
The $H_0$ hypothesis was rejected in **7** of the 18 camping sites analyzed (e.g., *Ayder Yaylası, Borçka Karagöl, Elevit Yaylası*), proving that precipitation has a significant effect on the number of visitors. In the remaining **11 camping sites**, no statistically significant difference was found.

**Key Finding:** Unlike temperature, precipitation does not affect every camping site equally. While some camping sites show high "sensitivity" to precipitation and lose visitors, others maintain their visitor base even in rainy weather. This variance likely depends on external factors such as physical infrastructure (availability of indoor areas), transportation difficulty, or visitor profile (adventurous vs. day-tripper).

---

#### C. Visitor Volume and Precipitation Sensitivity
> **Method:** Chi-Square Test of Independence <br>
> **Objective:** To examine whether we can make a general deduction such as *"camping sites that are visited more frequently (popular) are affected more/less by precipitation."*

* **$H_0$:** The visitor volume category (Above/Below Median) and precipitation sensitivity are independent of each other (No relationship).
* **$H_1$:** There is a significant relationship between the visitor category and precipitation sensitivity.

**Result and Interpretation:**
* **Test Statistic:** $\chi^2 = 0.0000$
* **P-Value:** $1.0000$

Since the calculated p-value is greater than $0.05$, **the $H_0$ hypothesis is accepted.**

**Key Finding:** Whether a camping site is popular (crowded) or quiet on an annual basis is not a factor that affects its visitor loss on rainy days (precipitation sensitivity). Both crowded and lesser-known quiet camping sites are affected by precipitation in entirely independent ways. 

## 6. Machine Learning Model

The system uses a Random Forest Regressor trained on three years of historical data (2022–2024), validated on 2025, and then retrained on the full dataset (2022–2025) for production forecasting.

The core idea is simple but powerful: instead of predicting absolute visitor numbers directly, the model predicts relative demand in log-space — a ratio of how busy a location is compared to its own yearly average. This isolates seasonal patterns from location size, making the model far more robust across very different locations (e.g., a small plateau vs. a major national park).

#### **Architecture*

The project uses a two-model design:

| Model | Training Data | Purpose |
| :--- | :--- | :--- |
| **Evaluation Model** (`rf_model`) | 2022–2024 | Reports honest performance metrics ($R^2$, MAE) on 2025. |
| **Production Model** (`rf_production`) | 2022–2025 | Generates forecasts for 2026+ using the freshest data. |

This separation is important: we measure performance with the evaluation model (so the 2025 test is genuinely unseen), but we forecast the future with the production model (so we don't waste a full year of valuable signal).

#### **Feature Engineering*

The model relies on five carefully designed feature families, all built to be leakage-free (no future information leaks into past predictions).

### A. Seasonal Pattern Features (The Strongest Signal)
*   `ayni_hafta_gecmis_ort`: Historical mean for the same week-of-year at this location.
*   `ayni_hafta_gecmis_std`: Historical standard deviation (an uncertainty signal).
*   `son_yil_ayni_hafta`: Last year's value for the same week (the freshest seasonal signal).
*   `komsu_hafta_ort`: Average of ±1 neighboring weeks (smoothing).

> **Note:** All of these use a strict "past only" logic. Predicting week 30 of 2025 only uses week 30 from 2022, 2023, and 2024 — never the target week itself.

### B. Weather and Interaction Features
*   **Base:** Temperature, precipitation, snow, rain, wind speed, humidity.
*   **Engineered:** `temp_kare` (squared temp), `sicak_yagmur` (warm rain), `kar_soguk` (cold snow), `sicaklik_konfor` (comfort score), `nem_sicaklik` (humidity × temp), `kotu_hava` (rain × wind).

### C. Time Features
*   Cyclical encoding (`ay_sin`/`cos`, `hafta_sin`/`cos`) so the model understands that December and January are neighbors.
*   Season, holiday flag, and holiday × season interactions.

### D. Location Growth Dynamics
*   `buyume_egimi`: A 3-year log-linear growth slope per location.
*   `lokasyon_olcek`: Location size baseline.
*   `buyume_x_trend`: Interaction term.

### E. Lagged Review Features
*   4-week and 8-week rolling averages of review counts, shifted by 1 week to prevent leakage.

#### **Modeling Pipeline*

The training pipeline runs in five stages:

1.  **Data Cleaning:** Drops two leaky columns (`Ziyaretci_2025`, `uygulanan_beta`) and fills missing rating values using location-level means.
2.  **Feature Engineering:** Builds all five feature families described above.
3.  **Hyperparameter Search:** `RandomizedSearchCV` with `TimeSeriesSplit` (3 folds, 20 candidates = 60 fits) finds the best Random Forest configuration without breaking time order.
4.  **Feature Selection:** Features with importance below $0.005$ are dropped to reduce noise.
5.  **Final Training:** The model is fit with `oob_score=True` for an independent out-of-bag validation signal.

#### **Forecasting Logic*

When predicting future visitor counts, the model combines two ingredients:

**growth_multiplier =**exp(growth_slope * (year - 2023.5))
**projected_yearly_avg =** yearly_avg_2022_2025 * growth_multiplier
**final_prediction =** predicted_relative_demand * projected_yearly_avg

**Safety Constraint:** A critical safety constraint caps the growth slope at ±15% per year in log-space. Without this cap, locations with a steep 3-year upward trend (like Abant, with slope $\approx 0.546$) would produce unrealistic 4× growth predictions for 2026. Tourism doesn't grow that fast, so we clip extrapolation to a physically plausible range.

#### **Validation Strategy*

The model is validated with multiple independent signals:

*   **Hold-out test on 2025:** The evaluation model never sees 2025 data during training.
*   **Out-of-bag (OOB) score:** Random Forest's built-in cross-validation, computed for free during training.
*   **4-fold TimeSeriesSplit CV:** Confirms performance is stable across different train/test splits (Folds 2–4 are reported as stable; Fold 1 is excluded because it has too little training data).
*   **Per-location performance breakdown:** Locations are categorized into high ($R^2 \geq 0.80$), medium ($0.65 \leq R^2 < 0.80$), and low ($R^2 < 0.65$) performance tiers, helping identify where the model is most and least reliable.

#### **Why This Design Works*

The model deliberately decouples two different things: **seasonal demand patterns** (handled by the Random Forest) and **long-term location growth** (handled by post-processing). 

This separation matters because seasonal patterns are highly predictable from history, but growth trends are noisy and need explicit safety bounds. By keeping them apart, the model stays accurate on stable locations and sensible on rapidly growing ones.

A final health check at the end of the pipeline scores the model on six dimensions (test $R^2$, overfitting gap, OOB score, CV consistency, leakage status, production readiness) and produces an overall score out of 100.


## 6. Academic Integrity & AI Disclosure

In accordance with the academic integrity guidelines of the DSA 210 course, I declare that AI tools (e.g., LLMs) were utilized to assist with the code generation, data processing, and text refinement stages of this project. 

All specific prompts used and the corresponding generated outputs (chat histories) have been fully saved and documented. They are readily available upon request if needed.
