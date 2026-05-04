# A Decision Support Model for Spatial and Temporal Analysis of Camping Sites in the Black Sea Region

## 1. Motivation and Purpose
The aim of this project is to analyze the 16 camping sites in the black sea region. The main goal is to build a machine learning algorithm which can determine in a specific date by analysing weather condition, transportation condition, human mobility.

## 2. Data Sources and Collection Methodology
Since there is no centralized data set which can contain all the specific parameters and information for analysing these 18 camp sites, the “Unified Data Set” will be achieved by collecting and cleaning the data set obtained by independent sources by the following pipeline:

* **Meteorological Data:** Data weather condition will be extracted from the meteostat Python library. The obtained data will be divided into weekly sections. Spatial interpolation will be used for high-altitude areas that lack direct measurement stations.
* **Digital footprint (Human Density):** for isolated camp sites there is no formal data about human density. Therefore to estimate the weekly digital footprint, the annual number of human visits data will be collected by web scraping and official formal information sources (news, journal etc.). After the annual data collection, Google comments will be analyzed and the human density will be estimated in weekly periods accordingly by comparing the google comment data and the data collected by web scraping and official formal information sources (news, journal etc.).
* **Accessibility to the healthcare and city site and road conditions:** to correctly measure the transportation condition and how much the camp site is isolated from the city, the nearest city site and healthcare center's distance and accessibility is estimated with google maps and web scraping. Slope data will be estimated accordingly with google earth data. Also the road surface data will be estimated from OpenStreetMap for each camp site.

## 3. Dataset Characteristics

* **Collection Period & Temporal Resolution:** The collected data will cover a historical period of 4 years (2022-2025). The collected data will be divided into weekly intervals and be used for the estimation of weather conditions. For the human density data all 2022, 2023, 2024 and 2025 Google comments data about camp sites will be analyzed to estimate the monthly human density.
* **Sample Size:** The Weather condition data will be the 16 camp sites' weather condition over a 4 years period (208 week) which will be approximately 3328 primary records. For the human density, Google comment data of the years 2022, 2023, 2024 and 2025 will be analysed, in total there is 52.719 google comments data.
* **Fixed Data:** The annual number of human visits data will be collected by web scraping and official formal information sources (news, journal etc.), the nearest city site and healthcare center's distance and accessibility is estimated with google maps and web scraping. Slope data will be estimated accordingly with google earth data. Also the road surface data will be estimated from OpenStreetMap for each camp site.
* **Key Variables and Expected Units:**
  * *Weather:* Temperature in Celsius (°C), Precipitation in millimeters (mm).
  * *Distance/Access:* Driving time in minutes.
  * *Terrain:* Slope in degrees, Soil composition in percentages (clay/sand).

## 4. Exploratory Data Analysis (EDA)
First, exploratory data analysis is performed on each of the 16 campsites: per-site weekly visitor time-series, dual-axis visitor-versus-temperature comparisons, full Pearson correlation heat-maps over all weather variables, and visitor-distribution histograms along with weather scatter plots. In this context, there are a total of 64 figures under EDA_Grafikleri/, plus clean weekly-visitor bar charts under EDA_GRAPHS_HUMAN_DENSITY/.

## 5. Hypothesis Tests

A set of five hypothesis tests was designed using non-parametric and randomization-based methods to account for the skewed distribution of visitor counts, non-normal data, and seasonal confounding (e.g., summer raising both temperature and visitors). Tests A, B, and C apply Benjamini-Hochberg FDR correction — a p-value adjustment method that controls the false-positive rate across multiple tests. P-values reported as < 0.0001 reflect the resolution limit of 10,000 permutations.

---

#### A. Season-Controlled Temperature Effect
> **Method:** Stratified Permutation Test with Spearman correlation (10,000 iterations)

* **$H_0$:** When season is held constant, temperature and visitors are unrelated.
* **$H_1$:** Temperature and visitors are related even when season is controlled.

Temperatures were permuted *within* each season group, isolating the genuine temperature effect from seasonal confounding.

**Result:** $H_0$ rejected in **16 of 16** sites after FDR correction. Spearman $r$ ranged from $0.46$ to $0.91$.

**Key Finding:** Temperature is a strong predictor independent of seasonal patterns, validating `temp` and `temp_kare` features in the ML model.

---

#### B. Precipitation Effect on Visitors
> **Method:** Mann-Whitney U Test with FDR correction

* **$H_0$:** Visitor distributions are equal between rainy and non-rainy weeks.
* **$H_1$:** The distributions differ.

The non-parametric alternative to the t-test, suitable for skewed data.

**Result:** $H_0$ rejected in **6 of 16** sites after FDR correction 

**Key Finding:** Precipitation sensitivity is location-dependent. In Eastern Black Sea plateaus, rain occurs in 190+/210 weeks, making the rainy/non-rainy distinction nearly meaningless. The effect is real in Western/Central sites where rain is a discrete event.

---

#### C. Seasonal Effect on Visitors
> **Method:** Kruskal-Wallis H Test with FDR correction

* **$H_0$:** Median visitor counts are equal across the four seasons.
* **$H_1$:** At least one season's median differs.

The non-parametric ANOVA alternative — necessary because variance is dramatically higher in summer.

**Result:** $H_0$ rejected in **16 of 16** sites. Global test: $H = 1033.73$, $p \approx 8.7 \times 10^{-224}$. Summer medians exceed other seasons by 5–30×.

**Key Finding:** Season is a near-universal predictor, validating `mevsim`, cyclical time encodings, and seasonal pattern features.

---

#### D. Region × Precipitation Sensitivity
> **Method:** Chi-Square Test of Independence

* **$H_0$:** Region and "low-visitor-on-rainy-week" status are independent.
* **$H_1$:** They are associated.

The unit of analysis was changed from location-level (n=4) to week-location records (n=3,360), satisfying the expected-frequency assumption (min expected = 375).

**Result:** $\chi^2 = 0.116$, $p = 0.944$, Cramér's $V = 0.0059$. $H_0$ accepted.

**Key Finding:** All three regions show ~44–45% "rainy weeks with below-median visitors" — region does not modulate precipitation sensitivity. A `Region × prcp` interaction feature is unnecessary in the ML model.

---

#### E. Statistical Significance of Weather Features in the ML Model
> **Method:** Permutation Test on model performance (×100 iterations)

* **$H_0$:** Weather features do not contribute to model accuracy.
* **$H_1$:** Weather features contribute significantly.

Weather columns were jointly permuted to break their relationship with visitors while preserving inter-correlations among weather variables themselves.

**Result:**
* Observed $R^2 = 0.194$ vs. mean null $R^2 = -0.178$
* $\Delta R^2 = +0.372$, $p < 0.01$ (0/100 null permutations exceeded observed)

**Key Finding:** Weather features make a statistically significant contribution, directly supporting the project's core premise. ($R^2 = 0.19$ here reflects a simplified RF without engineered features; the production model achieves substantially higher performance.)

---

### Summary

| Test | Method | Result |
| :--- | :--- | :--- |
| A. Season-controlled temperature | Stratified Permutation | 16/16 significant |
| B. Precipitation | Mann-Whitney U + FDR | 6/16 significant |
| C. Seasonal effect | Kruskal-Wallis + FDR | 16/16 significant |
| D. Region × precipitation | Chi-Square (n=3,360) | Not significant |
| E. ML weather contribution | Permutation on $R^2$ | $p < 0.01$ |

Temperature and season are universal drivers, precipitation contributes location-specific signal, and weather as a whole significantly improves predictive accuracy.


## 6. Machine Learning Model

The system uses a Random Forest Regressor trained on three years of historical data (2022–2024), validated on 2025, and then retrained on the full dataset (2022–2025) for production forecasting.

The core idea is simple but powerful: instead of predicting absolute visitor numbers directly, the model predicts relative demand in log-space — a ratio of how busy a location is compared to its own yearly average. This isolates seasonal patterns from location size, making the model far more robust across very different locations (e.g., a small plateau vs. a major national park).

### *Architecture*

The project uses a two-model design:

| Model | Training Data | Purpose |
| :--- | :--- | :--- |
| **Evaluation Model** (`rf_model`) | 2022–2024 | Reports honest performance metrics ($R^2$, MAE) on 2025. |
| **Production Model** (`rf_production`) | 2022–2025 | Generates forecasts for 2026+ using the freshest data. |

This separation is important: we measure performance with the evaluation model (so the 2025 test is genuinely unseen), but we forecast the future with the production model (so we don't waste a full year of valuable signal).

### *Feature Engineering*

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

### *Modeling Pipeline*

The training pipeline runs in five stages:

1.  **Data Cleaning:** Drops two leaky columns (`Ziyaretci_2025`, `uygulanan_beta`) and fills missing rating values using location-level means.
2.  **Feature Engineering:** Builds all five feature families described above.
3.  **Hyperparameter Search:** `RandomizedSearchCV` with `TimeSeriesSplit` (3 folds, 20 candidates = 60 fits) finds the best Random Forest configuration without breaking time order.
4.  **Feature Selection:** Features with importance below $0.005$ are dropped to reduce noise.
5.  **Final Training:** The model is fit with `oob_score=True` for an independent out-of-bag validation signal.

### *Forecasting Logic*

When predicting future visitor counts, the model combines two ingredients:

**growth_multiplier =**exp(growth_slope * (year - 2023.5))
**projected_yearly_avg =** yearly_avg_2022_2025 * growth_multiplier
**final_prediction =** predicted_relative_demand * projected_yearly_avg

**Safety Constraint:** A critical safety constraint caps the growth slope at ±15% per year in log-space. Without this cap, locations with a steep 3-year upward trend (like Abant, with slope $\approx 0.546$) would produce unrealistic 4× growth predictions for 2026. Tourism doesn't grow that fast, so we clip extrapolation to a physically plausible range.

### *Validation Strategy*

The model is validated with multiple independent signals:

*   **Hold-out test on 2025:** The evaluation model never sees 2025 data during training.
*   **Out-of-bag (OOB) score:** Random Forest's built-in cross-validation, computed for free during training.
*   **4-fold TimeSeriesSplit CV:** Confirms performance is stable across different train/test splits (Folds 2–4 are reported as stable; Fold 1 is excluded because it has too little training data).
*   **Per-location performance breakdown:** Locations are categorized into high ($R^2 \geq 0.80$), medium ($0.65 \leq R^2 < 0.80$), and low ($R^2 < 0.65$) performance tiers, helping identify where the model is most and least reliable.

### *Why This Design Works*

The model deliberately decouples two different things: **seasonal demand patterns** (handled by the Random Forest) and **long-term location growth** (handled by post-processing). 

This separation matters because seasonal patterns are highly predictable from history, but growth trends are noisy and need explicit safety bounds. By keeping them apart, the model stays accurate on stable locations and sensible on rapidly growing ones.

A final health check at the end of the pipeline scores the model on six dimensions (test $R^2$, overfitting gap, OOB score, CV consistency, leakage status, production readiness) and produces an overall score out of 100.


## 7. Academic Integrity & AI Disclosure

In accordance with the academic integrity guidelines of the DSA 210 course, I declare that AI tools (e.g., LLMs) were utilized to assist with the code generation, data processing, and text refinement stages of this project. 

All specific prompts used and the corresponding generated outputs (chat histories) have been fully saved and documented. They are readily available upon request if needed.
