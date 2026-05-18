# A Decision Support Model for Spatial and Temporal Analysis of Camping Sites in the Black Sea Region

**DSA 210 — Introduction to Data Science | Spring 2025–2026 Term Project**

---

## 1. Motivation

The Black Sea region is a great place with lots of amazing camping sites. There are dozens of places to visit. However, we don't have any data or information even for the most popular ones. My goal in this project is to choose 16 popular camping areas in the Black Sea region and collect data about them. I want to understand their overall popularity, which campsites become popular in which seasons, and how crowded they get on a weekly and yearly basis. With this information, I want to create a helpful guide that people can rely on when planning their trips.

---

## 2. Data Sources and Collection Methodology
Since there is no centralized data set which can contain all the specific parameters and information for analysing these 18 camp sites, the “Unified Data Set” will be achieved by collecting and cleaning the data set obtained by independent sources by the following pipeline:

* **Meteorological Data:** Data weather condition will be extracted from the meteostat Python library. The obtained data will be divided into weekly sections. Spatial interpolation will be used for high-altitude areas that lack direct measurement stations.
* **Digital footprint (Human Density):** for isolated camp sites there is no formal data about human density. Therefore to estimate the weekly digital footprint, the annual number of human visits data will be collected by web scraping and official formal information sources (news, journal etc.). After the annual data collection, Google comments will be analyzed and the human density will be estimated in weekly periods accordingly by comparing the google comment data and the data collected by web scraping and official formal information sources (news, journal etc.).
* **Accessibility to the healthcare and city site and road conditions:** to correctly measure the transportation condition and how much the camp site is isolated from the city, the nearest city site and healthcare center's distance and accessibility is estimated with google maps and web scraping. Slope data will be estimated accordingly with google earth data. Also the road surface data will be estimated from OpenStreetMap for each camp site.

### Dataset Characteristics

* **Collection Period & Temporal Resolution:** The collected data will cover a historical period of 4 years (2022-2025). The collected data will be divided into weekly intervals and be used for the estimation of weather conditions. For the human density data all 2022, 2023, 2024 and 2025 Google comments data about camp sites will be analyzed to estimate the monthly human density.
* **Sample Size:** The Weather condition data will be the 16 camp sites' weather condition over a 4 years period (208 week) which will be approximately 3328 primary records. For the human density, Google comment data of the years 2022, 2023, 2024 and 2025 will be analysed, in total there is 52.719 google comments data.
* **Fixed Data:** The annual number of human visits data will be collected by web scraping and official formal information sources (news, journal etc.), the nearest city site and healthcare center's distance and accessibility is estimated with google maps and web scraping. Slope data will be estimated accordingly with google earth data. Also the road surface data will be estimated from OpenStreetMap for each camp site.
* **Key Variables and Expected Units:**
  * *Weather:* Temperature in Celsius (°C), Precipitation in millimeters (mm).
  * *Distance/Access:* Driving time in minutes.
  * *Terrain:* Slope in degrees, Soil composition in percentages (clay/sand).
  
## 3. Exploratory Data Analysis (EDA)
First, exploratory data analysis is performed on each of the 16 campsites: per-site weekly visitor time-series, dual-axis visitor-versus-temperature comparisons, full Pearson correlation heat-maps over all weather variables, and visitor-distribution histograms along with weather scatter plots. In this context, there are a total of 64 figures under EDA_Grafikleri/, plus clean weekly-visitor bar charts under EDA_GRAPHS_HUMAN_DENSITY/.

## 4. Hypothesis Tests

Five hypothesis tests were designed to validate modeling assumptions before feature engineering. The tests use **non-parametric and randomization-based methods** because the visitor distribution is heavily skewed and confounded by season (summer simultaneously raises both temperature and visitors). Tests A, B, and C apply **Benjamini–Hochberg FDR correction** for multiple testing.

#### A. Season-Controlled Temperature Effect
> **Method:** Stratified Permutation Test with Spearman correlation (10,000 iterations)

* $H_0$: When season is held constant, temperature and visitors are unrelated.
* $H_1$: Temperature and visitors are related even when season is controlled.

Temperatures were permuted *within each season group*, isolating the genuine temperature effect from seasonal confounding.

**Result:** $H_0$ rejected in **16 / 16** sites after FDR correction. Spearman $r$ ranged from **0.46 to 0.91**.

**Key Finding:** Temperature is a strong predictor independent of seasonal patterns. This validates `temp` and `temp_kare` as core features.

#### B. Precipitation Effect on Visitors
> **Method:** Mann–Whitney U Test with FDR correction

* $H_0$: Visitor distributions are equal between rainy and non-rainy weeks.

**Result:** $H_0$ rejected in **6 / 16** sites after FDR correction.

**Key Finding:** Precipitation sensitivity is **location-dependent**. In Eastern Black Sea plateaus, it rains in 190+/210 weeks, making the rainy/non-rainy distinction nearly meaningless. The effect is real in Western/Central sites where rain is a discrete event. This justifies including precipitation features but expecting their importance to be modest.

#### C. Seasonal Effect on Visitors
> **Method:** Kruskal–Wallis H Test with FDR correction

* $H_0$: Median visitor counts are equal across the four seasons.

**Result:** $H_0$ rejected in **16 / 16** sites. Global test: $H = 1033.73$, $p \approx 8.7 \times 10^{-224}$. Summer medians exceed other seasons by **5–30×**.

**Key Finding:** Season is a near-universal predictor — validates `mevsim`, cyclical time encoding (`ay_sin/cos`, `hafta_sin/cos`), and seasonal pattern features.

#### D. Region × Precipitation Sensitivity
> **Method:** Chi-Square Test of Independence

* $H_0$: Region and "low-visitor-on-rainy-week" status are independent.

The unit of analysis was changed from location-level (n = 4) to week × location records (n = 3,360) to satisfy the expected-frequency assumption.

**Result:** $\chi^2 = 0.116$, $p = 0.944$, Cramér's $V = 0.0059$. $H_0$ **accepted**.

**Key Finding:** Region does **not** modulate precipitation sensitivity. A `Region × prcp` interaction term is unnecessary in the ML model — a useful design decision.

#### E. Statistical Significance of Weather Features in the ML Model
> **Method:** Permutation Test on model performance (100 iterations)

* $H_0$: Weather features do not contribute to model accuracy.

Weather columns were jointly permuted to break their relationship with visitors while preserving inter-correlations among the weather variables themselves.

**Result:**
* Observed $R^2 = 0.194$ vs. mean null $R^2 = -0.178$
* $\Delta R^2 = +0.372$, $p < 0.01$ (0/100 null permutations exceeded observed)

**Key Finding:** Weather features contribute significantly — directly supporting the project's core premise.

### Summary

| Test | Method | Result |
|---|---|---|
| A. Season-controlled temperature | Stratified Permutation | **16 / 16 significant** |
| B. Precipitation | Mann–Whitney U + FDR | 6 / 16 significant |
| C. Seasonal effect | Kruskal–Wallis + FDR | **16 / 16 significant** |
| D. Region × precipitation | Chi-Square (n = 3,360) | Not significant |
| E. ML weather contribution | Permutation on $R^2$ | $p < 0.01$ |

Temperature and season are universal drivers; precipitation contributes location-specific signal; weather as a whole significantly improves predictive accuracy. The model architecture in Section 5 is designed around these confirmed signals.

---

## 5. Machine Learning Model

### Architecture

The system uses a **Random Forest Regressor** with a two-model design:

| Model | Training Data | Purpose |
|---|---|---|
| **Evaluation Model** (`rf_model`) | 2022 – 2024 (1,680 rows) | Reports honest performance metrics ($R^2$, MAE) on held-out 2025 |
| **Production Model** (`rf_production`) | 2022 – 2025 (2,528 rows) | Generates forecasts for 2026+ using the freshest data |

The key design insight: instead of predicting absolute visitor numbers directly, the model predicts **relative demand in log-space** — a ratio of how busy a location is compared to its own annual average. This decouples seasonal patterns (which generalize across locations) from location size (which doesn't). The absolute forecast is then reconstructed by post-multiplying with a growth-projected annual mean.

### Feature Engineering (48 → 17 selected features)

**A. Seasonal Pattern Features (the strongest signal)**
* `ayni_hafta_gecmis_ort`: Historical mean of relative demand for the same week-of-year at this location.
* `ayni_hafta_gecmis_std`: Standard deviation of the same week across past years (uncertainty signal).
* `son_yil_ayni_hafta`: Last year's relative demand for the same week (freshest seasonal signal).
* `komsu_hafta_ort`: Average of ±1 neighboring weeks (smoothing).

> **Leakage prevention**: All four features use strict "past-only" logic. Predicting week 30 of 2025 only uses week 30 from 2022, 2023, and 2024 — never the target week itself.

**B. Weather and Interaction Features**
* Base: `temp`, `prcp`, `snow`, `rain`, `wspd`, `rhum`
* Engineered: `temp_kare` (squared temp), `sicak_yagmur` (warm rain), `kar_soguk` (cold snow), `sicaklik_konfor` (-((temp - 20)²), comfort score), `nem_sicaklik` (humidity × temp), `kotu_hava` (rain × wind)

**C. Time Features**
* Cyclical encoding: `ay_sin/cos`, `hafta_sin/cos` (so the model understands December and January are neighbors).
* `mevsim`, `tatil_mi` (holiday flag), `tatil_yaz`, `tatil_ilkbahar` (holiday × season interactions).

**D. Location Growth Dynamics**
* `buyume_egimi`: 3-year log-linear growth slope per location.
* `lokasyon_olcek`: Location size baseline.
* `buyume_x_trend`: Interaction term.

**E. Lagged Review Features**
* `yorum_rolling4`, `yorum_rolling8`: 4-week and 8-week rolling averages of review counts, shifted by 1 week to prevent leakage.

### Modeling Pipeline

1. **Data Cleaning**: Drops two leaky columns (`Ziyaretci_2025`, `uygulanan_beta`) and fills 362 missing rating values using location-level means.
2. **Feature Engineering**: Builds all five feature families above.
3. **Hyperparameter Search**: `RandomizedSearchCV` with `TimeSeriesSplit` (3 folds, 20 candidates = 60 fits) finds the best Random Forest configuration without breaking time order.
4. **Feature Selection**: Features with importance below 0.005 are dropped (48 → 17), reducing noise.
5. **Final Training**: Model is fit with `oob_score=True` for independent out-of-bag validation.

**Best hyperparameters found**: `n_estimators = 500`, `min_samples_split = 10`, `min_samples_leaf = 5`, `max_features = 0.4`, `max_depth = 10`.

### Forecasting Logic

```
growth_multiplier   = exp(growth_slope × (year − 2023.5))
projected_yearly_avg = yearly_avg_2022_2025 × growth_multiplier
final_prediction    = predicted_relative_demand × projected_yearly_avg × open_ratio_mask
```

**Safety constraints:**
* The growth slope is **capped at ±15% per year** in log-space. Without this cap, locations with steep recent trends (e.g. Abant, slope ≈ 0.546) would produce unrealistic 4× growth predictions for 2026.
* An **open-ratio mask** zeroes out predictions for weeks when the site has historically been closed (e.g. winter at high-altitude plateaus). This fixed v1's biggest error — predicting 1,000–3,000 visitors during weeks when the actual count was 0.

### Validation Strategy

The model is validated with **four independent signals**:

* **Hold-out test on 2025**: The evaluation model never sees 2025 data during training.
* **Out-of-bag (OOB) score**: Random Forest's built-in cross-validation.
* **4-fold TimeSeriesSplit CV**: Confirms performance is stable across different train/test splits.
* **Per-location performance breakdown**: Identifies sites where the model is most and least reliable.

---

## 6. Findings

### Overall Performance

| Metric | Train (2022 – 2024) | Test (2025, unseen) |
|---|---|---|
| $R^2$ | 0.948 | **0.880** |
| MAE | 1,239 visitors | **2,085 visitors** |
| Train–Test gap | — | **0.068** (no overfitting) |
| OOB $R^2$ (log-space) | — | **0.845** |

The Test $R^2$ of 0.88 exceeds the 0.85 health-target. The Train–Test gap of 0.068 (well below the 0.10 threshold) and the strong OOB score confirm the model is **not overfit**. Total prediction deviation across all of 2025 is **+5.3%** — the model is mildly optimistic in aggregate but not systematically biased.

### Cross-Validation Results

| Fold | Train size | Test size | $R^2$ | MAE |
|---|---|---|---|---|
| Fold 1 | 508 | 505 | 0.229 | 4,408 |
| Fold 2 | 1,013 | 505 | 0.874 | 2,439 |
| Fold 3 | 1,518 | 505 | 0.790 | 1,362 |
| Fold 4 | 2,023 | 505 | 0.878 | 2,655 |
| **Mean (Folds 2–4)** | — | — | **0.847 ± 0.041** | **2,152** |

**Interpretation:** Fold 1's weak performance is **structural**, not a bug — the seasonal pattern features require at least one year of past history to compute, and Fold 1 has too little. Folds 2–4 are extremely stable at $R^2 \approx 0.85$, matching the held-out 2025 result. The production model, trained on the full 2022–2025 history, has no such cold-start problem.

### Per-Location Performance (Sorted by R²)

| Tier | Location | $R^2$ | MAE | Avg. Visitors | MAE % |
|---|---|---|---|---|---|
| 🟢 High | Hıdırnebi yaylası | 0.918 | 912 | 3,774 | 24.2% |
| 🟢 High | Erfelek Tatlıca Şelaleleri | 0.894 | 1,681 | 6,038 | 27.8% |
| 🟢 High | Borçka Karagöl Tabiat Parkı | 0.890 | 470 | 1,779 | 26.4% |
| 🟢 High | Perşembe Yaylası | 0.883 | 6,388 | 26,415 | 24.2% |
| 🟢 High | Şahinkaya Kanyonu | 0.848 | 1,185 | 3,774 | 31.4% |
| 🟢 High | Kümbet yaylası | 0.847 | 1,518 | 4,717 | 32.2% |
| 🟢 High | Şaşvat Karagöl | 0.829 | 298 | 868 | 34.3% |
| 🟢 High | Elevit Yaylası | 0.829 | 381 | 1,038 | 36.7% |
| 🟢 High | Horma Kanyonu | 0.813 | 1,652 | 6,038 | 27.4% |
| 🟡 Mid | Kuzalan Şelalesi Tabiat Parkı | 0.753 | 972 | 2,547 | 38.2% |
| 🟡 Mid | Yedigöller Milli Parkı | 0.738 | 2,331 | 6,611 | 35.3% |
| 🟡 Mid | Ulugöl Tabiat Parkı | 0.679 | 2,381 | 5,661 | 42.1% |
| 🔴 Low | Güzeldere Şelalesi Tabiat Parkı | 0.645 | 750 | 2,741 | 27.4% |
| 🔴 Low | Gölcük Tabiat Parkı | 0.531 | 2,322 | 8,506 | 27.3% |
| 🔴 Low | Valla Kanyonu | 0.459 | 1,287 | 2,830 | 45.5% |
| 🔴 Low | Abant Gölü Tabiat Parkı | 0.413 | 8,836 | 30,224 | 29.2% |

**Summary:**
* 🟢 **High performance ($R^2 \geq 0.80$): 9 of 16 sites** | Mean $R^2$ = 0.861
* 🟡 **Mid performance ($0.65 \leq R^2 < 0.80$): 3 of 16 sites** | Mean $R^2$ = 0.723
* 🔴 **Low performance ($R^2 < 0.65$): 4 of 16 sites** | Mean $R^2$ = 0.512

**Why some locations underperform:** The low-tier sites (Abant, Gölcük, Valla, Güzeldere) tend to be open year-round and have less pronounced seasonal swings. The model's strongest signal — the seasonal pattern of the same week across past years — is much weaker for these sites because the pattern itself is flatter. Abant in particular is a major year-round attraction near Bolu, with crowds driven more by weekend/holiday dynamics than by season.

### Feature Importance

The top 10 features (out of 17 selected) carry the bulk of predictive weight:

| Feature | Importance |
|---|---|
| `komsu_hafta_ort` (±1-week historical avg) | **0.380** |
| `ayni_hafta_gecmis_ort` (same-week historical mean) | 0.162 |
| `son_yil_ayni_hafta` (last-year same week) | 0.154 |
| `nem_sicaklik` (humidity × temperature) | 0.067 |
| `temp` | 0.031 |
| `yil_ici_hafta` (week-of-year) | 0.028 |
| `hafta_cos` | 0.027 |
| `temp_kare` | 0.026 |
| `sicaklik_konfor` | 0.023 |
| `mevsim` | 0.020 |

**Contribution by feature category:**

| Category | Total Importance | Share |
|---|---|---|
| Seasonal pattern (past years) | 0.696 | **69.6%** |
| Weather + interactions | 0.170 | 17.0% |
| Time + holiday | 0.098 | 9.8% |
| Other (reviews, location) | 0.037 | 3.7% |
| Location growth dynamics | 0.000 | 0.0% |

> **Note:** `buyume_egimi` (growth slope) was eliminated *inside* the Random Forest by feature selection — but it is used as **post-processing** scaling in `gelecek_tahmin()`. This deliberate architecture lets the model separate **relative demand** (seasonal, handled by the RF) from **location scale** (growth, handled by multiplication). The cleanness of this separation is why low-importance features didn't hurt performance.

### Key Insights

1. **Past-week patterns dominate**: A single feature — the historical average of ±1 surrounding weeks at the same location — explains 38% of the model's decisions. Hypothesis Test A predicted this: temperature explains a lot, but a location's own history explains even more.
2. **Weather signal is real but secondary**: 17% of importance, with `nem_sicaklik` (humidity × temp interaction) outperforming raw temperature. This matches Hypothesis Test E ($\Delta R^2 = +0.37$, $p < 0.01$).
3. **Region × precipitation is empirically pointless**: Hypothesis Test D rejected the interaction, and the model never reaches for region-based features either. Saved engineering effort.
4. **Seasonal hypothesis is the strongest claim**: Test C showed summer medians 5–30× other seasons in 16/16 sites — and the model's top three features are all seasonal pattern features. The chain *Hypothesis Testing → Feature Design → Model Performance* is internally consistent.
5. **Forecasts for 2026 are plausible**: e.g., Perşembe Yaylası mid-August holiday = **100,521 visitors** (vs. annual avg 22,380, growth multiplier 1.26×); Horma Kanyonu in January at -2°C = **1,332 visitors**; Abant Gölü in November = 55,772 visitors. The model produces sensibly different magnitudes for sensibly different scenarios.

### Overall Model Health: **72 / 100**

The end-of-pipeline health check evaluates the model on 9 strict criteria:

| Criterion | Status | Score |
|---|---|---|
| Test $R^2 \geq 0.85$ | ✅ 0.880 | 1.0 |
| Train–test gap $< 0.10$ | ✅ 0.068 | 1.0 |
| OOB $R^2 \geq 0.80$ | ✅ 0.845 | 1.0 |
| Total deviation $< \pm 5\%$ | ➖ +5.3% (acceptable) | 0.5 |
| $R^2 > 0$ in all volume quartiles | ⚠️ Fails on low-volume weeks | 0.0 |
| Median APE $< 25\%$ | ➖ 33.7% (acceptable) | 0.5 |
| CV consistent across all folds | ➖ Fold 1 weak; Folds 2–4 solid | 0.5 |
| No data leakage | ✅ Verified | 1.0 |
| Production readiness | ✅ Full-data model deployed | 1.0 |

The two partial-credit losses are honest acknowledgements rather than bugs — see the Limitations section below.

---

## 7. Limitations and Future Work

### Limitations

* **Visitor counts are proxied, not measured.** Annual totals come from web-scraped news and tourism statistics; weekly disaggregation relies on Google review density. Silent visitors (those who don't post) are systematically underrepresented. A true ground-truth dataset (e.g., entrance gate counts from park authorities) would improve everything downstream.

* **Low-volume weeks remain hard.** The volume-quartile breakdown showed the model's $R^2$ is **+0.80 for high-volume weeks but negative for low-volume weeks** ($R^2 = -8.9$ for the lowest quartile). Translation: when actual visitors are very low (e.g., winter or shoulder-season weeks), small absolute errors translate into large *relative* errors, and predicting the bucket mean would do better than the model. The open-ratio mask reduces this, but doesn't eliminate it.

* **Spatial interpolation introduces uncertainty for high-altitude sites.** Several plateaus lack nearby weather stations, so weather is interpolated. The actual microclimate of a high-altitude plateau can differ measurably from the station closest to it.

* **Growth slope is capped at ±15% / year.** This is a deliberate safety constraint — without it, sites with steep recent trends (Abant's slope was ≈ 0.55) would produce 4× forecasts for 2026. But for genuinely fast-growing sites, the cap will systematically *underestimate* their 2026 numbers.

* **The 4-year window includes the post-pandemic boom.** 2022–2025 captures the rapid post-COVID recovery in domestic tourism. The seasonal pattern features assume this trend continues; a sudden flattening (e.g., economic downturn, new restrictions) would degrade predictions.

* **Social media virality is not modeled.** A site that goes viral on TikTok or Instagram can see a sudden, structurally-unpredictable spike. The model has no mechanism for this.

* **Holiday encoding is simple.** The `tatil_mi` flag treats all holidays equally; religious holidays (Ramazan/Kurban Bayramı) and secular national holidays (29 Ekim) likely have different effects on camping demand but aren't separated.

### Future Work

* **Mobile cell-tower data or Strava/Komoot heatmaps** would give a far more accurate measure of human density than review counts, especially during weekdays when reviews lag actual visits.

* **Sentinel-2 / Landsat satellite imagery** could be combined with simple computer vision to count vehicles in parking lots — giving an entirely independent ground truth.

* **Real-time forecast API** built on top of the production model, exposed through the `Local Website/` prototype already included in this repository, would let users query 2026 forecasts directly.

* **Confidence intervals via Quantile Regression Forest** would replace the current point estimates with 10/50/90 percentile predictions — much more useful for planning under uncertainty.

* **Multi-output modeling**: jointly predict visitor count, average stay duration, and satisfaction score (from forward-looking review sentiment).

* **Generalization to other regions**: Toros mountains, Kaçkar interior, Mediterranean coast — the same pipeline should port without architectural changes, only data substitution.

* **Disambiguating holiday types**: Separating religious vs. national vs. school-vacation holidays would likely sharpen the holiday × season interaction terms.

---

## 8. Repository Structure & Reproducibility

```
.
├── All Codes/                        # All Python scripts (ML, EDA, hypothesis tests)
│   └── ml.py                         # Main ML training & forecast script
├── Fixed Data/                       # The merged dataset used by all scripts
│   └── AA_Makine_Ogrenmesi_Hazir_Tum_Veri_YENI.csv
├── Merged data/                      # Intermediate merged datasets
├── Weather data 2022-2025 daily/     # Raw daily weather pulls (Meteostat)
├── Google Maps comments/             # Raw scraped Google Maps reviews
├── EDA_Grafikleri/                   # 64 per-site EDA figures
├── EDA_GRAPHS_HUMAN_DENSITY/         # Weekly visitor bar charts
├── Hypothesis tests/                 # Scripts for Tests A–E
├── Local Website/                    # Interactive forecast prototype
├── Code Guidelines/                  # Development standards
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

### Reproducing the Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main ML pipeline (training + validation + 2026 forecast examples)
cd "All Codes"
python ml.py
```

Expected runtime: ~2–3 minutes on a modern laptop. The script prints all metrics reported in Section 6 plus example 2026 predictions.

For the interactive prototype:
```bash
cd "Local Website"
# Follow the README inside the Local Website folder
```

---

## 9. Academic Integrity & AI Disclosure

In accordance with the academic integrity guidelines of the DSA 210 course, I declare that AI tools (Large Language Models) were used to assist with **code generation, debugging, data-processing utilities, and text refinement** during this project. The conceptual design, feature engineering choices, hypothesis test selection, and analysis of the results are my own.

All specific prompts used and the corresponding generated outputs (chat histories) have been **fully saved and documented** and are available upon request.

No part of the dataset construction, statistical reasoning, or model evaluation was outsourced to AI without my review and verification.
