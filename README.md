# A Decision Support Model for Spatial and Temporal Analysis of Camping Sites in the Black Sea Region

## 1. Motivation and Purpose
The Black Sea region is a spectacular destination hosting numerous magnificent camping areas. There are dozens of alternatives to visit and dozens of places to see. However, we lack a data or information source even for the most popular among these locations. My objective in this project is to select 16 of the most popular camping spots in the Black Sea region, gather information about them, and understand their general popularity. By analyzing which camping areas become popular during specific seasons and determining the overall crowd density of these areas on an annual and weekly basis, the ultimate goal is to create a reliable information source that people can depend on when planning their trips.

## 2. Data Sources and Collection Methodology
Since there is no centralized data set which can contain all the specific parameters and information for analysing these 16 camp sites, the “Unified Data Set” will be achieved by collecting and cleaning the data set obtained by independent sources by the following pipeline:

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

Before jumping into feature engineering, I ran five hypothesis tests to double-check the assumptions I was building my model on. Since the visitor data is skewed a lot and season plays a huge role (summer brings both high temperatures and lots of people, so a basic correlation wouldn't work), I used non-parametric and randomization-based methods. For tests A, B, and C, I used the Benjamini–Hochberg FDR correction to handle multiple testing.

### A. Season-controlled temperature effect

- **Method:** Stratified Permutation Test with Spearman correlation (10,000 iterations)
- **H₀:** If we keep the season constant, temperature and the number of visitors have no relationship.
- **H₁:** Temperature and visitors are related, even when we control for the season.

I shuffled the temperatures within each season to divide the actual temperature effect from just the "summer effect."

- **Result:** $H_0$ was rejected in all 16 sites after FDR correction. Spearman r was between 0.46 and 0.91.
- **Takeaway:** Temperature is a strong predictor on its own. This proved that it was a good idea to use temp and temp_kare as main features.

### B. Precipitation effect on visitors

- **Method:** Mann–Whitney U Test with FDR correction
- **H₀:** Visitor distributions are equal between rainy and non-rainy weeks.
- **Result:** H₀ rejected in 6 / 16 sites after FDR correction.
- **Takeaway:** How much rain matters really depends on the location. In the Eastern Black Sea area, it rains over 190 out of 210 weeks, therefore comparing the "rainy vs. non-rainy" kinda meaningless in these areas. But in the Western and Central sites where rain is less constant, it actually makes a difference.

### C. Seasonal effect on visitors

- **Method:** Kruskal-Wallis H Test with FDR correction
- **H₀:** The median number of visitors is the same across all four seasons.
- **Result:** H₀ was rejected in all 16 sites.
- **Takeaway:** Season is a good predictor for every site. This showed that using the variable mevsim, cyclical time encoding (ay_sin/cos, hafta_sin/cos), and seasonal patterns is meaningful.

### D. Region × precipitation sensitivity

- **Method:** Chi-Square Test of Independence
- **H₀:** The region and whether a site gets "low visitors on rainy weeks" have no relations.

The approach should be fixeds here: at first, the test maden at the location level ($n = 4$), but the math didn't work. So, it is swicthed up to analyzing week-by-location data ($n = 3360$), which makes way more sense statistically.

- **Result:** χ² = 0.116, p = 0.944, Cramér's V = 0.0059. H₀ accepted.
- **Takeaway:** The region doesn't change how sensitive people are to rain. Because of this, I didn't need to add a "Region × precipitation" interaction term in the ML model, which saved me a lot of time.

### E. Statistical significance of weather features in the ML model

- **Method:** Permutation Test on model performance (100 iterations)
- **H₀:** Weather features don't improve the model's accuracy.

all the weather columns shuffled together to break their link to visitor numbers, while keeping the weather variables properly correlated with each other.

- **Result:** Observed R² = 0.194 vs. mean null R² = -0.178, so ΔR² = +0.372, p < 0.01 (zero out of 100 null permutations beat the observed value).
- **Takeaway:** Weather features contribute significantly. This is more or less the empirical justification for the whole project premise.

### Summary

| Test | Method | Result |
|---|---|---|
| A. Season-controlled temperature | Stratified Permutation | 16 / 16 significant |
| B. Precipitation | Mann–Whitney U + FDR | 6 / 16 significant |
| C. Seasonal effect | Kruskal–Wallis + FDR | 16 / 16 significant |
| D. Region × precipitation | Chi-Square (n = 3,360) | Not significant |
| E. ML weather contribution | Permutation on R² | p < 0.01 |

So the overall picture is: temperature and season are universal drivers, precipitation gives location-specific signal, and weather as a whole significantly improves predictive accuracy. The model architecture in the next section is built around these confirmed signals.

---

## 5. Machine Learning Model

### Architecture

The system uses a Random Forest Regressor in a two-model setup:

| Model | Training Data | Purpose |
|---|---|---|
| Evaluation Model (`rf_model`) | 2022 – 2024 (1,680 rows) | Reports honest performance (R², MAE) on held-out 2025 |
| Production Model (`rf_production`) | 2022 – 2025 (2,528 rows) | Generates forecasts for 2026+ using the freshest data |

The most important design choice in the whole project: instead of predicting absolute visitor numbers directly, the model predicts *relative demand in log-space* — basically a ratio of how busy a location is compared to its own annual average. The reason is that seasonal patterns generalize well across locations (summer is busier everywhere), but location size does not (Abant gets way more visitors than Şaşvat Karagöl). By decoupling these two things and reconstructing the absolute forecast at the end through multiplication, the model can transfer seasonal knowledge across locations without getting confused by scale differences.

### Feature engineering (48 → 17 selected features)

**A. Seasonal pattern features (the strongest signal):**
- `ayni_hafta_gecmis_ort`: historical mean of relative demand for the same week-of-year at this location.
- `ayni_hafta_gecmis_std`: standard deviation of the same week across past years (an uncertainty signal).
- `son_yil_ayni_hafta`: last year's relative demand for the same week (the freshest seasonal signal).
- `komsu_hafta_ort`: average of ±1 neighboring weeks (a smoothing feature).

These all use strict past-only logic to prevent leakage. When predicting week 30 of 2025, the features only see week 30 from 2022, 2023, and 2024 — never the target week itself. I had to be careful about this because if you get the index window even slightly wrong you leak future information into training.

**B. Weather and interaction features:**
- Base: `temp`, `prcp`, `snow`, `rain`, `wspd`, `rhum`
- Engineered: `temp_kare` (temperature squared), `sicak_yagmur` (warm rain), `kar_soguk` (cold snow), `sicaklik_konfor` (-((temp - 20)²), a comfort score), `nem_sicaklik` (humidity × temperature), `kotu_hava` (rain × wind)

**C. Time features:**
- Cyclical encoding: `ay_sin/cos`, `hafta_sin/cos`, so the model understands that December and January are neighbors.
- `mevsim`, `tatil_mi` (holiday flag), `tatil_yaz`, `tatil_ilkbahar` (holiday × season interactions).

**D. Location growth dynamics:**
- `buyume_egimi`: 3-year log-linear growth slope per location.
- `lokasyon_olcek`: a location size baseline.
- `buyume_x_trend`: interaction term.

**E. Lagged review features:**
- `yorum_rolling4`, `yorum_rolling8`: 4-week and 8-week rolling averages of review counts, shifted by 1 week to prevent leakage.

### Modeling pipeline

1. **Data cleaning:** drops two leaky columns (`Ziyaretci_2025`, `uygulanan_beta`) and fills 362 missing rating values using location-level means.
2. **Feature engineering:** builds all five feature families described above.
3. **Hyperparameter search:** RandomizedSearchCV with TimeSeriesSplit (3 folds, 20 candidates = 60 fits total) to find the best Random Forest configuration without breaking the time order.
4. **Feature selection:** features with importance below 0.005 are dropped (48 → 17), reducing noise.
5. **Final training:** the model is fit with `oob_score=True` so I get an independent out-of-bag validation score.

Best hyperparameters: `n_estimators = 500`, `min_samples_split = 10`, `min_samples_leaf = 5`, `max_features = 0.4`, `max_depth = 10`.

### Forecasting logic

```
growth_multiplier   = exp(growth_slope × (year − 2023.5))
projected_yearly_avg = yearly_avg_2022_2025 × growth_multiplier
final_prediction    = predicted_relative_demand × projected_yearly_avg × open_ratio_mask
```

I added two safety constraints to this:

The growth slope is capped at ±15% per year in log-space. Without this cap, locations with steep recent trends (e.g. Abant, slope ≈ 0.546) would predict around 4× growth for 2026, which is clearly unrealistic.

There's an open-ratio mask that zeroes out predictions for weeks when the site has historically been closed (e.g. winter at high-altitude plateaus). This was actually the biggest fix from v1 of the model — before this mask, the model would predict 1,000–3,000 visitors for weeks when the actual count was zero. Adding the mask was a small change but it made a noticeable difference in the MAE.

### Validation strategy

The model gets validated through four independent signals:

- Hold-out test on 2025: the evaluation model never sees 2025 data during training.
- Out-of-bag (OOB) score: Random Forest's built-in cross-validation.
- 4-fold TimeSeriesSplit CV: confirms that performance is stable across different splits.
- Per-location breakdown: shows which sites the model handles well and which it struggles with.

---

## 6. Findings

### Overall performance

| Metric | Train (2022 – 2024) | Test (2025, unseen) |
|---|---|---|
| R² | 0.948 | 0.880 |
| MAE | 1,239 visitors | 2,085 visitors |
| Train–Test gap | — | 0.068 (no overfitting) |
| OOB R² (log-space) | — | 0.845 |

The Test R² of 0.88 is above my 0.85 target. The Train–Test gap of 0.068 is well below the 0.10 threshold, and the OOB score backs up that the model is not overfit. The total prediction deviation across all of 2025 is +5.3%, so the model is slightly optimistic in aggregate but not systematically biased.

### Cross-validation results

| Fold | Train size | Test size | R² | MAE |
|---|---|---|---|---|
| Fold 1 | 508 | 505 | 0.229 | 4,408 |
| Fold 2 | 1,013 | 505 | 0.874 | 2,439 |
| Fold 3 | 1,518 | 505 | 0.790 | 1,362 |
| Fold 4 | 2,023 | 505 | 0.878 | 2,655 |
| **Mean (Folds 2–4)** | — | — | **0.847 ± 0.041** | **2,152** |

Fold 1 looks bad on the surface but the reason is actually structural, not a bug. The seasonal pattern features need at least one full year of past history to compute, and Fold 1 simply doesn't have enough past data for that. Folds 2 through 4 are very stable at R² ≈ 0.85, which matches the held-out 2025 result. The production model is trained on all four years so it doesn't have this cold-start problem.

### Per-location performance (sorted by R²)

| Tier | Location | R² | MAE | Avg. Visitors | MAE % |
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

Summary:
- 🟢 High performance (R² ≥ 0.80): 9 of 16 sites, mean R² = 0.861
- 🟡 Mid performance (0.65 ≤ R² < 0.80): 3 of 16 sites, mean R² = 0.723
- 🔴 Low performance (R² < 0.65): 4 of 16 sites, mean R² = 0.512

The low-tier sites (Abant, Gölcük, Valla, Güzeldere) tend to be open year-round and have less pronounced seasonal swings. The model's main signal — the seasonal pattern of the same week across past years — is much weaker for these sites because the pattern itself is flatter. Abant especially is a major year-round attraction near Bolu and its visitor numbers are driven more by weekend and holiday dynamics than by season, which the current feature set doesn't capture as cleanly.

### Feature importance

The top 10 features (out of 17 selected) carry most of the predictive weight:

| Feature | Importance |
|---|---|
| `komsu_hafta_ort` (±1-week historical avg) | 0.380 |
| `ayni_hafta_gecmis_ort` (same-week historical mean) | 0.162 |
| `son_yil_ayni_hafta` (last-year same week) | 0.154 |
| `nem_sicaklik` (humidity × temperature) | 0.067 |
| `temp` | 0.031 |
| `yil_ici_hafta` (week-of-year) | 0.028 |
| `hafta_cos` | 0.027 |
| `temp_kare` | 0.026 |
| `sicaklik_konfor` | 0.023 |
| `mevsim` | 0.020 |

Grouped by category:

| Category | Total Importance | Share |
|---|---|---|
| Seasonal pattern (past years) | 0.696 | 69.6% |
| Weather + interactions | 0.170 | 17.0% |
| Time + holiday | 0.098 | 9.8% |
| Other (reviews, location) | 0.037 | 3.7% |
| Location growth dynamics | 0.000 | 0.0% |

One small note on the last row: `buyume_egimi` (growth slope) was eliminated by feature selection inside the Random Forest, but it's still being used as a post-processing scaling factor in `gelecek_tahmin()`. This was actually intentional — the architecture lets the Random Forest handle *relative* demand (seasonal patterns) while the multiplication step handles *location scale* (growth). Keeping those two roles separate is part of why dropping the low-importance features didn't hurt overall performance.

### Key insights

A few patterns stand out from the results:

1. **Past-week patterns dominate.** A single feature — the historical average of ±1 surrounding weeks at the same location — explains 38% of the model's decisions. Hypothesis Test A already hinted at this: temperature explains a lot, but a location's own history explains even more.

2. **Weather signal is real but secondary.** 17% of importance, with `nem_sicaklik` (humidity × temp interaction) actually outperforming raw temperature. This is consistent with Hypothesis Test E (ΔR² = +0.37, p < 0.01).

3. **Region × precipitation is empirically pointless.** Hypothesis Test D rejected the interaction, and the model doesn't reach for region-based features either when given the choice. So that was engineering effort I saved.

4. **The seasonal hypothesis is the strongest claim.** Test C showed summer medians 5–30× other seasons in 16/16 sites, and the top three features in the final model are all seasonal pattern features. The chain hypothesis testing → feature design → model performance is internally consistent here, which was satisfying to see.

5. **Forecasts for 2026 look plausible.** Some examples: Perşembe Yaylası in mid-August during a holiday gives 100,521 visitors (vs. annual avg 22,380, growth multiplier 1.26×); Horma Kanyonu in January at -2°C gives 1,332 visitors; Abant Gölü in November gives 55,772 visitors. The model produces sensibly different magnitudes for sensibly different scenarios.

### Overall model health: 72 / 100

I ran an end-of-pipeline health check on 9 strict criteria:

| Criterion | Status | Score |
|---|---|---|
| Test R² ≥ 0.85 | ✅ 0.880 | 1.0 |
| Train–test gap < 0.10 | ✅ 0.068 | 1.0 |
| OOB R² ≥ 0.80 | ✅ 0.845 | 1.0 |
| Total deviation < ±5% | ➖ +5.3% (acceptable) | 0.5 |
| R² > 0 in all volume quartiles | ⚠️ Fails on low-volume weeks | 0.0 |
| Median APE < 25% | ➖ 33.7% (acceptable) | 0.5 |
| CV consistent across all folds | ➖ Fold 1 weak; Folds 2–4 solid | 0.5 |
| No data leakage | ✅ Verified | 1.0 |
| Production readiness | ✅ Full-data model deployed | 1.0 |

The two partial-credit losses are honest acknowledgements rather than bugs — I unpack them in the next section.

---

## 7. Limitations and Future Work

### Limitations

**Visitor counts are proxied, not measured.** Annual totals come from web-scraped news and tourism statistics, and weekly disaggregation relies on Google review density. Silent visitors (people who don't post a review) are systematically underrepresented. A real ground-truth dataset, like entrance gate counts from park authorities, would improve basically everything downstream of this.

**Low-volume weeks remain hard.** The volume-quartile breakdown showed the model's R² is +0.80 for high-volume weeks but actually *negative* (R² = -8.9 for the lowest quartile) for low-volume weeks. In plain terms: when actual visitors are very low (winter or shoulder-season weeks), small absolute errors translate into large *relative* errors, and just predicting the bucket mean would do better than the model. The open-ratio mask reduces this but doesn't eliminate it.

**Spatial interpolation adds uncertainty for high-altitude sites.** Several plateaus don't have a nearby weather station, so weather has to be interpolated from the closest one. The microclimate of a high-altitude plateau can genuinely differ from the station closest to it, so this is a real source of error I couldn't fully fix.

**The growth slope cap at ±15% per year is a tradeoff.** It's a safety constraint, but for genuinely fast-growing sites the cap will systematically underestimate 2026 numbers. Abant's raw slope was around 0.55, which would have given an unrealistic 4× forecast, so the cap was clearly needed — but it's a blunt instrument.

**The 4-year window includes the post-pandemic boom.** 2022–2025 captures the rapid post-COVID recovery in domestic tourism. The seasonal pattern features implicitly assume this trend continues, so a sudden flattening (e.g. economic downturn or new restrictions) would degrade the predictions.

**Social media virality is not modeled.** If a site goes viral on TikTok or Instagram it can see a sudden, structurally unpredictable spike. The model has no way to handle this.

**Holiday encoding is simple.** The `tatil_mi` flag treats all holidays equally, but religious holidays (Ramazan / Kurban Bayramı) and secular national holidays (29 Ekim) almost certainly affect camping demand differently. I didn't separate them in this version.

### Future work

A few directions I'd take this if I had more time:

- Mobile cell-tower data or Strava/Komoot heatmaps would give a much better measure of actual human density than review counts, especially for weekdays when reviews tend to lag visits.
- Sentinel-2 or Landsat satellite imagery combined with simple computer vision (counting vehicles in parking lots) would give a completely independent ground truth.
- A real-time forecast API built on top of the production model, exposed through the `Local Website/` prototype already included in the repo, would let users query 2026 forecasts directly.
- Replacing point estimates with confidence intervals using Quantile Regression Forest (10/50/90 percentiles) would be much more useful for actual planning.
- Multi-output modeling: jointly predicting visitor count, average stay duration, and satisfaction score from forward-looking review sentiment.
- Generalization to other regions (Toros mountains, Kaçkar interior, Mediterranean coast) — the pipeline should port over with mostly just a data substitution.
- Disambiguating holiday types: separating religious vs. national vs. school-vacation holidays would likely sharpen the holiday × season interactions.

---

## 8. Repository Structure and Reproducibility

```
.
├── All Codes/                          # All Python scripts (ML, EDA, hypothesis tests)
│   └── ml.py                           # Main ML training & forecast script
├── Fixed Data/                         # The merged dataset used by all scripts
│   └── AA_Makine_Ogrenmesi_Hazir_Tum_Veri_YENI.csv
├── Merged data/                        # Intermediate merged datasets
├── Weather data 2022-2025 daily/       # Raw daily weather pulls (Meteostat)
├── Google Maps comments/               # Raw scraped Google Maps reviews
├── EDA_Grafikleri/                     # 64 per-site EDA figures
├── EDA_GRAPHS_HUMAN_DENSITY/           # Weekly visitor bar charts
├── Hypothesis tests/                   # Scripts for Tests A–E
├── Local Website/                      # Interactive forecast prototype
├── Code Guidelines/                    # Development standards
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

### Reproducing the results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main ML pipeline (training + validation + 2026 forecast examples)
cd "All Codes"
python ml.py
```

Expected runtime is around 2–3 minutes on a modern laptop. The script prints all the metrics reported in Section 6 plus example 2026 predictions.

For the interactive prototype:

```bash
cd "Local Website"
# Follow the README inside the Local Website folder
```

---

## 9. Academic Integrity and AI Disclosure

In line with the DSA 210 academic integrity guidelines, I want to be transparent: AI tools (Large Language Models) were used during this project for code generation, debugging, data-processing utilities, and text refinement. The conceptual design, feature engineering decisions, hypothesis test selection, and the analysis of results are my own work.

All specific prompts I used along with the corresponding generated outputs (chat histories) have been saved and documented, and they're available on request.

No part of the dataset construction, statistical reasoning, or model evaluation was outsourced to AI without my review and verification.
