# A Decision Support Model for Spatial and Temporal Analysis of Camping Sites in the Black Sea Region

## 1. Motivation and Purpose

The Black Sea region is a spectacular destination hosting numerous magnificent camping areas. There are dozens of alternatives to visit and dozens of places to see. However, we lack a data or information source even for the most popular among these locations. My objective in this project is to select 16 of the most popular camping spots in the Black Sea region, gather information about them, and understand their general popularity. By analyzing which camping areas become popular during specific seasons and determining the overall crowd density of these areas on an annual and weekly basis, the ultimate goal is to create a reliable information source that people can depend on when planning their trips.

## 2. Data Sources and Collection Methodology

There is no single dataset somewhere with all of this stuff in one place. I checked. So I had to build my own "Unified Dataset" by grabbing data from different places and cleaning it up. Here is how I did it.

**Weather:** I used the `meteostat` Python library. Got daily data and then grouped it into weekly chunks. A few of the camp sites are on high mountains and don't have a weather station close by, so for those I used spatial interpolation to estimate the conditions. It is not perfect but it was the best I could do.

**Visitor counts (digital footprint):** This was the tricky part. Some of the camp sites are pretty remote and nobody officially counts how many people show up each week. What I did instead: first I scraped yearly visitor numbers from news articles, blogs, and other semi-official sources. Then I went through Google reviews for each site week by week. By comparing how many reviews got written in each week to the yearly total I already had, I could estimate roughly how many people came each week. Not the cleanest method but I think it works.

**Access to healthcare, cities and road conditions:** To figure out how isolated a site is, I used Google Maps + some web scraping to find the closest city and the closest hospital, plus how long it takes to drive there. For slope I used Google Earth. And for road surface info I pulled from OpenStreetMap.

### Dataset Characteristics

The data covers 4 years, 2022 to 2025. Weather is in weekly intervals. For visitor numbers I went through Google reviews from all four years to get monthly estimates.

Sample size wise: weather data is 16 sites × 208 weeks = about 3,328 weekly records. On the visitor side, I went through 52,719 Google reviews in total. That part took forever.

A few things stay fixed and don't change over time: yearly visitor numbers (from news and official sources), driving distance and time to the closest city and hospital (from Google Maps), slope (from Google Earth), and road surface (from OpenStreetMap).

Main variables and units:
- Weather: Temperature in °C, precipitation in mm
- Distance/access: Driving time in minutes
- Terrain: Slope in degrees, soil composition in percentages (clay/sand)

## 3. Exploratory Data Analysis (EDA)

I did EDA for each of the 16 sites separately. For every site there is a weekly visitor time-series, a dual-axis chart of visitors vs. temperature, a Pearson correlation heatmap with all the weather variables, plus visitor histograms and weather scatter plots. So that's 64 figures in total under `EDA_Grafikleri/`. The cleaner weekly visitor bar charts are saved separately under `EDA_GRAPHS_HUMAN_DENSITY/`.

## 4. Hypothesis Tests

Before doing any feature engineering I ran 5 hypothesis tests. Mostly to sanity check what I was assuming.

The visitor data is super skewed and season has a huge effect (summer means both high temps and lots of people at the same time), so a plain correlation would just give you the obvious answer. That's why I used non-parametric and randomization-based methods. For tests A, B, and C I also applied Benjamini–Hochberg FDR correction because I was doing the same test on 16 sites and didn't want false positives piling up.

### A. Temperature effect, with season held constant

**Method:** Stratified Permutation Test with Spearman correlation (10,000 iterations)
**H₀:** When season is fixed, temperature and visitor count have no relationship.
**H₁:** Even after you control for season, temperature and visitors are still related.

What I actually did: I shuffled temperatures only within each season. That way I could pull apart the "real" temperature effect from the "well duh, it's summer" effect.

**Result:** H₀ rejected for all 16 sites after FDR correction. Spearman r came out between 0.46 and 0.91.

**Takeaway:** Temperature is a strong predictor on its own, even with season fixed. So including `temp` and `temp_kare` as features made sense.

### B. Precipitation effect on visitors

**Method:** Mann–Whitney U Test with FDR correction
**H₀:** Visitor distributions are the same in rainy and non-rainy weeks.
**Result:** H₀ rejected for 6 out of 16 sites.

**Takeaway:** Rain matters but it really depends on where you are. In the Eastern Black Sea region it rains in like 190 out of 210 weeks, so "rainy vs not rainy" is kind of a meaningless split there. In the Western and Central sites where rain is more of an on-and-off thing, it actually shows up in the data.

### C. Seasonal effect on visitors

**Method:** Kruskal–Wallis H Test with FDR correction
**H₀:** Median visitor count is the same across all four seasons.
**Result:** H₀ rejected for all 16 sites.

**Takeaway:** Season is a strong predictor everywhere. So putting `mevsim`, cyclical time features (`ay_sin/cos`, `hafta_sin/cos`), and seasonal patterns into the model was a good call.

### D. Region × precipitation sensitivity

**Method:** Chi-Square Test of Independence
**H₀:** Whether a site has "low visitors on rainy weeks" doesn't depend on its region.

This one I had to redo. At first I ran the test at the location level (n = 4) and the math just didn't really work with that small a sample. So I switched to week-by-location data (n = 3,360) and that worked much better.

**Result:** χ² = 0.116, p = 0.944, Cramér's V = 0.0059. H₀ accepted.

**Takeaway:** Region does not change how sensitive people are to rain. So I didn't add a "Region × precipitation" interaction term to the model. Saved me some effort.

### E. Do weather features actually help the ML model?

**Method:** Permutation Test on model performance (100 iterations)
**H₀:** Weather features don't improve the model's accuracy.

I shuffled all the weather columns together. That breaks the link between weather and visitors, but the weather variables stay correlated with each other (which matters, otherwise you mess up the correlation structure).

**Result:** Observed R² = 0.194 vs. mean null R² = -0.178. So ΔR² = +0.372, p < 0.01. Zero out of 100 null runs beat the actual model.

**Takeaway:** Weather features clearly contribute. This is basically the empirical justification for doing the whole project.

### Summary table

| Test | Method | Result |
|---|---|---|
| A. Season-controlled temperature | Stratified Permutation | 16 / 16 significant |
| B. Precipitation | Mann–Whitney U + FDR | 6 / 16 significant |
| C. Seasonal effect | Kruskal–Wallis + FDR | 16 / 16 significant |
| D. Region × precipitation | Chi-Square (n = 3,360) | Not significant |
| E. ML weather contribution | Permutation on R² | p < 0.01 |

So if I sum it up: temperature and season are universal, precipitation gives a signal but only in some locations, and the weather variables as a whole really do improve predictions. The model in the next section is built around exactly these signals.

---

## 5. Machine Learning Model

### Architecture

I used a Random Forest Regressor, with two separate models actually:

| Model | Training Data | Purpose |
|---|---|---|
| Evaluation Model (`rf_model`) | 2022 – 2024 (1,680 rows) | Gives me honest R² and MAE on unseen 2025 |
| Production Model (`rf_production`) | 2022 – 2025 (2,528 rows) | Makes 2026+ forecasts using the freshest data |

Now, the most important design decision in the whole project, by far, is this: instead of predicting visitor counts directly, the model predicts *relative demand in log-space*. Basically it predicts how busy a site is compared to its own yearly average.

Why did I do it like this? Because seasonal patterns are similar across all sites (summer is busy at every camping site, more or less), but the size of each site is totally different. Abant gets way more visitors than Şaşvat Karagöl, and there is nothing the model can do to learn that from weather alone. So by separating these two things and putting them back together at the end with a multiplication, the model can learn seasonal patterns across all 16 sites without getting confused by their different sizes. This was probably the single biggest improvement over my first version.

### Feature engineering (48 features narrowed down to 17)

I built five groups of features.

**A. Seasonal pattern features (turned out to be the strongest):**
- `ayni_hafta_gecmis_ort`: historical mean of relative demand for the same week-of-year at this site
- `ayni_hafta_gecmis_std`: standard deviation of the same week across past years (kind of an uncertainty signal)
- `son_yil_ayni_hafta`: last year's relative demand for the same week (the most recent seasonal signal)
- `komsu_hafta_ort`: average of the ±1 surrounding weeks (smoothing)

All of these only use past data. No leakage. When the model predicts week 30 of 2025, it sees week 30 from 2022, 2023, and 2024, but never the target week itself. I had to be really careful with the indexing because even a one-week mistake leaks the future into training. I actually messed this up the first time around and got suspiciously good results, which is how I caught it.

**B. Weather and interaction features:**

Base ones are `temp`, `prcp`, `snow`, `rain`, `wspd`, `rhum`.

Then I engineered a few interactions: `temp_kare` (temperature squared), `sicak_yagmur` (warm rain), `kar_soguk` (cold snow), `sicaklik_konfor` which is -((temp - 20)²), basically a comfort score peaking around 20°C, `nem_sicaklik` (humidity × temperature), and `kotu_hava` (rain × wind).

**C. Time features:**
- Cyclical encoding with `ay_sin/cos`, `hafta_sin/cos`. This way the model understands that December and January are neighbors.
- `mevsim`, `tatil_mi` (holiday flag), `tatil_yaz`, `tatil_ilkbahar` (holiday × season interactions).

**D. Location growth dynamics:**
- `buyume_egimi`: a 3-year log-linear growth slope per location
- `lokasyon_olcek`: a baseline for site size
- `buyume_x_trend`: interaction term

**E. Lagged review features:**
- `yorum_rolling4` and `yorum_rolling8`: 4-week and 8-week rolling averages of review counts, shifted by 1 week to avoid leakage.

### Modeling pipeline

The pipeline goes like this:

1. **Data cleaning.** I drop two columns that would cause leakage (`Ziyaretci_2025`, `uygulanan_beta`) and fill 362 missing ratings using each site's mean.
2. **Feature engineering.** All 5 groups above get built here.
3. **Hyperparameter search.** RandomizedSearchCV with TimeSeriesSplit (3 folds, 20 candidates, so 60 fits total). Time-aware splits because otherwise you'd be cheating.
4. **Feature selection.** Features with importance below 0.005 get dropped. That cuts 48 down to 17.
5. **Final training.** I set `oob_score=True` so I also get an independent out-of-bag validation.

The best hyperparameters ended up being: `n_estimators = 500`, `min_samples_split = 10`, `min_samples_leaf = 5`, `max_features = 0.4`, `max_depth = 10`.

### Forecasting logic

```
growth_multiplier    = exp(growth_slope × (year − 2023.5))
projected_yearly_avg = yearly_avg_2022_2025 × growth_multiplier
final_prediction     = predicted_relative_demand × projected_yearly_avg × open_ratio_mask
```

There are two safety rules in there.

First, the growth slope is capped at ±15% per year in log-space. Without the cap, sites with steep recent trends (like Abant, which has a slope of about 0.546) would predict like 4× growth for 2026. That is obviously not going to happen.

Second, there is an open-ratio mask that zeros out predictions for weeks when a site has historically been closed (winter at high-altitude plateaus, for example). This was actually the single biggest fix from v1 of the model. Before the mask, my model was confidently predicting 1,000 to 3,000 visitors for weeks where the actual count was zero, because it had never been told the place was closed. Tiny change, huge effect on MAE.

### Validation strategy

I checked the model in four independent ways:

- Hold-out test on 2025 (the evaluation model never sees 2025 in training)
- Out-of-bag (OOB) score from the Random Forest itself
- 4-fold TimeSeriesSplit cross-validation
- Per-location breakdown to see which sites are easy and which are hard

---

## 6. Findings

### Overall performance

| Metric | Train (2022 – 2024) | Test (2025, unseen) |
|---|---|---|
| R² | 0.948 | 0.880 |
| MAE | 1,239 visitors | 2,085 visitors |
| Train–Test gap | — | 0.068 (no overfitting) |
| OOB R² (log-space) | — | 0.845 |

Test R² is 0.88, which is above my target of 0.85. The train-test gap is 0.068, well under the 0.10 threshold I set, and the OOB score backs up the fact that the model isn't overfitting. Total prediction deviation over all of 2025 is +5.3%, so the model is slightly optimistic in total but not in a systematic way.

### Cross-validation results

| Fold | Train size | Test size | R² | MAE |
|---|---|---|---|---|
| Fold 1 | 508 | 505 | 0.229 | 4,408 |
| Fold 2 | 1,013 | 505 | 0.874 | 2,439 |
| Fold 3 | 1,518 | 505 | 0.790 | 1,362 |
| Fold 4 | 2,023 | 505 | 0.878 | 2,655 |
| **Mean (Folds 2–4)** | — | — | **0.847 ± 0.041** | **2,152** |

Fold 1 looks bad but it's actually a structural thing, not a bug. The seasonal pattern features need at least one full year of past data to compute. Fold 1 just doesn't have enough history to work with. Folds 2 to 4 are very stable around R² ≈ 0.85, which matches the 2025 hold-out result, so I'm not too worried about it. The production model is trained on all 4 years so this cold-start problem doesn't apply to it.

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

Quick summary:
- 🟢 High (R² ≥ 0.80): 9 of 16 sites, average R² = 0.861
- 🟡 Mid (0.65 ≤ R² < 0.80): 3 of 16 sites, average R² = 0.723
- 🔴 Low (R² < 0.65): 4 of 16 sites, average R² = 0.512

The low-tier sites are Abant, Gölcük, Valla, and Güzeldere. What they have in common is they are open all year round and don't have huge seasonal swings. Since my model leans really hard on seasonal patterns, when the seasonal pattern is flat, there isn't much signal left. Abant especially is a massive year-round attraction close to Bolu, and its visitor numbers are driven much more by weekends and holidays than by which season it is. The features I built don't capture that as well as I'd like.

### Feature importance

The top 10 features (out of 17) carry most of the weight:

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

And grouped by category:

| Category | Total Importance | Share |
|---|---|---|
| Seasonal pattern (past years) | 0.696 | 69.6% |
| Weather + interactions | 0.170 | 17.0% |
| Time + holiday | 0.098 | 9.8% |
| Other (reviews, location) | 0.037 | 3.7% |
| Location growth dynamics | 0.000 | 0.0% |

About that last row: `buyume_egimi` (the growth slope) got dropped by feature selection inside the Random Forest, but I'm still using it as a multiplier in `gelecek_tahmin()` after the model runs. That was on purpose. The Random Forest handles *relative* demand (seasonal stuff) and the multiplication step handles *site size and growth*. Keeping those two roles separate is partly why dropping the low-importance features didn't hurt anything.

### Key insights

A few things stood out:

1. **Past-week patterns are by far the biggest signal.** Just one feature, the average of the ±1 surrounding weeks at the same site, explains 38% of the model's decisions on its own. Hypothesis Test A already pointed in this direction, but it was still surprising to see one feature dominate that much.

2. **Weather matters, but it's in second place.** 17% of importance, and the humidity × temperature interaction (`nem_sicaklik`) actually beat raw temperature. Consistent with Hypothesis Test E.

3. **Region × precipitation really is useless.** Test D said no, and the model also doesn't pick region-based features when it has the option. So that lined up nicely.

4. **The seasonal hypothesis turned out to be the strongest claim.** Test C showed that summer medians are 5 to 30 times the other seasons in all 16 sites. And then the top three features in the final model are all seasonal pattern features. The whole chain from hypothesis testing to feature design to model results is internally consistent, which honestly felt good to see.

5. **2026 forecasts look reasonable.** A few examples: Perşembe Yaylası in mid-August during a holiday gives 100,521 visitors (vs. yearly average 22,380, growth multiplier 1.26×). Horma Kanyonu in January at -2°C gives 1,332 visitors. Abant Gölü in November gives 55,772. The numbers look sensible for sensibly different conditions.

### Overall model health: 72 / 100

I ran a final health check with 9 criteria:

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

The two partial-credit cases are honest weaknesses, not bugs. I'll talk about them next.

---

## 7. Limitations and Future Work

### Limitations

**Visitor counts are estimated, not measured.** Yearly totals come from web-scraped news and tourism stats. Weekly numbers come from Google review density. Anyone who visits but doesn't post a review is invisible to me. If I had actual entrance gate counts from park authorities, basically everything downstream would improve.

**Low-volume weeks are still hard.** The volume-quartile check showed R² is around +0.80 for high-volume weeks but actually *negative* (R² = -8.9) for the lowest quartile. In normal language: when there are very few visitors, small absolute errors turn into big relative errors. At that point, just predicting the bucket mean would do better than my model. The open-ratio mask helps but doesn't completely fix this.

**Spatial interpolation adds uncertainty for high-altitude sites.** Some plateaus don't have a nearby weather station, so I had to estimate weather from the closest one. Microclimates at altitude can be pretty different from the station nearby, so this is a real source of error I couldn't fully get rid of.

**The ±15% growth slope cap is a trade-off.** It's a safety constraint, but for sites that really are growing fast, the cap will underestimate 2026. Abant's raw slope was around 0.55, which would have given a 4× forecast, so the cap was clearly necessary. But it's a blunt tool.

**The 4-year window includes the post-pandemic boom.** 2022 to 2025 captures the rapid post-COVID recovery in domestic tourism. My seasonal pattern features sort of assume this trend keeps going, so if things suddenly flatten (economic downturn, new restrictions, whatever), predictions will get worse.

**Social media virality isn't modeled.** A site can go viral on TikTok or Instagram and get a huge sudden spike that nothing in my data could predict. The model has no way to handle that.

**Holiday encoding is simple.** The `tatil_mi` flag treats every holiday the same. But religious holidays (Ramazan, Kurban Bayramı) almost certainly affect camping demand differently from national holidays (29 Ekim) and school vacations. I didn't split them out in this version. Would be a good next step.

### Future work

A few things I'd add with more time:

- Mobile cell tower data, or Strava/Komoot heatmaps, would give a much better measure of actual human density than counting reviews. Especially for weekdays, when reviews lag behind visits.
- Sentinel-2 or Landsat satellite images plus some simple computer vision (counting cars in parking lots) could give a totally independent ground truth.
- A real-time forecast API on top of the production model, served through the `Local Website/` prototype that's already in the repo.
- Replacing point estimates with confidence intervals using Quantile Regression Forest (10/50/90 percentiles). Way more useful for actual planning than a single number.
- Multi-output modeling: predict visitor count, average stay length, and satisfaction score together, using forward-looking review sentiment.
- Testing the pipeline in other regions (Toros, Kaçkar interior, Mediterranean coast). Should mostly just need a data swap.
- Splitting holiday types properly. Religious vs. national vs. school-vacation holidays would probably make the holiday × season interactions sharper.

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

It takes about 2 to 3 minutes on a normal laptop. The script prints all the metrics from Section 6 plus some 2026 prediction examples.

For the interactive prototype:

```bash
cd "Local Website"
# Follow the README inside the Local Website folder
```

---

## 9. Academic Integrity and AI Disclosure

To be upfront, in line with the DSA 210 academic integrity guidelines: I did use AI tools (LLMs) during this project. Mainly for code generation, debugging, small data-processing utilities, and some text editing. The conceptual design, feature engineering choices, hypothesis test selection, and the interpretation of the results are my own work.

I saved all the specific prompts I used and the chat histories that came out of them. Happy to share them if asked.

No part of the dataset construction, the statistical reasoning, or the model evaluation was outsourced to AI without me reviewing and verifying it myself.
