"""
============================================================================
BLACK SEA CAMPSITES - ENHANCED HYPOTHESIS TESTS
============================================================================

This script applies 5 different hypothesis tests to statistically validate 
the core assumptions of the visitor prediction model:

  TEST 1: Permutation Test (Season-Controlled Temperature-Visitor Relationship)
  TEST 2: Mann-Whitney U + FDR Correction (Precipitation-Visitor Effect)
  TEST 3: Kruskal-Wallis Test (Season Effect)
  TEST 4: Chi-Square Test of Independence (Region × Precipitation Sensitivity)
  TEST 5: Permutation Test (Importance of Weather in the ML Model)

ALLOWED TESTS UTILIZED:
  A. Randomization (Permutation) Test  → TEST 1, TEST 5
  D. Chi-Square Test of Independence   → TEST 4
  K. Mann-Whitney U Test               → TEST 2
  T. Kruskal-Wallis Test               → TEST 3
============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import (
    spearmanr, mannwhitneyu, kruskal, chi2_contingency
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 0. VERİ YÜKLEME
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Bir üst klasöre (DSA_PROJE_ML) çıkar ve oradan Fixed Data klasörüne gider
PROJE_ANA_DIZIN = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJE_ANA_DIZIN, "Fixed Data", "AA_Makine_Ogrenmesi_Hazir_Tum_Veri_YENI.csv")

df = pd.read_csv(CSV_PATH)
df = df.sort_values(by=['Lokasyon Adı', 'hafta_indeksi']).reset_index(drop=True)

# Season mapping: 1=Winter, 2=Spring, 3=Summer, 4=Autumn
mevsim_map = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
              6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
df['mevsim'] = df['ay'].map(mevsim_map)
mevsim_isim = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}

# Combine region column (from one-hot to a single column)
def bolge_belirle(row):
    if row['Bölge_Batı Karadeniz'] == 1: return 'West'
    if row['Bölge_Orta Karadeniz'] == 1: return 'Central'
    if row['Bölge_Doğu Karadeniz'] == 1: return 'East'
    return 'Unknown'
df['bolge'] = df.apply(bolge_belirle, axis=1)

print("="*78)
print("BLACK SEA CAMPSITES — HYPOTHESIS TESTS v2")
print("="*78)
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Number of locations: {df['Lokasyon Adı'].nunique()}")
print(f"Week range: {df['hafta_indeksi'].min()} - {df['hafta_indeksi'].max()}")
print("="*78)

# List to collect all results
tum_sonuclar = []


# ============================================================================
# TEST 1: PERMUTATION TEST — SEASON-CONTROLLED TEMPERATURE-VISITOR RELATIONSHIP
# ============================================================================
"""
GOAL:
  The main issue with the previous Pearson test was confounding.
  Both temperature and visitor counts increase towards summer — therefore, 
  even if there is no real "temperature → visitor" effect, a high correlation appears.

METHOD:
  We permute the temperature ONLY WITHIN THE SAME SEASON. Meaning:
  - Summer week temperatures are shuffled among themselves
  - Winter week temperatures are shuffled among themselves
  This way, the seasonal "summer=hot+crowded" pattern is preserved in the 
  null distribution. If our observed correlation is SIGNIFICANTLY higher 
  than the null distribution, it indicates a season-independent temperature effect.

H0: When the season is held constant, temperature and visitor count are independent.
H1: When the season is held constant, there is a relationship between temperature and visitor count.
"""

print("\n" + "="*78)
print("TEST 1 — PERMUTATION TEST: SEASON-CONTROLLED TEMPERATURE-VISITOR")
print("="*78)
print("""H0: When the season is held constant, there is no relationship between temperature and visitors.
H1: When the season is held constant, there is a relationship between temperature and visitors.
Method: Randomization test permuting temperature within seasons (10000 iter).
Statistic: Spearman correlation (ordinal, robust to skewness).""")

N_PERM = 10000
test1_sonuclari = []

for lokasyon in df['Lokasyon Adı'].unique():
    sub = df[df['Lokasyon Adı'] == lokasyon][['temp', 'gercek_ziyaretci', 'mevsim']].dropna()
    if len(sub) < 20:
        continue

    # Observed Spearman correlation
    obs_corr, _ = spearmanr(sub['temp'], sub['gercek_ziyaretci'])

    # Permutation: shuffle temperature within the season
    perm_corrs = np.zeros(N_PERM)
    temp_values = sub['temp'].values.copy()
    mevsim_values = sub['mevsim'].values
    visit_values = sub['gercek_ziyaretci'].values

    for i in range(N_PERM):
        permuted_temp = temp_values.copy()
        for m in np.unique(mevsim_values):
            mask = mevsim_values == m
            permuted_in_mevsim = np.random.permutation(permuted_temp[mask])
            permuted_temp[mask] = permuted_in_mevsim
        perm_corrs[i], _ = spearmanr(permuted_temp, visit_values)

    # Two-sided p-value: |permuted correlations| ≥ |observed|
    p_value = np.mean(np.abs(perm_corrs) >= np.abs(obs_corr))

    test1_sonuclari.append({
        'Location': lokasyon,
        'Spearman_r': round(obs_corr, 4),
        'Permutation_p': round(p_value, 4),
        'Significant (p<0.05)': 'Yes' if p_value < 0.05 else 'No'
    })

df_test1 = pd.DataFrame(test1_sonuclari)

# FDR correction (Benjamini-Hochberg)
def benjamini_hochberg(pvals, alpha=0.05):
    """B-H FDR correction."""
    pvals = np.array(pvals)
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    adjusted = sorted_p * n / (np.arange(1, n + 1))
    # Monotonic correction (cumulative minimum, backwards)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    final = np.empty(n)
    final[sorted_idx] = adjusted
    return final

df_test1['FDR_p'] = benjamini_hochberg(df_test1['Permutation_p'].values).round(4)
df_test1['FDR Significant?'] = np.where(df_test1['FDR_p'] < 0.05, 'Yes', 'No')

print(df_test1.to_string(index=False))
print(f"\nSUMMARY: In {(df_test1['FDR Significant?']=='Yes').sum()}/{len(df_test1)} locations, "
      f"the season-independent temperature effect is significant after FDR correction.")

tum_sonuclar.append(("TEST 1: Permutation (Season-Controlled Temperature)", df_test1))


# ============================================================================
# TEST 2: MANN-WHITNEY U TEST — PRECIPITATION EFFECT ON VISITORS (FDR CORRECTED)
# ============================================================================
"""
GOAL:
  The previous Welch's t-test was not suitable for the skewed distribution of visitor data.
  Mann-Whitney U is a non-parametric, rank-based test that does not assume normality.
  Additionally, since we are doing 16 tests for 16 locations, a multiple comparisons 
  correction is required (Benjamini-Hochberg FDR).

METHOD:
  Compare the visitor medians of rainy weeks (prcp > 1.0 mm) and dry weeks 
  for each location. Then apply FDR to all p-values.

H0: The visitor distributions for rainy and dry weeks are the same.
H1: The visitor distributions for rainy and dry weeks are different.
"""

print("\n" + "="*78)
print("TEST 2 — MANN-WHITNEY U: PRECIPITATION EFFECT ON VISITORS (FDR CORRECTED)")
print("="*78)
print("""H0: Visitor distributions are the same in rainy/dry weeks.
H1: Visitor distributions are different in rainy/dry weeks.
Method: Mann-Whitney U test (non-parametric) + B-H FDR correction.""")

test2_sonuclari = []
for lokasyon in df['Lokasyon Adı'].unique():
    sub = df[df['Lokasyon Adı'] == lokasyon][['prcp', 'gercek_ziyaretci']].dropna()
    yagisli = sub[sub['prcp'] > 1.0]['gercek_ziyaretci']
    yagissiz = sub[sub['prcp'] <= 1.0]['gercek_ziyaretci']

    if len(yagisli) < 5 or len(yagissiz) < 5:
        continue

    u_stat, p_val = mannwhitneyu(yagissiz, yagisli, alternative='two-sided')

    test2_sonuclari.append({
        'Location': lokasyon,
        'n_dry': len(yagissiz),
        'n_rainy': len(yagisli),
        'Median_dry': int(yagissiz.median()),
        'Median_rainy': int(yagisli.median()),
        'U_statistic': round(u_stat, 1),
        'Raw_p': round(p_val, 4)
    })

df_test2 = pd.DataFrame(test2_sonuclari)
df_test2['FDR_p'] = benjamini_hochberg(df_test2['Raw_p'].values).round(4)
df_test2['FDR Significant?'] = np.where(df_test2['FDR_p'] < 0.05, 'Yes', 'No')

print(df_test2.to_string(index=False))
print(f"\nSUMMARY: The precipitation effect is significant in {(df_test2['Raw_p']<0.05).sum()}/{len(df_test2)} locations before FDR correction, "
      f"and in {(df_test2['FDR Significant?']=='Yes').sum()}/{len(df_test2)} locations after.")

tum_sonuclar.append(("TEST 2: Mann-Whitney U (Precipitation-Visitor)", df_test2))


# ============================================================================
# TEST 3: KRUSKAL-WALLIS — SEASON EFFECT
# ============================================================================
"""
GOAL:
  The ML model uses the season variable as a feature. We need to prove that 
  this feature carries statistically significant information.
  Kruskal-Wallis is the non-parametric version of ANOVA — it tests whether 
  there is a median difference across 3+ groups.

METHOD:
  Are visitor medians different across the 4 season groups for each location?
  Also, a global test is conducted over the entire dataset.

H0: The visitor medians across the four seasons are equal.
H1: At least one pair of seasons has a different visitor median.
"""

print("\n" + "="*78)
print("TEST 3 — KRUSKAL-WALLIS: SEASON EFFECT ON VISITORS")
print("="*78)
print("""H0: Visitor medians are equal across the four seasons.
H1: At least one pair of seasons has a different visitor median.
Method: Kruskal-Wallis H test (non-parametric version of ANOVA).""")

test3_sonuclari = []
for lokasyon in df['Lokasyon Adı'].unique():
    sub = df[df['Lokasyon Adı'] == lokasyon][['mevsim', 'gercek_ziyaretci']].dropna()
    gruplar = [sub[sub['mevsim'] == m]['gercek_ziyaretci'].values
               for m in [1, 2, 3, 4] if (sub['mevsim'] == m).sum() >= 5]

    if len(gruplar) < 2:
        continue

    h_stat, p_val = kruskal(*gruplar)
    medyanlar = {mevsim_isim[m]: int(sub[sub['mevsim']==m]['gercek_ziyaretci'].median())
                 for m in [1,2,3,4] if (sub['mevsim']==m).sum() >= 5}

    test3_sonuclari.append({
        'Location': lokasyon,
        'H_statistic': round(h_stat, 2),
        'p_value': round(p_val, 4),
        'Significant?': 'Yes' if p_val < 0.05 else 'No',
        'Med_Winter': medyanlar.get('Winter', '-'),
        'Med_Spring': medyanlar.get('Spring', '-'),
        'Med_Summer': medyanlar.get('Summer', '-'),
        'Med_Autumn': medyanlar.get('Autumn', '-')
    })

df_test3 = pd.DataFrame(test3_sonuclari)
df_test3['FDR_p'] = benjamini_hochberg(df_test3['p_value'].values).round(4)
df_test3['FDR Significant?'] = np.where(df_test3['FDR_p'] < 0.05, 'Yes', 'No')

print(df_test3.to_string(index=False))

# Global Kruskal-Wallis (combining all locations)
print("\n--- GLOBAL KRUSKAL-WALLIS (Entire dataset) ---")
gruplar_global = [df[df['mevsim']==m]['gercek_ziyaretci'].values for m in [1,2,3,4]]
h_global, p_global = kruskal(*gruplar_global)
print(f"H = {h_global:.2f}, p = {p_global:.4e}")
print(f"Result: {'H0 rejected — the season effect is GLOBALLY significant.' if p_global<0.05 else 'H0 could not be rejected.'}")

tum_sonuclar.append(("TEST 3: Kruskal-Wallis (Season Effect)", df_test3))


# ============================================================================
# TEST 4: CHI-SQUARE TEST OF INDEPENDENCE — REGION × LOW VISITORS IN RAIN
# ============================================================================
"""
GOAL:
  The previous Chi-square test was done with n=4 (assumption violation).
  This new version works on a week-location basis (n>3000).
  Question: Is there a relationship between the region and having low visitors 
  during a rainy week?

METHOD:
  Two categorical variables are created for each week-location record:
  - Region: West / Central / East Black Sea
  - Rainy low visitor: If the week is rainy (>1mm) AND the visitors are below 
    the median for that location, it's "Yes", otherwise "No"
  Then a Chi-square test of independence with a 3x2 cross-tabulation.

H0: Region and visitor-drop-during-rain are independent (no relationship).
H1: There is a relationship between region and visitor-drop-during-rain.
"""

print("\n" + "="*78)
print("TEST 4 — CHI-SQUARE INDEPENDENCE: REGION × LOW VISITORS IN RAIN")
print("="*78)
print("""H0: Region and 'low visitors during rainy week' are independent.
H1: There is a relationship between Region and 'low visitors during rainy week'.
Method: Chi-square test of independence (chi2_contingency), n=week-location records.""")

# Calculate median for each location, then apply the "rainy + low visitor" flag
df['lok_medyan'] = df.groupby('Lokasyon Adı')['gercek_ziyaretci'].transform('median')
df['yagisli_dusuk'] = np.where(
    (df['prcp'] > 1.0) & (df['gercek_ziyaretci'] < df['lok_medyan']),
    'Yes', 'No'
)

# Cross tabulation (Region × yagisli_dusuk)
capraz_tablo = pd.crosstab(df['bolge'], df['yagisli_dusuk'])
print("\n--- CROSS TABULATION ---")
print(capraz_tablo)

chi2_stat, p_chi2, dof, expected = chi2_contingency(capraz_tablo)
print(f"\n--- TEST RESULT ---")
print(f"Chi-Square statistic: {chi2_stat:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_chi2:.4e}")
print(f"Minimum of expected frequencies: {expected.min():.1f} (Should be ≥5: "
      f"{'✓ SATISFIED' if expected.min()>=5 else '✗ VIOLATION'})")

# Effect size: Cramér's V
n_total = capraz_tablo.values.sum()
cramers_v = np.sqrt(chi2_stat / (n_total * (min(capraz_tablo.shape) - 1)))
print(f"Cramér's V (effect size): {cramers_v:.4f} "
      f"({'small' if cramers_v<0.1 else 'medium' if cramers_v<0.3 else 'large'} effect)")

if p_chi2 < 0.05:
    print("\nResult: H0 REJECTED. There is a statistically significant RELATIONSHIP "
          "between the region and having low visitors during rain.")
else:
    print("\nResult: H0 could not be rejected. Region and sensitivity to rain are INDEPENDENT.")

df_test4 = pd.DataFrame({
    'Metric': ['Chi2 Statistic', 'Degrees of Freedom', 'p-value',
               'Cramér V', 'Min Expected Frequency', 'Result'],
    'Value': [round(chi2_stat,4), dof, f"{p_chi2:.4e}",
              round(cramers_v,4), round(expected.min(),1),
              'H0 Rejected' if p_chi2<0.05 else 'H0 Not Rejected']
})
tum_sonuclar.append(("TEST 4: Chi-Square (Region × Precipitation Sensitivity)", df_test4))
tum_sonuclar.append(("TEST 4: Cross Tabulation", capraz_tablo.reset_index()))


# ============================================================================
# TEST 5: PERMUTATION TEST — IMPORTANCE OF WEATHER IN THE ML MODEL
# ============================================================================
"""
GOAL:
  This test has a DIRECT contribution to the final goal of your project (visitor prediction).
  Question: Do weather features (temp, prcp, snow, rain, wspd, rhum) make a 
  statistically significant contribution to the predictive power of the ML model?

METHOD:
  1. Train the full model (with all features) → R²_real
  2. RANDOMLY permute the weather features (breaking their relationships)
  3. Retrain the model with this new data → R²_permuted
  4. Repeat 100 times → build a null distribution
  5. p-value = ratio of permutations where (R²_permuted >= R²_real)

H0: Weather features do not contribute to the model's performance.
H1: Weather features make a significant contribution to the model's performance.
"""

print("\n" + "="*78)
print("TEST 5 — PERMUTATION: IMPORTANCE OF WEATHER FEATURES IN ML MODEL")
print("="*78)
print("""H0: Weather features do not contribute to the model's predictive power.
H1: Weather features make a significant contribution to the model's predictive power.
Method: Build a null distribution by permuting weather columns together, 
        compare with the real R².""")

# Prepare dataset — only leakage-free features
WEATHER_COLS = ['temp', 'prcp', 'snow', 'rain', 'wspd', 'rhum']
LOCATION_COLS = ['Rakım (m)', 'Has. Mesafe (km)', 'Ort. Eğim (%)',
                 'Has. Varış Süresi (Dk)']
TIME_COLS = ['ay', 'hafta_indeksi', 'tatil_mi']

# Encode location numerically
df['lok_kod'] = pd.Categorical(df['Lokasyon Adı']).codes

feature_cols = WEATHER_COLS + LOCATION_COLS + TIME_COLS + ['lok_kod']
data_ml = df[feature_cols + ['gercek_ziyaretci']].dropna().reset_index(drop=True)

X = data_ml[feature_cols].values
y = np.log1p(data_ml['gercek_ziyaretci'].values)  # log transform (skewness)

# Train/test split: year-based (last ~20% test)
split_idx = int(len(data_ml) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 1) Real model performance
rf_real = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
rf_real.fit(X_train, y_train)
r2_real = r2_score(y_test, rf_real.predict(X_test))
print(f"\nReal model R² (test set): {r2_real:.4f}")

# 2) Permutation — shuffle all weather columns together
weather_idx = [feature_cols.index(c) for c in WEATHER_COLS]
N_PERM_ML = 100  # 100 is sufficient since ML training is costly
perm_r2s = []

print(f"Permutation starting ({N_PERM_ML} iterations)...")
for i in range(N_PERM_ML):
    X_train_perm = X_train.copy()
    # Randomly shuffle weather rows within train data
    perm_indices = np.random.permutation(X_train.shape[0])
    X_train_perm[:, weather_idx] = X_train[perm_indices][:, weather_idx]

    rf_perm = RandomForestRegressor(n_estimators=100, random_state=i, n_jobs=-1)
    rf_perm.fit(X_train_perm, y_train)
    perm_r2s.append(r2_score(y_test, rf_perm.predict(X_test)))

    if (i+1) % 20 == 0:
        print(f"  ... {i+1}/{N_PERM_ML} permutations completed")

perm_r2s = np.array(perm_r2s)
p_value_ml = np.mean(perm_r2s >= r2_real)

print(f"\n--- TEST RESULT ---")
print(f"Real R²:              {r2_real:.4f}")
print(f"Permutation R² mean:  {perm_r2s.mean():.4f}")
print(f"Permutation R² std:   {perm_r2s.std():.4f}")
print(f"Permutation R² max:   {perm_r2s.max():.4f}")
print(f"R² difference (effect):{r2_real - perm_r2s.mean():.4f}")
print(f"p-value:              {p_value_ml:.4f}")

if p_value_ml < 0.05:
    print("\nResult: H0 REJECTED. Weather features make a STATISTICALLY SIGNIFICANT "
          "contribution to the model's predictive power.")
else:
    print("\nResult: H0 could not be rejected.")

df_test5 = pd.DataFrame({
    'Metric': ['Real R²', 'Permutation R² mean', 'Permutation R² std',
               'Permutation R² max', 'Effect size (ΔR²)',
               'Permutation p-value', 'Result'],
    'Value': [round(r2_real,4), round(perm_r2s.mean(),4),
              round(perm_r2s.std(),4), round(perm_r2s.max(),4),
              round(r2_real - perm_r2s.mean(),4), round(p_value_ml,4),
              'H0 Rejected' if p_value_ml<0.05 else 'H0 Not Rejected']
})
tum_sonuclar.append(("TEST 5: Permutation (ML Weather Importance)", df_test5))


# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================
print("\n" + "="*78)
print("SAVING RESULTS TO CSV FILES")
print("="*78)

ozet_yolu = os.path.join(SCRIPT_DIR, "Hypothesis_Results_Summary.csv")
with open(ozet_yolu, 'w', encoding='utf-8-sig') as f:
    for baslik, tablo in tum_sonuclar:
        f.write(f"\n{'='*70}\n{baslik}\n{'='*70}\n")
        tablo.to_csv(f, index=False)
        f.write("\n")

# Individual CSVs as well
df_test1.to_csv(os.path.join(SCRIPT_DIR, "Test1_Permutation_Temperature.csv"),
                index=False, encoding='utf-8-sig')
df_test2.to_csv(os.path.join(SCRIPT_DIR, "Test2_MannWhitney_Precipitation.csv"),
                index=False, encoding='utf-8-sig')
df_test3.to_csv(os.path.join(SCRIPT_DIR, "Test3_KruskalWallis_Season.csv"),
                index=False, encoding='utf-8-sig')
df_test4.to_csv(os.path.join(SCRIPT_DIR, "Test4_ChiSquare_Region.csv"),
                index=False, encoding='utf-8-sig')
df_test5.to_csv(os.path.join(SCRIPT_DIR, "Test5_Permutation_ML.csv"),
                index=False, encoding='utf-8-sig')

print("✓ Hypothesis_Results_Summary.csv (all tests in one file)")
print("✓ Test1_Permutation_Temperature.csv")
print("✓ Test2_MannWhitney_Precipitation.csv")
print("✓ Test3_KruskalWallis_Season.csv")
print("✓ Test4_ChiSquare_Region.csv")
print("✓ Test5_Permutation_ML.csv")

print("\n" + "="*78)
print("ALL HYPOTHESIS TESTS COMPLETED ✓")
print("="*78)