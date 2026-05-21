"""
==============================================================
  Campsite Visitor Prediction Model — TRAINING SCRIPT
==============================================================
Production-adapted version of the original ml_v2 code.
Trains the model and saves it to a pickle with critical auxiliary data.
The Flask application loads and uses these saved artifacts.

Run: python train_model.py
Output: models/production_artifact.pkl
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "data", "AA_Makine_Ogrenmesi_Hazir_Tum_Veri_YENI.csv")
MODEL_OUT  = os.path.join(SCRIPT_DIR, "models", "production_artifact.pkl")

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
print("="*70)
print("CAMPSITE VISITOR PREDICTION MODEL — TRAINING")
print("="*70)

df = pd.read_csv(CSV_PATH)
df = df.sort_values(by=['Lokasyon Adı', 'hafta_indeksi']).reset_index(drop=True)

LEAKY_COLS = ['Ziyaretci_2025', 'uygulanan_beta']
df = df.drop(columns=LEAKY_COLS, errors='ignore')
print(f"✓ Leaky columns removed: {LEAKY_COLS}")

df['ortalama_puan'] = df.groupby('Lokasyon Adı')['ortalama_puan'].transform(
    lambda x: x.fillna(x.mean())
)
df['ortalama_puan'] = df['ortalama_puan'].fillna(df['ortalama_puan'].mean())

all_lokasyonlar = df['Lokasyon Adı'].unique().tolist()
print(f"✓ {len(all_lokasyonlar)} locations loaded")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df['yaklasik_yil']  = ((df['hafta_indeksi'] - 1) // 52) + 2022
df['yil_ici_hafta'] = ((df['hafta_indeksi'] - 1) % 52) + 1

mevsim_map = {12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4}
df['mevsim']    = df['ay'].map(mevsim_map)
df['ay_sin']    = np.sin(2*np.pi*df['ay']/12)
df['ay_cos']    = np.cos(2*np.pi*df['ay']/12)
df['hafta_sin'] = np.sin(2*np.pi*df['yil_ici_hafta']/52)
df['hafta_cos'] = np.cos(2*np.pi*df['yil_ici_hafta']/52)

df['temp_kare']       = df['temp']**2
df['sicak_yagmur']    = df['temp']*df['rain']
df['kar_soguk']       = df['snow']*(df['temp']<0).astype(int)
df['sicaklik_konfor'] = -(df['temp']-20)**2
df['nem_sicaklik']    = df['rhum']*df['temp']/100
df['kotu_hava']       = df['rain']*df['wspd']

df['yil_trendi']      = df['yaklasik_yil']-2022
df['tatil_yaz']       = df['tatil_mi']*(df['mevsim']==3).astype(int)
df['tatil_ilkbahar']  = df['tatil_mi']*(df['mevsim']==2).astype(int)

# Location growth dynamics
def yillik_egim(row):
    yillar = np.array([0, 1, 2])
    log_ziy = np.log1p([row['Ziyaretci_2022'], row['Ziyaretci_2023'], row['Ziyaretci_2024']])
    return np.polyfit(yillar, log_ziy, 1)[0]

egim_map = df.groupby('Lokasyon Adı').first().apply(yillik_egim, axis=1)
df['buyume_egimi']   = df['Lokasyon Adı'].map(egim_map)
df['lokasyon_olcek'] = np.log1p(df['Ziyaretci_2024'])
df['buyume_x_trend'] = df['buyume_egimi']*df['yil_trendi']

df['yorum_rolling4'] = df.groupby('Lokasyon Adı')['yorum_sayisi'].transform(
    lambda x: x.shift(1).rolling(4).mean())
df['yorum_rolling8'] = df.groupby('Lokasyon Adı')['yorum_sayisi'].transform(
    lambda x: x.shift(1).rolling(8).mean())

# yillik_ort (leakage-free: only 2022-2024)
yillik_ort_map = (df[df['hafta_indeksi'] <= 157]
                  .groupby('Lokasyon Adı')['gercek_ziyaretci'].mean())
df['yillik_ort']      = df['Lokasyon Adı'].map(yillik_ort_map)
df['relative_demand'] = df['gercek_ziyaretci']/(df['yillik_ort']+1)
df['rd_log']          = np.log1p(df['relative_demand'])

print("✓ Basic feature engineering complete")

# ─────────────────────────────────────────────
# 3. SEASONAL PATTERN FEATURES (leakage-free)
# ─────────────────────────────────────────────
df = df.sort_values(['Lokasyon Adı', 'yil_ici_hafta', 'hafta_indeksi']).reset_index(drop=True)
gr = df.groupby(['Lokasyon Adı', 'yil_ici_hafta'])['relative_demand']
df['ayni_hafta_gecmis_ort'] = gr.transform(lambda x: x.shift(1).expanding().mean())
df['ayni_hafta_gecmis_std'] = gr.transform(lambda x: x.shift(1).expanding().std()).fillna(0.0)

df = df.sort_values(['Lokasyon Adı', 'hafta_indeksi']).reset_index(drop=True)
df['son_yil_ayni_hafta'] = df.groupby('Lokasyon Adı')['relative_demand'].shift(52)

def komsu_hesapla(grup_lok):
    grup_lok = grup_lok.sort_values('hafta_indeksi').copy()
    sonuc = np.full(len(grup_lok), np.nan)
    yihs = grup_lok['yil_ici_hafta'].values
    rds  = grup_lok['relative_demand'].values
    his  = grup_lok['hafta_indeksi'].values
    for i in range(len(grup_lok)):
        h, yih = his[i], yihs[i]
        komsular = {((yih-2)%52)+1, yih, (yih%52)+1}
        mask = (his < h) & np.isin(yihs, list(komsular))
        if mask.any():
            sonuc[i] = rds[mask].mean()
    return pd.Series(sonuc, index=grup_lok.index)

df['komsu_hafta_ort'] = df.groupby('Lokasyon Adı', group_keys=False).apply(komsu_hesapla)
df = df.sort_values(['Lokasyon Adı', 'hafta_indeksi']).reset_index(drop=True)

# acik_orani_gecmis (winter closure signal)
df = df.sort_values(['Lokasyon Adı', 'yil_ici_hafta', 'hafta_indeksi']).reset_index(drop=True)
gr_acik = df.groupby(['Lokasyon Adı', 'yil_ici_hafta'])['gercek_ziyaretci']
df['acik_orani_gecmis'] = gr_acik.transform(
    lambda x: (x.shift(1) > 0).astype(float).expanding().mean())
df = df.sort_values(['Lokasyon Adı', 'hafta_indeksi']).reset_index(drop=True)
print("✓ Seasonal pattern + acik_orani_gecmis complete")

# ─────────────────────────────────────────────
# 4. TRAIN/TEST + FEATURE LIST
# ─────────────────────────────────────────────
required_cols = ['yorum_rolling4', 'yorum_rolling8',
                 'ayni_hafta_gecmis_ort', 'son_yil_ayni_hafta', 'komsu_hafta_ort',
                 'acik_orani_gecmis']

train_df = df[df['hafta_indeksi'] <= 157].copy()
test_df  = df[df['hafta_indeksi'] >  157].copy()
train_lag = train_df.dropna(subset=required_cols)
test_lag  = test_df.dropna(subset=required_cols)

print(f"\nTrain: {len(train_lag)} rows | Test: {len(test_lag)} rows (2025)")

yol_bolge_cols = [c for c in df.columns if c.startswith('Yol Türü') or c.startswith('Bölge')]
feature_cols = [
    'temp','prcp','snow','rain','wspd','rhum',
    'tatil_mi','ay','mevsim','ay_sin','ay_cos','hafta_sin','hafta_cos','yil_ici_hafta',
    'temp_kare','sicak_yagmur','kar_soguk','sicaklik_konfor','nem_sicaklik','kotu_hava',
    'yil_trendi','buyume_egimi','lokasyon_olcek','buyume_x_trend',
    'tatil_yaz','tatil_ilkbahar',
    'Rakım (m)','Has. Mesafe (km)','Ort. Eğim (%)','Has. Varış Süresi (Dk)','ortalama_puan',
    'yorum_rolling4','yorum_rolling8',
    'ayni_hafta_gecmis_ort','ayni_hafta_gecmis_std','son_yil_ayni_hafta','komsu_hafta_ort',
    'acik_orani_gecmis',
] + yol_bolge_cols

X_train = train_lag[feature_cols]
X_test  = test_lag[feature_cols]
y_train_log = train_lag['rd_log']
y_test_log  = test_lag['rd_log']

# ─────────────────────────────────────────────
# 5. HYPERPARAMETER OPTIMIZATION
# ─────────────────────────────────────────────
print("\nHyperparameter search (20 combinations × 3 folds)...")
param_dist = {
    'n_estimators':      [300, 500, 700],
    'max_depth':         [6, 8, 10, 12, None],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf':  [3, 5, 8, 12],
    'max_features':      [0.4, 0.55, 0.7],
}
search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_dist, n_iter=20,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='r2', random_state=42, n_jobs=-1, verbose=0)
search.fit(X_train, y_train_log)
best_params = search.best_params_
print(f"✓ Best parameters: {best_params}")
print(f"  CV R² (log): {search.best_score_:.4f}")

# ─────────────────────────────────────────────
# 6. FEATURE SELECTION
# ─────────────────────────────────────────────
final_params = dict(best_params)
final_params['random_state'] = 42
final_params['n_jobs'] = -1

prelim_rf = RandomForestRegressor(**final_params)
prelim_rf.fit(X_train, y_train_log)
prelim_imp = pd.Series(prelim_rf.feature_importances_, index=feature_cols)
selected_features = prelim_imp[prelim_imp >= 0.005].index.tolist()
print(f"\n✓ Feature selection: {len(feature_cols)} → {len(selected_features)}")

# ─────────────────────────────────────────────
# 7. EVAL MODEL (for performance, reporting to user)
# ─────────────────────────────────────────────
X_train_sel = train_lag[selected_features]
X_test_sel  = test_lag[selected_features]

final_params['oob_score'] = True
rf_eval = RandomForestRegressor(**final_params)
rf_eval.fit(X_train_sel, y_train_log)

# Growth scaling (for eval)
REFERENCE_YEAR_EVAL = 2023.0
MAX_YEARLY_GROWTH   = 0.15

egim_test = np.clip(test_lag['buyume_egimi'].values, -MAX_YEARLY_GROWTH, MAX_YEARLY_GROWTH)
test_lag = test_lag.copy()
test_lag['buyume_carpani_eval'] = np.exp(egim_test*(test_lag['yaklasik_yil'].values-REFERENCE_YEAR_EVAL))
test_lag['yillik_ort_proj']     = test_lag['yillik_ort']*test_lag['buyume_carpani_eval']

test_rel_pred = np.expm1(rf_eval.predict(X_test_sel))
test_abs_pred = np.clip(test_rel_pred*test_lag['yillik_ort_proj'].values, 0, None)

# acik_orani mask
def acik_maskesi(acik_orani):
    return np.clip((acik_orani-0.1)/0.4, 0, 1)
test_abs_pred *= acik_maskesi(test_lag['acik_orani_gecmis'].values)

test_r2  = r2_score(test_lag['gercek_ziyaretci'], test_abs_pred)
test_mae = mean_absolute_error(test_lag['gercek_ziyaretci'], test_abs_pred)
print(f"\n--- EVAL MODEL (2025 test) ---")
print(f"  Test R²:  {test_r2:.4f}")
print(f"  Test MAE: {test_mae:,.0f} people")
print(f"  OOB R²:   {rf_eval.oob_score_:.4f}")

# ─────────────────────────────────────────────
# 8. PRODUCTION MODEL (all 2022-2025)
# ─────────────────────────────────────────────
print("\nTraining production model (all 2022-2025 data)...")
production_data = df.dropna(subset=required_cols).copy()
X_production = production_data[selected_features]
y_production_log = production_data['rd_log']

rf_production = RandomForestRegressor(**final_params)
rf_production.fit(X_production, y_production_log)
print(f"✓ Production model trained ({len(X_production)} rows)")
print(f"  OOB R² (log): {rf_production.oob_score_:.4f}")

# yillik_ort for production: 2022-2025 average
yillik_ort_map_production = df.groupby('Lokasyon Adı')['gercek_ziyaretci'].mean().to_dict()

# Seasonal table (last valid value for location × yil_ici_hafta)
mevsimsel_tablo = (df.dropna(subset=['ayni_hafta_gecmis_ort'])
                     .groupby(['Lokasyon Adı', 'yil_ici_hafta'])
                     .agg({'ayni_hafta_gecmis_ort':'last',
                           'ayni_hafta_gecmis_std':'last',
                           'son_yil_ayni_hafta':'last',
                           'komsu_hafta_ort':'last',
                           'acik_orani_gecmis':'last'})
                     .reset_index())

# Location references (static features of each location)
lokasyon_ref_cols = ['buyume_egimi', 'lokasyon_olcek',
                     'Rakım (m)', 'Has. Mesafe (km)', 'Ort. Eğim (%)',
                     'Has. Varış Süresi (Dk)', 'ortalama_puan'] + yol_bolge_cols
lokasyon_ref = df.groupby('Lokasyon Adı').first()[lokasyon_ref_cols].to_dict('index')

# Average number of reviews per location
yorum_ort_map = df.groupby('Lokasyon Adı')['yorum_sayisi'].mean().to_dict()

# ─────────────────────────────────────────────
# 9. ARTIFACT SAVING
# ─────────────────────────────────────────────
artifact = {
    'model': rf_production,
    'selected_features': selected_features,
    'mevsimsel_tablo': mevsimsel_tablo,
    'yillik_ort_map': yillik_ort_map_production,
    'lokasyon_ref': lokasyon_ref,
    'yorum_ort_map': yorum_ort_map,
    'yol_bolge_cols': yol_bolge_cols,
    'all_lokasyonlar': all_lokasyonlar,
    'mevsim_map': mevsim_map,
    'config': {
        'REFERENCE_YEAR': 2023.5,
        'MAX_YEARLY_GROWTH': 0.15,
    },
    'eval_metrics': {
        'test_r2':  float(test_r2),
        'test_mae': float(test_mae),
        'oob_r2':   float(rf_eval.oob_score_),
        'n_train':  int(len(X_train_sel)),
        'n_test':   int(len(X_test_sel)),
        'best_params': {k: (v if not isinstance(v, np.generic) else v.item())
                        for k, v in best_params.items()},
    },
}

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
with open(MODEL_OUT, 'wb') as f:
    pickle.dump(artifact, f)

print(f"\n{'='*70}")
print(f"✓ Model artifact saved: {MODEL_OUT}")
print(f"  Size: {os.path.getsize(MODEL_OUT)/1024:.1f} KB")
print(f"{'='*70}")
