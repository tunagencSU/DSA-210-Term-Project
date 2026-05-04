"""
============================================================================
KARADENİZ KAMP ALANLARI - GELİŞTİRİLMİŞ HİPOTEZ TESTLERİ
============================================================================

Bu script, ziyaretçi tahmin modelinin temel varsayımlarını istatistiksel olarak
doğrulamak için 5 farklı hipotez testi uygular:

  TEST 1: Permütasyon Testi (Mevsim Kontrollü Sıcaklık-Ziyaretçi İlişkisi)
  TEST 2: Mann-Whitney U + FDR Düzeltmesi (Yağış-Ziyaretçi Etkisi)
  TEST 3: Kruskal-Wallis Testi (Mevsim Etkisi)
  TEST 4: Ki-Kare Bağımsızlık Testi (Bölge × Yağış Duyarlılığı)
  TEST 5: Permütasyon Testi (ML Modelinde Hava Durumunun Önemi)

İZİN VERİLEN TESTLERDEN KULLANILANLAR:
  A. Randomization (Permütasyon) Test  → TEST 1, TEST 5
  D. Chi-Square Test of Independence   → TEST 4
  K. Mann-Whitney U Testi              → TEST 2
  T. Kruskal-Wallis Testi              → TEST 3
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

# Tekrarlanabilirlik için seed
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 0. VERİ YÜKLEME
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "AA_Makine_Ogrenmesi_Hazir_Tum_Veri_YENI.csv")

df = pd.read_csv(CSV_PATH)
df = df.sort_values(by=['Lokasyon Adı', 'hafta_indeksi']).reset_index(drop=True)

# Mevsim eşlemesi: 1=Kış, 2=İlkbahar, 3=Yaz, 4=Sonbahar
mevsim_map = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
              6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
df['mevsim'] = df['ay'].map(mevsim_map)
mevsim_isim = {1: 'Kış', 2: 'İlkbahar', 3: 'Yaz', 4: 'Sonbahar'}

# Bölge sütununu birleştir (one-hot'tan tek sütuna)
def bolge_belirle(row):
    if row['Bölge_Batı Karadeniz'] == 1: return 'Batı'
    if row['Bölge_Orta Karadeniz'] == 1: return 'Orta'
    if row['Bölge_Doğu Karadeniz'] == 1: return 'Doğu'
    return 'Bilinmiyor'
df['bolge'] = df.apply(bolge_belirle, axis=1)

print("="*78)
print("KARADENİZ KAMP ALANLARI — HİPOTEZ TESTLERİ v2")
print("="*78)
print(f"Veri seti: {df.shape[0]} satır × {df.shape[1]} sütun")
print(f"Lokasyon sayısı: {df['Lokasyon Adı'].nunique()}")
print(f"Hafta aralığı: {df['hafta_indeksi'].min()} - {df['hafta_indeksi'].max()}")
print("="*78)

# Tüm sonuçları toplayacağımız liste
tum_sonuclar = []


# ============================================================================
# TEST 1: PERMÜTASYON TESTİ — MEVSİM KONTROLLÜ SICAKLIK-ZİYARETÇİ İLİŞKİSİ
# ============================================================================
"""
AMAÇ:
  Önceki Pearson testinin temel sorunu confounding (gizli karıştırıcı) idi.
  Sıcaklık ve ziyaretçi sayısı ikisi de yaza doğru artar — bu yüzden gerçek
  bir "sıcaklık → ziyaretçi" etkisi olmasa bile yüksek korelasyon görünür.

YÖNTEM:
  Sıcaklığı sadece AYNI MEVSİM İÇİNDE permüte ediyoruz. Yani:
  - Yaz haftalarının sıcaklıkları kendi aralarında karıştırılıyor
  - Kış haftalarının sıcaklıkları kendi aralarında karıştırılıyor
  Bu şekilde mevsimsel "yaz=sıcak+kalabalık" örüntüsü null dağılımda da
  korunmuş olur. Eğer gerçek korelasyonumuz null dağılımdan ANLAMLI ŞEKİLDE
  yüksekse, bu mevsimden bağımsız bir sıcaklık etkisi olduğunu gösterir.

H0: Mevsim sabit tutulduğunda sıcaklık ile ziyaretçi sayısı bağımsızdır.
H1: Mevsim sabit tutulduğunda sıcaklık ile ziyaretçi sayısı arasında ilişki vardır.
"""

print("\n" + "="*78)
print("TEST 1 — PERMÜTASYON TESTİ: MEVSİM KONTROLLÜ SICAKLIK-ZİYARETÇİ")
print("="*78)
print("""H0: Mevsim sabit tutulduğunda sıcaklık ile ziyaretçi arasında ilişki yoktur.
H1: Mevsim sabit tutulduğunda sıcaklık ile ziyaretçi arasında ilişki vardır.
Yöntem: Sıcaklığı mevsim içinde permüte eden randomization test (10000 iter).
İstatistik: Spearman korelasyonu (sıralı, çarpıklığa dayanıklı).""")

N_PERM = 10000
test1_sonuclari = []

for lokasyon in df['Lokasyon Adı'].unique():
    sub = df[df['Lokasyon Adı'] == lokasyon][['temp', 'gercek_ziyaretci', 'mevsim']].dropna()
    if len(sub) < 20:
        continue

    # Gözlenen Spearman korelasyonu
    obs_corr, _ = spearmanr(sub['temp'], sub['gercek_ziyaretci'])

    # Permütasyon: sıcaklığı mevsim içinde karıştır
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

    # İki yönlü p-değeri: |permüte korelasyonlar| ≥ |gözlenen|
    p_value = np.mean(np.abs(perm_corrs) >= np.abs(obs_corr))

    test1_sonuclari.append({
        'Lokasyon': lokasyon,
        'Spearman_r': round(obs_corr, 4),
        'Permutasyon_p': round(p_value, 4),
        'Anlamlı (p<0.05)': 'Evet' if p_value < 0.05 else 'Hayır'
    })

df_test1 = pd.DataFrame(test1_sonuclari)

# FDR düzeltmesi (Benjamini-Hochberg)
def benjamini_hochberg(pvals, alpha=0.05):
    """B-H FDR düzeltmesi."""
    pvals = np.array(pvals)
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    adjusted = sorted_p * n / (np.arange(1, n + 1))
    # Monotonik düzeltme (kümülatif minimum, sondan başa)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    final = np.empty(n)
    final[sorted_idx] = adjusted
    return final

df_test1['FDR_p'] = benjamini_hochberg(df_test1['Permutasyon_p'].values).round(4)
df_test1['FDR Anlamlı?'] = np.where(df_test1['FDR_p'] < 0.05, 'Evet', 'Hayır')

print(df_test1.to_string(index=False))
print(f"\nÖZET: {(df_test1['FDR Anlamlı?']=='Evet').sum()}/{len(df_test1)} lokasyonda "
      f"FDR düzeltmesi sonrası mevsimden bağımsız sıcaklık etkisi anlamlı.")

tum_sonuclar.append(("TEST 1: Permütasyon (Mevsim Kontrollü Sıcaklık)", df_test1))


# ============================================================================
# TEST 2: MANN-WHITNEY U TESTİ — YAĞIŞIN ZİYARETÇİYE ETKİSİ (FDR DÜZELTMELİ)
# ============================================================================
"""
AMAÇ:
  Önceki Welch's t-test ziyaretçi verisinin çarpık dağılımı için uygun değildi.
  Mann-Whitney U normallik gerektirmeyen, sıralamaya dayalı bir testtir.
  Ayrıca 16 lokasyon = 16 test yaptığımız için çoklu karşılaştırma düzeltmesi
  şart (Benjamini-Hochberg FDR).

YÖNTEM:
  Her lokasyon için yağışlı haftalar (prcp > 1.0 mm) ile yağışsız haftaların
  ziyaretçi medyanlarını karşılaştır. Sonra tüm p-değerlerine FDR uygula.

H0: Yağışlı ve yağışsız haftaların ziyaretçi dağılımları aynıdır.
H1: Yağışlı ve yağışsız haftaların ziyaretçi dağılımları farklıdır.
"""

print("\n" + "="*78)
print("TEST 2 — MANN-WHITNEY U: YAĞIŞIN ZİYARETÇİYE ETKİSİ (FDR DÜZELTMELİ)")
print("="*78)
print("""H0: Yağışlı/yağışsız haftalarda ziyaretçi dağılımları aynıdır.
H1: Yağışlı/yağışsız haftalarda ziyaretçi dağılımları farklıdır.
Yöntem: Mann-Whitney U testi (parametrik olmayan) + B-H FDR düzeltmesi.""")

test2_sonuclari = []
for lokasyon in df['Lokasyon Adı'].unique():
    sub = df[df['Lokasyon Adı'] == lokasyon][['prcp', 'gercek_ziyaretci']].dropna()
    yagisli = sub[sub['prcp'] > 1.0]['gercek_ziyaretci']
    yagissiz = sub[sub['prcp'] <= 1.0]['gercek_ziyaretci']

    if len(yagisli) < 5 or len(yagissiz) < 5:
        continue

    u_stat, p_val = mannwhitneyu(yagissiz, yagisli, alternative='two-sided')

    test2_sonuclari.append({
        'Lokasyon': lokasyon,
        'n_yağışsız': len(yagissiz),
        'n_yağışlı': len(yagisli),
        'Medyan_yağışsız': int(yagissiz.median()),
        'Medyan_yağışlı': int(yagisli.median()),
        'U_istatistik': round(u_stat, 1),
        'Ham_p': round(p_val, 4)
    })

df_test2 = pd.DataFrame(test2_sonuclari)
df_test2['FDR_p'] = benjamini_hochberg(df_test2['Ham_p'].values).round(4)
df_test2['FDR Anlamlı?'] = np.where(df_test2['FDR_p'] < 0.05, 'Evet', 'Hayır')

print(df_test2.to_string(index=False))
print(f"\nÖZET: FDR düzeltmesi öncesi {(df_test2['Ham_p']<0.05).sum()}/{len(df_test2)}, "
      f"sonrası {(df_test2['FDR Anlamlı?']=='Evet').sum()}/{len(df_test2)} "
      f"lokasyonda yağış etkisi anlamlı.")

tum_sonuclar.append(("TEST 2: Mann-Whitney U (Yağış-Ziyaretçi)", df_test2))


# ============================================================================
# TEST 3: KRUSKAL-WALLIS — MEVSİM ETKİSİ
# ============================================================================
"""
AMAÇ:
  ML modeli mevsim değişkenini bir özellik olarak kullanıyor. Bu özelliğin
  istatistiksel olarak anlamlı bir bilgi taşıdığını kanıtlamamız gerekiyor.
  Kruskal-Wallis, ANOVA'nın parametrik olmayan versiyonudur — 3+ grup
  arasında medyan farkı olup olmadığını test eder.

YÖNTEM:
  Her lokasyon için 4 mevsim grubu arasında ziyaretçi medyanı farklı mı?
  Ayrıca tüm veri üzerinde de global bir test yapılıyor.

H0: Dört mevsim arasında ziyaretçi medyanları eşittir.
H1: En az bir mevsim çiftinin ziyaretçi medyanı farklıdır.
"""

print("\n" + "="*78)
print("TEST 3 — KRUSKAL-WALLIS: MEVSİM ETKİSİ ZİYARETÇİ ÜZERİNE")
print("="*78)
print("""H0: Dört mevsim arasında ziyaretçi medyanları eşittir.
H1: En az bir mevsim çiftinde ziyaretçi medyanı farklıdır.
Yöntem: Kruskal-Wallis H testi (ANOVA'nın parametrik olmayan versiyonu).""")

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
        'Lokasyon': lokasyon,
        'H_istatistik': round(h_stat, 2),
        'p_değeri': round(p_val, 4),
        'Anlamlı?': 'Evet' if p_val < 0.05 else 'Hayır',
        'Med_Kış': medyanlar.get('Kış', '-'),
        'Med_İlkbahar': medyanlar.get('İlkbahar', '-'),
        'Med_Yaz': medyanlar.get('Yaz', '-'),
        'Med_Sonbahar': medyanlar.get('Sonbahar', '-')
    })

df_test3 = pd.DataFrame(test3_sonuclari)
df_test3['FDR_p'] = benjamini_hochberg(df_test3['p_değeri'].values).round(4)
df_test3['FDR Anlamlı?'] = np.where(df_test3['FDR_p'] < 0.05, 'Evet', 'Hayır')

print(df_test3.to_string(index=False))

# Global Kruskal-Wallis (tüm lokasyonları birleştirerek)
print("\n--- GLOBAL KRUSKAL-WALLIS (Tüm veri seti) ---")
gruplar_global = [df[df['mevsim']==m]['gercek_ziyaretci'].values for m in [1,2,3,4]]
h_global, p_global = kruskal(*gruplar_global)
print(f"H = {h_global:.2f}, p = {p_global:.4e}")
print(f"Sonuç: {'H0 reddedildi — mevsim etkisi GLOBAL olarak anlamlı.' if p_global<0.05 else 'H0 reddedilemedi.'}")

tum_sonuclar.append(("TEST 3: Kruskal-Wallis (Mevsim Etkisi)", df_test3))


# ============================================================================
# TEST 4: KI-KARE BAĞIMSIZLIK TESTİ — BÖLGE × YAĞIŞ DURUMUNDA DÜŞÜK ZİYARETÇİ
# ============================================================================
"""
AMAÇ:
  Önceki Ki-kare testi n=4 ile yapılmıştı (varsayım ihlali).
  Bu yeni versiyon hafta-lokasyon bazında (n>3000) çalışır.
  Soru: Bölge ile yağışlı haftada düşük ziyaretçi olması arasında ilişki var mı?

YÖNTEM:
  Her hafta-lokasyon kaydı için iki kategorik değişken oluşturulur:
  - Bölge: Batı / Orta / Doğu Karadeniz
  - Yağışlı düşük ziyaretçi: Hafta yağışlı (>1mm) VE ziyaretçi o lokasyonun
    medyanının altında ise "Evet", aksi halde "Hayır"
  Sonra 3×2 çapraz tablo ile Ki-kare bağımsızlık testi.

H0: Bölge ile yağışta-ziyaretçi-düşmesi arasında ilişki yoktur (bağımsızdırlar).
H1: Bölge ile yağışta-ziyaretçi-düşmesi arasında ilişki vardır.
"""

print("\n" + "="*78)
print("TEST 4 — Kİ-KARE BAĞIMSIZLIK: BÖLGE × YAĞIŞTA DÜŞÜK ZİYARETÇİ")
print("="*78)
print("""H0: Bölge ile 'yağışlı haftada düşük ziyaretçi' bağımsızdır.
H1: Bölge ile 'yağışlı haftada düşük ziyaretçi' arasında ilişki vardır.
Yöntem: Ki-kare bağımsızlık testi (chi2_contingency), n=hafta-lokasyon kayıtları.""")

# Her lokasyon için medyan hesapla, sonra "yağışlı + düşük ziyaretçi" flag'i koy
df['lok_medyan'] = df.groupby('Lokasyon Adı')['gercek_ziyaretci'].transform('median')
df['yagisli_dusuk'] = np.where(
    (df['prcp'] > 1.0) & (df['gercek_ziyaretci'] < df['lok_medyan']),
    'Evet', 'Hayır'
)

# Çapraz tablo (Bölge × yagisli_dusuk)
capraz_tablo = pd.crosstab(df['bolge'], df['yagisli_dusuk'])
print("\n--- ÇAPRAZ TABLO ---")
print(capraz_tablo)

chi2_stat, p_chi2, dof, expected = chi2_contingency(capraz_tablo)
print(f"\n--- TEST SONUCU ---")
print(f"Ki-Kare istatistiği: {chi2_stat:.4f}")
print(f"Serbestlik derecesi: {dof}")
print(f"p-değeri: {p_chi2:.4e}")
print(f"Beklenen frekansların minimumu: {expected.min():.1f} (≥5 olmalı: "
      f"{'✓ SAĞLANIYOR' if expected.min()>=5 else '✗ İHLAL'})")

# Effect size: Cramér's V
n_total = capraz_tablo.values.sum()
cramers_v = np.sqrt(chi2_stat / (n_total * (min(capraz_tablo.shape) - 1)))
print(f"Cramér's V (etki büyüklüğü): {cramers_v:.4f} "
      f"({'küçük' if cramers_v<0.1 else 'orta' if cramers_v<0.3 else 'büyük'} etki)")

if p_chi2 < 0.05:
    print("\nSonuç: H0 REDDEDİLDİ. Bölge ile yağışta düşük ziyaretçi olması arasında "
          "istatistiksel olarak anlamlı bir İLİŞKİ VARDIR.")
else:
    print("\nSonuç: H0 reddedilemedi. Bölge ile yağışa karşı duyarlılık BAĞIMSIZDIR.")

df_test4 = pd.DataFrame({
    'Metrik': ['Chi2 İstatistiği', 'Serbestlik Derecesi', 'p-değeri',
               'Cramér V', 'Min Beklenen Frekans', 'Sonuç'],
    'Değer': [round(chi2_stat,4), dof, f"{p_chi2:.4e}",
              round(cramers_v,4), round(expected.min(),1),
              'H0 Reddedildi' if p_chi2<0.05 else 'H0 Reddedilemedi']
})
tum_sonuclar.append(("TEST 4: Ki-Kare (Bölge × Yağış Duyarlılığı)", df_test4))
tum_sonuclar.append(("TEST 4: Çapraz Tablo", capraz_tablo.reset_index()))


# ============================================================================
# TEST 5: PERMÜTASYON TESTİ — ML MODELİNDE HAVA DURUMUNUN ÖNEMİ
# ============================================================================
"""
AMAÇ:
  Bu testin projendeki nihai hedefe (ziyaretçi tahmini) DOĞRUDAN katkısı var.
  Soru: Hava durumu özellikleri (temp, prcp, snow, rain, wspd, rhum) ML
  modelinin tahmin gücüne istatistiksel olarak anlamlı katkı sağlıyor mu?

YÖNTEM:
  1. Tam modeli eğit (tüm özelliklerle) → R²_gerçek
  2. Hava durumu özelliklerini RASTGELE permüte et (ilişkilerini kır)
  3. Bu yeni veriyle modeli yeniden eğit → R²_permüte
  4. 100 kez tekrarla → null dağılım oluştur
  5. p-değeri = (R²_permüte ≥ R²_gerçek) olan permütasyon oranı

H0: Hava durumu özellikleri modelin performansına katkı sağlamaz.
H1: Hava durumu özellikleri modelin performansına anlamlı katkı sağlar.
"""

print("\n" + "="*78)
print("TEST 5 — PERMÜTASYON: ML MODELİNDE HAVA DURUMU ÖZELLİKLERİNİN ÖNEMİ")
print("="*78)
print("""H0: Hava durumu özellikleri modelin tahmin gücüne katkı sağlamaz.
H1: Hava durumu özellikleri modelin tahmin gücüne anlamlı katkı sağlar.
Yöntem: Hava durumu kolonlarını birlikte permüte ederek null dağılım oluştur,
        gerçek R² ile karşılaştır.""")

# Veri setini hazırla — sadece sızıntısız özellikler
WEATHER_COLS = ['temp', 'prcp', 'snow', 'rain', 'wspd', 'rhum']
LOCATION_COLS = ['Rakım (m)', 'Has. Mesafe (km)', 'Ort. Eğim (%)',
                 'Has. Varış Süresi (Dk)']
TIME_COLS = ['ay', 'hafta_indeksi', 'tatil_mi']

# Lokasyonu sayısal kodla
df['lok_kod'] = pd.Categorical(df['Lokasyon Adı']).codes

feature_cols = WEATHER_COLS + LOCATION_COLS + TIME_COLS + ['lok_kod']
data_ml = df[feature_cols + ['gercek_ziyaretci']].dropna().reset_index(drop=True)

X = data_ml[feature_cols].values
y = np.log1p(data_ml['gercek_ziyaretci'].values)  # log dönüşümü (çarpıklık)

# Train/test ayrımı: yıl bazlı (son ~%20 test)
split_idx = int(len(data_ml) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 1) Gerçek model performansı
rf_real = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
rf_real.fit(X_train, y_train)
r2_real = r2_score(y_test, rf_real.predict(X_test))
print(f"\nGerçek model R² (test seti): {r2_real:.4f}")

# 2) Permütasyon — hava durumu kolonlarının tümünü birlikte karıştır
weather_idx = [feature_cols.index(c) for c in WEATHER_COLS]
N_PERM_ML = 100  # ML eğitimi maliyetli olduğu için 100 yeterli
perm_r2s = []

print(f"Permütasyon başlıyor ({N_PERM_ML} iterasyon)...")
for i in range(N_PERM_ML):
    X_train_perm = X_train.copy()
    # Hava durumu satırlarını train içinde rastgele karıştır
    perm_indices = np.random.permutation(X_train.shape[0])
    X_train_perm[:, weather_idx] = X_train[perm_indices][:, weather_idx]

    rf_perm = RandomForestRegressor(n_estimators=100, random_state=i, n_jobs=-1)
    rf_perm.fit(X_train_perm, y_train)
    perm_r2s.append(r2_score(y_test, rf_perm.predict(X_test)))

    if (i+1) % 20 == 0:
        print(f"  ... {i+1}/{N_PERM_ML} permütasyon tamamlandı")

perm_r2s = np.array(perm_r2s)
p_value_ml = np.mean(perm_r2s >= r2_real)

print(f"\n--- TEST SONUCU ---")
print(f"Gerçek R²:           {r2_real:.4f}")
print(f"Permütasyon R² ort:  {perm_r2s.mean():.4f}")
print(f"Permütasyon R² std:  {perm_r2s.std():.4f}")
print(f"Permütasyon R² maks: {perm_r2s.max():.4f}")
print(f"R² farkı (etki):     {r2_real - perm_r2s.mean():.4f}")
print(f"p-değeri:            {p_value_ml:.4f}")

if p_value_ml < 0.05:
    print("\nSonuç: H0 REDDEDİLDİ. Hava durumu özellikleri modelin tahmin gücüne "
          "İSTATİSTİKSEL OLARAK ANLAMLI katkı sağlıyor.")
else:
    print("\nSonuç: H0 reddedilemedi.")

df_test5 = pd.DataFrame({
    'Metrik': ['Gerçek R²', 'Permütasyon R² ortalama', 'Permütasyon R² std',
               'Permütasyon R² maks', 'Etki büyüklüğü (ΔR²)',
               'Permütasyon p-değeri', 'Sonuç'],
    'Değer': [round(r2_real,4), round(perm_r2s.mean(),4),
              round(perm_r2s.std(),4), round(perm_r2s.max(),4),
              round(r2_real - perm_r2s.mean(),4), round(p_value_ml,4),
              'H0 Reddedildi' if p_value_ml<0.05 else 'H0 Reddedilemedi']
})
tum_sonuclar.append(("TEST 5: Permütasyon (ML Hava Durumu Önemi)", df_test5))


# ============================================================================
# SONUÇLARI CSV'YE KAYDET
# ============================================================================
print("\n" + "="*78)
print("SONUÇLAR CSV DOSYALARINA KAYDEDILIYOR")
print("="*78)

ozet_yolu = os.path.join(SCRIPT_DIR, "Hipotez_Sonuclari_Ozet.csv")
with open(ozet_yolu, 'w', encoding='utf-8-sig') as f:
    for baslik, tablo in tum_sonuclar:
        f.write(f"\n{'='*70}\n{baslik}\n{'='*70}\n")
        tablo.to_csv(f, index=False)
        f.write("\n")

# Ayrı ayrı CSV'ler de
df_test1.to_csv(os.path.join(SCRIPT_DIR, "Test1_Permutasyon_Sicaklik.csv"),
                index=False, encoding='utf-8-sig')
df_test2.to_csv(os.path.join(SCRIPT_DIR, "Test2_MannWhitney_Yagis.csv"),
                index=False, encoding='utf-8-sig')
df_test3.to_csv(os.path.join(SCRIPT_DIR, "Test3_KruskalWallis_Mevsim.csv"),
                index=False, encoding='utf-8-sig')
df_test4.to_csv(os.path.join(SCRIPT_DIR, "Test4_KiKare_Bolge.csv"),
                index=False, encoding='utf-8-sig')
df_test5.to_csv(os.path.join(SCRIPT_DIR, "Test5_Permutasyon_ML.csv"),
                index=False, encoding='utf-8-sig')

print("✓ Hipotez_Sonuclari_Ozet.csv (tüm testler tek dosyada)")
print("✓ Test1_Permutasyon_Sicaklik.csv")
print("✓ Test2_MannWhitney_Yagis.csv")
print("✓ Test3_KruskalWallis_Mevsim.csv")
print("✓ Test4_KiKare_Bolge.csv")
print("✓ Test5_Permutasyon_ML.csv")

print("\n" + "="*78)
print("TÜM HİPOTEZ TESTLERİ TAMAMLANDI ✓")
print("="*78)
