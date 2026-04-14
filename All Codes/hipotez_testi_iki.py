import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import ttest_ind, chi2_contingency

# ==========================================
# ZİYARETÇİ MEDYANI & YAĞIŞ DUYARLILIĞI TESTİ
# Kullanılan Yöntem: Ki-Kare Bağımsızlık Testi (Chi-Square Test of Independence)
# ==========================================

# 1. Kullanıcıdan klasör yolunu al
hedef_klasor = input("Lütfen verilerin bulunduğu klasör yolunu girin: ").strip('"').strip("'")
temiz_veri_yolu = os.path.join(hedef_klasor, "Temizlenmis_Kamp_Verileri.csv")

if not os.path.exists(temiz_veri_yolu):
    print(f"HATA: '{temiz_veri_yolu}' bulunamadı!")
    exit()

# Temizlenmiş verileri yükle ve toplam ziyaretçiyi hesapla
df_temiz = pd.read_csv(temiz_veri_yolu)
df_temiz['Toplam_Ziyaretci'] = df_temiz['Ziyaretci_2024'] + df_temiz['Ziyaretci_2025']

# 2. Merge dosyalarını tara ve yağışın p-değerini hesapla (Slaytlardaki T-Test yaklaşımı)
arama_deseni = os.path.join(hedef_klasor, "*_haftalık_merge.csv")
dosya_yollari = glob.glob(arama_deseni)

kamp_sonuclari = []

print("\nVeriler okunuyor ve Bağımsız İki Örneklem T-Testleri (Welch's) hesaplanıyor...")
for dosya in dosya_yollari:
    # Dosya adını temizle (Örn: "##Ayder Yaylası - Rize_haftalık_merge.csv" -> "Ayder Yaylası")
    kamp_adi_ham = os.path.basename(dosya).replace("_haftalık_merge.csv", "").replace("##", "")
    gercek_isim = kamp_adi_ham.split('-')[0].strip()
    
    df = pd.read_csv(dosya)
    
    # Yağış sütununu dinamik olarak bul
    prcp_col = next((col for col in df.columns if col.endswith('_prcp')), None)
    ziyaretci_col = 'gercek_ziyaretci'
    
    yagis_anlamli = "Hayır"
    if prcp_col and ziyaretci_col in df.columns:
        valid_prcp = df[[prcp_col, ziyaretci_col]].dropna()
        yagisli = valid_prcp[valid_prcp[prcp_col] > 1.0][ziyaretci_col]
        yagissiz = valid_prcp[valid_prcp[prcp_col] <= 1.0][ziyaretci_col]
        
        # T-Testi için yeterli örneklem kontrolü
        if len(yagisli) > 1 and len(yagissiz) > 1:
            t_stat, p_val_prcp = ttest_ind(yagissiz, yagisli, equal_var=False)
            if p_val_prcp < 0.05:
                yagis_anlamli = "Evet"
                
    kamp_sonuclari.append({'Lokasyon Adı': gercek_isim, 'Yağış_Anlamlı': yagis_anlamli})

# 3. İki veri setini tam eşleşme (Exact Match) ile birleştir
df_yagis = pd.DataFrame(kamp_sonuclari)
df_final = pd.merge(df_temiz, df_yagis, on='Lokasyon Adı', how='inner')

if df_final.empty:
    print("\nHATA: Dosya isimleri eşleşmedi!")
    exit()

# 4. Medyanı bul ve kamp alanlarını kategorilere ayır (Slaytlardaki Kategorik Veri Dönüşümü)
medyan_nufus = df_final['Toplam_Ziyaretci'].median()
df_final['Nüfus_Kategorisi'] = np.where(df_final['Toplam_Ziyaretci'] > medyan_nufus, 'Medyandan Yüksek', 'Medyandan Düşük')

# 5. Çapraz Tablo (Contingency Table) Oluştur
capraz_tablo = pd.crosstab(df_final['Nüfus_Kategorisi'], df_final['Yağış_Anlamlı'])

# ==========================================
# 6. Kİ-KARE (CHI-SQUARE) BAĞIMSIZLIK TESTİ
# ==========================================
# H0: Ziyaretçi Kategorisi ile Yağışa Duyarlılık birbirinden bağımsızdır (İlişki yoktur).
# H1: İki kategori arasında istatistiksel olarak anlamlı bir ilişki vardır.
chi2_stat, p_val_chi2, dof, expected = chi2_contingency(capraz_tablo)

# ==========================================
# SONUÇLARIN YAZDIRILMASI VE CSV'YE KAYDEDİLMESİ
# ==========================================
print("-" * 70)
print(f"KARADENİZ KAMP ALANLARI ZİYARETÇİ MEDYANI: {medyan_nufus:,.0f} Kişi")
print("-" * 70)
print("\n--- ÇAPRAZ TABLO (DAĞILIM) ---")
print(capraz_tablo)
print("-" * 70)
print("\n--- Kİ-KARE (CHI-SQUARE) TESTİ SONUCU ---")
print(f"Ki-Kare İstatistiği (Chi2): {chi2_stat:.4f}")
print(f"Serbestlik Derecesi (DoF): {dof}")
print(f"P-Değeri (P-Value): {p_val_chi2:.4f}")

if p_val_chi2 < 0.05:
    sonuc_metni = "H0 Reddedildi! Ziyaretçi kategorisi ile yağışa duyarlılık arasında anlamlı bir ilişki VARDIR."
else:
    sonuc_metni = "H0 Kabul Edildi! Ziyaretçi kategorisi ile yağışa duyarlılık arasında istatistiksel bir İLİŞKİ YOKTUR."
print(f"Yorum: {sonuc_metni}")
print("-" * 70)

# CSV Kaydetme İşlemleri
# 1. Tüm detaylı veriyi kaydet (Hocanın incelemesi için)
detayli_csv_yolu = os.path.join(hedef_klasor, "Kategorik_Analiz_Detayli_Veri.csv")
df_final[['Lokasyon Adı', 'Toplam_Ziyaretci', 'Nüfus_Kategorisi', 'Yağış_Anlamlı']].to_csv(detayli_csv_yolu, index=False, encoding='utf-8-sig')

# 2. Test sonucunu ve özet tabloyu ayrı bir CSV'ye kaydet
ozet_csv_yolu = os.path.join(hedef_klasor, "Chi_Square_Test_Sonucu.csv")
with open(ozet_csv_yolu, 'w', encoding='utf-8-sig') as f:
    f.write("Kİ-KARE BAĞIMSIZLIK TESTİ SONUCU\n")
    f.write(f"Kullanilan Yontem: Chi-Square Test of Independence (scipy.stats.chi2_contingency)\n")
    f.write(f"Chi2 Istatistigi: {chi2_stat:.4f}\n")
    f.write(f"P-Value: {p_val_chi2:.4f}\n")
    f.write(f"Test Sonucu: {sonuc_metni}\n\n")
    f.write("CAPRAZ TABLO\n")
    capraz_tablo.to_csv(f)

print(f"\n✅ Detaylı veriler '{detayli_csv_yolu}' dosyasına kaydedildi.")
print(f"✅ İstatistiksel test sonuçları '{ozet_csv_yolu}' dosyasına kaydedildi.")