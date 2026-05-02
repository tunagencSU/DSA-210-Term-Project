import pandas as pd
import numpy as np
import re
import os
import ssl

# SSL sertifika doğrulamasını global olarak devre dışı bırakır
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# 1. METİN PARÇALAYICI (INPUT İŞLEYİCİ)
# ==========================================
def girdi_cozumle(kullanici_girdisi):
    """
    Kullanıcı girdisinden kamp alanının ismini ve 4 yıllık ziyaretçi sayılarını çeker.
    Örnek: Borçka Karagöl Tabiat Parkı - Artvin 300000 450000 600000 850000
    """
    pattern = r"^(.*?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$"
    match = re.match(pattern, kullanici_girdisi.strip())
    
    if match:
        isim = match.group(1).strip()
        ziyaretci_2022 = int(match.group(2))
        ziyaretci_2023 = int(match.group(3))
        ziyaretci_2024 = int(match.group(4))
        ziyaretci_2025 = int(match.group(5))
        return isim, ziyaretci_2022, ziyaretci_2023, ziyaretci_2024, ziyaretci_2025
    else:
        print("\nHATA: Girdi formatı anlaşılamadı! Örnek format: Borçka Karagöl - Artvin 300000 450000 600000 850000")
        return None, None, None, None, None

# ==========================================
# 2. ANA MERGE VE GRUPLAMA FONKSİYONU
# ==========================================
def veri_birlestir(isim, ziyaretci_2022, ziyaretci_2023, ziyaretci_2024, ziyaretci_2025):
    # Girdi Dosyaları
    yorum_dosyasi = f"---{isim}.csv"
    hava_dosyasi = f"-- {isim}_final_hava_gunluk.csv"
    
    # Çıktı Dosyası 
    haftalik_cikti = f"##{isim}_haftalık_merge.csv"

    print(f"\n--- {isim.upper()} İÇİN İŞLEMLER BAŞLATILIYOR ---")
    
    if not os.path.exists(yorum_dosyasi):
        print(f"Kritik Hata: Yorum dosyası bulunamadı! Aranan dosya: '{yorum_dosyasi}'")
        return
    if not os.path.exists(hava_dosyasi):
        print(f"Kritik Hata: Hava durumu dosyası bulunamadı! Aranan dosya: '{hava_dosyasi}'")
        return

    df_reviews = pd.read_csv(yorum_dosyasi)
    df_weather = pd.read_csv(hava_dosyasi)

    # ------------------------------------------
    # YORUMLARI GÜNLÜK BAZDA GRUPLAMA 
    # ------------------------------------------
    df_reviews['tarih'] = pd.to_datetime(df_reviews['publishedAtDate']).dt.tz_localize(None).dt.floor('D')

    gunluk_yorumlar = df_reviews.groupby('tarih').agg(
        yorum_sayisi=('stars', 'count'),
        ortalama_puan=('stars', 'mean')
    ).reset_index()

    # ------------------------------------------
    # 4 YILLIK BETA (ÇARPAN) HESAPLAMASI
    # ------------------------------------------
    yorum_2022 = gunluk_yorumlar[gunluk_yorumlar['tarih'].dt.year == 2022]['yorum_sayisi'].sum()
    yorum_2023 = gunluk_yorumlar[gunluk_yorumlar['tarih'].dt.year == 2023]['yorum_sayisi'].sum()
    yorum_2024 = gunluk_yorumlar[gunluk_yorumlar['tarih'].dt.year == 2024]['yorum_sayisi'].sum()
    yorum_2025 = gunluk_yorumlar[gunluk_yorumlar['tarih'].dt.year == 2025]['yorum_sayisi'].sum()

    beta_2022 = ziyaretci_2022 / yorum_2022 if yorum_2022 > 0 else 0
    beta_2023 = ziyaretci_2023 / yorum_2023 if yorum_2023 > 0 else 0
    beta_2024 = ziyaretci_2024 / yorum_2024 if yorum_2024 > 0 else 0
    beta_2025 = ziyaretci_2025 / yorum_2025 if yorum_2025 > 0 else 0

    # ------------------------------------------
    # HAVA DURUMU İLE BİRLEŞTİRME VE DÜZENLEME
    # ------------------------------------------
    df_weather.rename(columns={df_weather.columns[0]: 'tarih'}, inplace=True)
    df_weather['tarih'] = pd.to_datetime(df_weather['tarih'])

    # 2022-2025 aralığını filtrele
    hedef_baslangic = pd.to_datetime('2022-01-01')
    hedef_bitis = pd.to_datetime('2025-12-31')
    df_weather = df_weather[(df_weather['tarih'] >= hedef_baslangic) & (df_weather['tarih'] <= hedef_bitis)]

    df_final = pd.merge(df_weather, gunluk_yorumlar, on='tarih', how='left')
    df_final['yorum_sayisi'] = df_final['yorum_sayisi'].fillna(0)

    # 4 yılın koşulları ve betaları eklendi
    kosullar = [
        df_final['tarih'].dt.year == 2022,
        df_final['tarih'].dt.year == 2023,
        df_final['tarih'].dt.year == 2024,
        df_final['tarih'].dt.year == 2025
    ]
    betalar = [beta_2022, beta_2023, beta_2024, beta_2025]
    df_final['uygulanan_beta'] = np.select(kosullar, betalar, default=0)

    df_final['gercek_ziyaretci'] = (df_final['yorum_sayisi'] * df_final['uygulanan_beta']).round()

    # ------------------------------------------
    # HAFTALIK VERİ DÖNÜŞÜMÜ
    # ------------------------------------------
    agg_dict = {
        'gercek_ziyaretci': 'sum',          
        'yorum_sayisi': 'sum',              
        'ortalama_puan': 'mean',
        'uygulanan_beta': 'mean' 
    }
    
    # Sütun toplama/ortalama işlemlerini biraz daha temiz bir yapıya soktuk
    for col in ['temp', 'prcp', 'snow', 'rain', 'wspd', 'rhum']:
        col_name = f'{isim}_{col}'
        if col_name in df_final.columns:
            agg_dict[col_name] = 'sum' if col in ['prcp', 'snow', 'rain'] else 'mean'

    haftalik_df = df_final.set_index('tarih').resample('W-MON').agg(agg_dict)
    haftalik_df = haftalik_df.round(2).reset_index()

    baslangic_tarihi = haftalik_df['tarih'].dt.strftime('%Y.%m.%d')
    bitis_tarihi = (haftalik_df['tarih'] + pd.Timedelta(days=6)).dt.strftime('%Y.%m.%d')
    haftalik_df['tarih'] = baslangic_tarihi + ' - ' + bitis_tarihi

    haftalik_df.to_csv(haftalik_cikti, index=False)
    print(f"HAFTALIK veri başarıyla kaydedildi: '{haftalik_cikti}'")
    print("-" * 55)

# ==========================================
# 3. ÇALIŞTIRMA BLOKU
# ==========================================
if __name__ == "__main__":
    girdi = input("\nLütfen lokasyon ve 4 yıllık (2022-2023-2024-2025) ziyaretçi bilgisini formatına uygun yapıştırın:\n> ")
    hedef_isim, z_2022, z_2023, z_2024, z_2025 = girdi_cozumle(girdi)
    
    if hedef_isim and all(z is not None for z in [z_2022, z_2023, z_2024, z_2025]):
        veri_birlestir(hedef_isim, z_2022, z_2023, z_2024, z_2025)