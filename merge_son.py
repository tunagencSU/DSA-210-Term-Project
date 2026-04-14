import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 1. METİN PARÇALAYICI (INPUT İŞLEYİCİ)
# ==========================================
def girdi_cozumle(kullanici_girdisi):
    """
    Kullanıcı girdisinden kamp alanının ismini ve yıllık ziyaretçi sayılarını çeker.
    Örnek: Borçka Karagöl Tabiat Parkı - Artvin 600000 850000
    Çıktı: "Borçka Karagöl Tabiat Parkı - Artvin", 600000, 850000
    """
    pattern = r"^(.*?)\s+(\d+)\s+(\d+)$"
    match = re.match(pattern, kullanici_girdisi.strip())
    
    if match:
        isim = match.group(1).strip()
        ziyaretci_2024 = int(match.group(2))
        ziyaretci_2025 = int(match.group(3))
        return isim, ziyaretci_2024, ziyaretci_2025
    else:
        print("\nHATA: Girdi formatı anlaşılamadı! Örnek format: Borçka Karagöl Tabiat Parkı - Artvin 600000 850000")
        return None, None, None

# ==========================================
# 2. ANA MERGE VE GRUPLAMA FONKSİYONU
# ==========================================
def veri_birlestir(isim, ziyaretci_2024, ziyaretci_2025):
    # Girdi Dosyaları
    yorum_dosyasi = f"---{isim}.csv"
    hava_dosyasi = f"-- {isim}_final_hava_gunluk.csv"
    
    # Çıktı Dosyası (Sadece haftalık)
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
    # YORUMLARI GÜNLÜK BAZDA GRUPLAMA (Sadece olan veriler)
    # ------------------------------------------
    df_reviews['tarih'] = pd.to_datetime(df_reviews['publishedAtDate']).dt.tz_localize(None).dt.floor('D')

    gunluk_yorumlar = df_reviews.groupby('tarih').agg(
        yorum_sayisi=('stars', 'count'),
        ortalama_puan=('stars', 'mean')
    ).reset_index()

    # ------------------------------------------
    # YILLIK BETA (ÇARPAN) HESAPLAMASI
    # ------------------------------------------
    yorum_2024 = gunluk_yorumlar[gunluk_yorumlar['tarih'].dt.year == 2024]['yorum_sayisi'].sum()
    yorum_2025 = gunluk_yorumlar[gunluk_yorumlar['tarih'].dt.year == 2025]['yorum_sayisi'].sum()

    beta_2024 = ziyaretci_2024 / yorum_2024 if yorum_2024 > 0 else 0
    beta_2025 = ziyaretci_2025 / yorum_2025 if yorum_2025 > 0 else 0

    # ------------------------------------------
    # HAVA DURUMU İLE BİRLEŞTİRME VE DÜZENLEME
    # ------------------------------------------
    df_weather.rename(columns={df_weather.columns[0]: 'tarih'}, inplace=True)
    df_weather['tarih'] = pd.to_datetime(df_weather['tarih'])

    # Sadece 2024 ve 2025 hava durumu verilerini filtrele
    hedef_baslangic = pd.to_datetime('2024-01-01')
    hedef_bitis = pd.to_datetime('2025-12-31')
    df_weather = df_weather[(df_weather['tarih'] >= hedef_baslangic) & (df_weather['tarih'] <= hedef_bitis)]

    # Hava durumunun tam takvimine yorumları sol birleştirme ile ekle (left merge)
    df_final = pd.merge(df_weather, gunluk_yorumlar, on='tarih', how='left')

    # İnterpolasyon YASAK. Olmayan günlerde yorum_sayisi = 0 olmalıdır.
    df_final['yorum_sayisi'] = df_final['yorum_sayisi'].fillna(0)

    # Hangi yıla aitse o yılın betasını uygula
    kosullar = [
        df_final['tarih'].dt.year == 2024,
        df_final['tarih'].dt.year == 2025
    ]
    betalar = [beta_2024, beta_2025]
    df_final['uygulanan_beta'] = np.select(kosullar, betalar, default=0)

    # Günlük gerçek ziyaretçi sayısı hesabı (Sadece yorum olan günlerde değer çıkar, diğerleri 0 olur)
    df_final['gercek_ziyaretci'] = (df_final['yorum_sayisi'] * df_final['uygulanan_beta']).round()

    # ------------------------------------------
    # HAFTALIK VERİ DÖNÜŞÜMÜ
    # ------------------------------------------
    agg_dict = {
        'gercek_ziyaretci': 'sum',          
        'yorum_sayisi': 'sum',              
        'ortalama_puan': 'mean', # Yorum olmayan haftalarda bu değer NaN (boş) çıkar (matematiksel olarak doğru olan budur)
        'uygulanan_beta': 'mean' # Sadece kontrol amaçlı o haftanın betası
    }
    
    if f'{isim}_temp' in df_final.columns: agg_dict[f'{isim}_temp'] = 'mean'
    if f'{isim}_prcp' in df_final.columns: agg_dict[f'{isim}_prcp'] = 'sum'
    if f'{isim}_snow' in df_final.columns: agg_dict[f'{isim}_snow'] = 'sum'
    if f'{isim}_rain' in df_final.columns: agg_dict[f'{isim}_rain'] = 'sum'
    if f'{isim}_wspd' in df_final.columns: agg_dict[f'{isim}_wspd'] = 'mean'
    if f'{isim}_rhum' in df_final.columns: agg_dict[f'{isim}_rhum'] = 'mean'

    haftalik_df = df_final.set_index('tarih').resample('W-MON').agg(agg_dict)
    haftalik_df = haftalik_df.round(2).reset_index()

    # Tarih aralığı formatlama
    baslangic_tarihi = haftalik_df['tarih'].dt.strftime('%Y.%m.%d')
    bitis_tarihi = (haftalik_df['tarih'] + pd.Timedelta(days=6)).dt.strftime('%Y.%m.%d')
    haftalik_df['tarih'] = baslangic_tarihi + ' - ' + bitis_tarihi

    # Haftalık dosyayı kaydet
    haftalik_df.to_csv(haftalik_cikti, index=False)
    print(f"HAFTALIK veri başarıyla kaydedildi: '{haftalik_cikti}'")
    print("-" * 55)

# ==========================================
# 3. ÇALIŞTIRMA BLOKU
# ==========================================
if __name__ == "__main__":
    girdi = input("\nLütfen lokasyon ve ziyaretçi bilgisini formatına uygun yapıştırın:\n> ")
    hedef_isim, z_2024, z_2025 = girdi_cozumle(girdi)
    
    if hedef_isim and z_2024 is not None and z_2025 is not None:
        veri_birlestir(hedef_isim, z_2024, z_2025)