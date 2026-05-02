import pandas as pd
import numpy as np
import meteostat as ms
from datetime import datetime
import re
import ssl

# SSL sertifika doğrulamasını global olarak devre dışı bırakır
ssl._create_default_https_context = ssl._create_unverified_context

# günlük hava durumu tahmini yapan kod son 2 yıl 2024-2025

# ==========================================
# 1. METİN PARÇALAYICI (INPUT İŞLEYİCİ)
# ==========================================
def girdi_cozumle(kullanici_girdisi):
    """
    Format: Ayder yaylası - Rize (40.953940783330886, 41.10212257916922) 1298
    Bu fonksiyon metni isim, enlem, boylam ve rakım olarak 4 parçaya ayırır.
    """
    # Regex ile formatı yakalıyoruz
    pattern = r"^(.*?)\s*\(\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*\)\s*([0-9.-]+)$"
    match = re.match(pattern, kullanici_girdisi.strip())
    
    if match:
        isim = match.group(1).strip()
        enlem = float(match.group(2))
        boylam = float(match.group(3))
        rakim = float(match.group(4))
        return isim, enlem, boylam, rakim
    else:
        print("\nHATA: Girdi formatı anlaşılamadı!")
        print("Lütfen şu formatta girdiğinizden emin olun:")
        print("Örnek: Ayder yaylası - Rize (40.953940783330886, 41.10212257916922) 1298")
        return None

# ==========================================
# 2. İSTASYON BULUCU VE VERİ TESTİ
# ==========================================
def istasyon_getir(enlem, boylam, baslangic, bitis, adet=3):
    hedef_nokta = ms.Point(enlem, boylam)
    istasyonlar = ms.stations.nearby(hedef_nokta, radius=150000, limit=40)
    
    if istasyonlar.empty:
        print("Uyarı: 150 km çapında istasyon bulunamadı!")
        return {}

    beklenen_gun_sayisi = (bitis - baslangic).days + 1 
    istasyon_sozlugu = {}
    bulunan_adet = 0

    print(f"\n--- YAKIN İSTASYONLAR TARANIYOR VE TEST EDİLİYOR ---")
    print(f"{'İSTASYON ADI':<25} | {'ID':<6} | {'UZAKLIK':<8} | {'RAKIM':<6} | {'VERİ DOLULUĞU'}")
    print("-" * 75)
    
    for station_id, row in istasyonlar.iterrows():
        if bulunan_adet >= adet:
            break
            
        mesafe_km = round(row['distance'] / 1000, 1)
        rakim_m = row['elevation']
        isim = str(row['name'])[:24]
        
        # Gerçek veri testi
        test_verisi = ms.daily(station_id, baslangic, bitis).fetch()
        
        # Boş dönme hatalarına karşı güvenlik ağı
        if test_verisi is None or test_verisi.empty:
            continue
            
        dolu_gun_sayisi = test_verisi['temp'].count() if 'temp' in test_verisi.columns else 0
        doluluk_orani = (dolu_gun_sayisi / beklenen_gun_sayisi) * 100
        
        # %70'ten fazla verisi olanı kabul et
        if doluluk_orani >= 70.0:
            istasyon_sozlugu[station_id] = (mesafe_km, rakim_m)
            print(f"{isim:<25} | {station_id:<6} | {mesafe_km} km | {rakim_m} m | %{doluluk_orani:.1f} Dolu")
            bulunan_adet += 1

    return istasyon_sozlugu

# ==========================================
# 3. VERİ ÇEKME, HESAPLAMA VE CSV OLUŞTURMA
# ==========================================
def hava_durumu_olustur(isim, hedef_rakim, istasyonlar, baslangic, bitis):
    dfs = []
    features = ['temp', 'prcp', 'wspd', 'rhum', 'tmin', 'tmax']
    
    # IDW (Ters Mesafe Ağırlıklandırması)
    valid_weights = {s: 1/(d if d > 0.1 else 0.1) for s, (d, r) in istasyonlar.items()}
    
    print("\n--- METEOSTAT'TAN VERİLER İNDİRİLİYOR VE HESAPLANIYOR ---")
    for station_id, (distance, elevation) in istasyonlar.items():
        data = ms.daily(station_id, baslangic, bitis).fetch()
        
        if data is not None and not data.empty:
            existing_cols = [c for c in features if c in data.columns]
            df = data[existing_cols].copy()
            
            # Rakım - Sıcaklık Düzeltmesi (Lapse Rate)
            rakim_farki = hedef_rakim - elevation
            sicaklik_dusus_miktari = (rakim_farki / 100) * 0.65
            
            for temp_col in ['temp', 'tmin', 'tmax']:
                if temp_col in df.columns:
                    df[temp_col] = df[temp_col] - sicaklik_dusus_miktari
            
            df = df.add_suffix(f'_{station_id}')
            dfs.append(df)

    # Verileri Birleştirme ve Günlük Index
    # DEĞİŞİKLİK: freq='W' yerine freq='D' yapıldı
    gunluk_index = pd.date_range(start=baslangic, end=bitis, freq='D')
    
    if not dfs:
        print(f"\nKritik Uyarı: {isim.upper()} için hiçbir veri işlenemedi. Tablo boş (NaN) dönecek.")
        daily_data = pd.DataFrame(index=gunluk_index)
        for f in features:
            daily_data[f'{isim}_{f}'] = np.nan
        daily_data[f'{isim}_snow'] = np.nan
        daily_data[f'{isim}_rain'] = np.nan
    else:
        merged_data = pd.concat(dfs, axis=1)
        # Veriyi baştan sona eksiksiz bir takvime (gunluk_index) oturtuyoruz:
        merged_data = merged_data.reindex(gunluk_index)

        # Eksikleri doldur ve ağırlıklı ortalama al

        # Eksikleri doldur ve ağırlıklı ortalama al
        for feat in features:
            feat_cols = [c for c in merged_data.columns if c.startswith(f"{feat}_")]
            if not feat_cols: continue
                
            if feat == 'prcp':
                merged_data[feat_cols] = merged_data[feat_cols].fillna(0.0)
            else:
                merged_data[feat_cols] = merged_data[feat_cols].interpolate(method='linear', limit_direction='both').ffill().bfill()

            merged_data[f'{isim}_{feat}'] = 0.0
            current_total_weight = 0
            
            for col in feat_cols:
                s_id = col.split('_')[-1]
                w = valid_weights[s_id]
                merged_data[f'{isim}_{feat}'] += merged_data[col] * w
                current_total_weight += w
            
            merged_data[f'{isim}_{feat}'] /= current_total_weight

        # Kar ve Yağmur Ayrımı (Numpy ile hızlandırılmış)
        merged_data[f'{isim}_snow'] = np.where(merged_data[f'{isim}_temp'] <= 2.0, merged_data[f'{isim}_prcp'], 0.0)
        merged_data[f'{isim}_rain'] = np.where(merged_data[f'{isim}_temp'] > 2.0, merged_data[f'{isim}_prcp'], 0.0)

        # DEĞİŞİKLİK: resample('W') kaldırıldı, günlük veriler doğrudan filtrelenip yuvarlandı
        sutunlar = [
            f'{isim}_temp', f'{isim}_prcp', f'{isim}_snow', 
            f'{isim}_rain', f'{isim}_tmin', f'{isim}_tmax', 
            f'{isim}_wspd', f'{isim}_rhum'
        ]
        daily_data = merged_data[sutunlar].round(2)

    # Dosya Kaydı
    # DEĞİŞİKLİK: Haftalık olanlarla karışmaması için dosya ismine "_gunluk" eklendi
    
    dosya_adi = f"-- {isim}_final_hava_gunluk.csv"
    
    # İsimdeki geçersiz karakterleri temizle (Windows dosya sistemi hatası almamak için)
    dosya_adi = "".join(c for c in dosya_adi if c.isalnum() or c in (' ', '.', '_', '-'))
    
    daily_data.to_csv(dosya_adi)
    print(f"\n BAŞARILI! Günlük veriler '{dosya_adi}' dosyasına kaydedildi!")

# ==========================================
# 4. ANA ÇALIŞMA DÖNGÜSÜ
# ==========================================s
if __name__ == "__main__":
    start_date = datetime(2021, 12, 31)
    end_date = datetime(2026, 1, 1)

    print("="*65)
    print(" HAVA DURUMU VE LOKASYON OTOMASYONU BAŞLATILDI")
    print("="*65)
    
    # Kullanıcıdan girdi iste
    girdi = input("\nLütfen lokasyon bilgisini formatına uygun yapıştırın:\n> ")
    
    cozumlenmis = girdi_cozumle(girdi)
    
    if cozumlenmis:
        hedef_isim, enlem, boylam, rakım = cozumlenmis
        print(f"\nSistem algıladı -> İsim: {hedef_isim} | Enlem: {enlem} | Boylam: {boylam} | Rakım: {rakım}")
        
        # 1. İstasyonları Bul
        bulunan_istasyonlar = istasyon_getir(enlem, boylam, start_date, end_date, adet=3)
        
        if bulunan_istasyonlar:
            # 2. Verileri Çek ve CSV'ye Yazdır
            hava_durumu_olustur(hedef_isim, rakım, bulunan_istasyonlar, start_date, end_date)
        else:
            print(f"\nKritik Hata: {hedef_isim} için kullanılabilecek geçerli istasyon bulunamadığı için işlem iptal edildi.")
