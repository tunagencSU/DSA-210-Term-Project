import pandas as pd
import numpy as np
import meteostat as ms
from datetime import datetime
import re
import ssl
import os

# Globally disables SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Code that produces daily weather estimates, last 2 years 2024-2025

# ==========================================
# 1. TEXT PARSER (INPUT HANDLER)
# ==========================================
def girdi_cozumle(kullanici_girdisi):
    """
    Format: Ayder yaylası - Rize (40.953940783330886, 41.10212257916922) 1298
    This function splits the text into 4 parts: name, latitude, longitude, and elevation.
    """
    # Capture the format with regex
    pattern = r"^(.*?)\s*\(\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*\)\s*([0-9.-]+)$"
    match = re.match(pattern, kullanici_girdisi.strip())
    
    if match:
        isim = match.group(1).strip()
        enlem = float(match.group(2))
        boylam = float(match.group(3))
        rakim = float(match.group(4))
        return isim, enlem, boylam, rakim
    else:
        print("\nERROR: Input format not recognized!")
        print("Please make sure you enter it in the following format:")
        print("Example: Ayder yaylası - Rize (40.953940783330886, 41.10212257916922) 1298")
        return None

# ==========================================
# 2. STATION FINDER AND DATA TEST
# ==========================================
def istasyon_getir(enlem, boylam, baslangic, bitis, adet=3):
    hedef_nokta = ms.Point(enlem, boylam)
    istasyonlar = ms.stations.nearby(hedef_nokta, radius=150000, limit=40)
    
    if istasyonlar.empty:
        print("Warning: No stations found within 150 km radius!")
        return {}

    beklenen_gun_sayisi = (bitis - baslangic).days + 1 
    istasyon_sozlugu = {}
    bulunan_adet = 0

    print(f"\n--- SCANNING AND TESTING NEARBY STATIONS ---")
    print(f"{'STATION NAME':<25} | {'ID':<6} | {'DISTANCE':<8} | {'ELEV.':<6} | {'DATA COMPLETENESS'}")
    print("-" * 75)
    
    for station_id, row in istasyonlar.iterrows():
        if bulunan_adet >= adet:
            break
            
        mesafe_km = round(row['distance'] / 1000, 1)
        rakim_m = row['elevation']
        isim = str(row['name'])[:24]
        
        # Real data test
        test_verisi = ms.daily(station_id, baslangic, bitis).fetch()
        
        # Safety net against empty returns
        if test_verisi is None or test_verisi.empty:
            continue
            
        dolu_gun_sayisi = test_verisi['temp'].count() if 'temp' in test_verisi.columns else 0
        doluluk_orani = (dolu_gun_sayisi / beklenen_gun_sayisi) * 100
        
        # Accept stations with more than 70% data
        if doluluk_orani >= 70.0:
            istasyon_sozlugu[station_id] = (mesafe_km, rakim_m)
            print(f"{isim:<25} | {station_id:<6} | {mesafe_km} km | {rakim_m} m | {doluluk_orani:.1f}% Filled")
            bulunan_adet += 1

    return istasyon_sozlugu

# ==========================================
# 3. DATA FETCH, CALCULATION AND CSV CREATION
# ==========================================
def hava_durumu_olustur(isim, hedef_rakim, istasyonlar, baslangic, bitis):
    dfs = []
    features = ['temp', 'prcp', 'wspd', 'rhum', 'tmin', 'tmax']
    
    # IDW (Inverse Distance Weighting)
    valid_weights = {s: 1/(d if d > 0.1 else 0.1) for s, (d, r) in istasyonlar.items()}
    
    print("\n--- DOWNLOADING AND CALCULATING DATA FROM METEOSTAT ---")
    for station_id, (distance, elevation) in istasyonlar.items():
        data = ms.daily(station_id, baslangic, bitis).fetch()
        
        if data is not None and not data.empty:
            existing_cols = [c for c in features if c in data.columns]
            df = data[existing_cols].copy()
            
            # Elevation - Temperature correction (Lapse Rate)
            rakim_farki = hedef_rakim - elevation
            sicaklik_dusus_miktari = (rakim_farki / 100) * 0.65
            
            for temp_col in ['temp', 'tmin', 'tmax']:
                if temp_col in df.columns:
                    df[temp_col] = df[temp_col] - sicaklik_dusus_miktari
            
            df = df.add_suffix(f'_{station_id}')
            dfs.append(df)

    # Merge data and build daily index
    # CHANGE: freq='W' was replaced with freq='D'
    gunluk_index = pd.date_range(start=baslangic, end=bitis, freq='D')
    
    if not dfs:
        print(f"\nCritical Warning: No data could be processed for {isim.upper()}. Table will return empty (NaN).")
        daily_data = pd.DataFrame(index=gunluk_index)
        for f in features:
            daily_data[f'{isim}_{f}'] = np.nan
        daily_data[f'{isim}_snow'] = np.nan
        daily_data[f'{isim}_rain'] = np.nan
    else:
        merged_data = pd.concat(dfs, axis=1)
        # Fit the data onto a complete calendar (gunluk_index) from start to end:
        merged_data = merged_data.reindex(gunluk_index)

        # Fill missing values and compute the weighted average
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

        # Snow vs rain split (accelerated with Numpy)
        merged_data[f'{isim}_snow'] = np.where(merged_data[f'{isim}_temp'] <= 2.0, merged_data[f'{isim}_prcp'], 0.0)
        merged_data[f'{isim}_rain'] = np.where(merged_data[f'{isim}_temp'] > 2.0, merged_data[f'{isim}_prcp'], 0.0)

        # CHANGE: resample('W') removed; daily data is filtered and rounded directly
        sutunlar = [
            f'{isim}_temp', f'{isim}_prcp', f'{isim}_snow', 
            f'{isim}_rain', f'{isim}_tmin', f'{isim}_tmax', 
            f'{isim}_wspd', f'{isim}_rhum'
        ]
        daily_data = merged_data[sutunlar].round(2)

    # File saving
    # CHANGE: "_gunluk" appended to the file name to avoid mixing with the weekly ones
    
    # NEW: Define the output folder one level up (DSA_PROJE_ML) and create if missing
    cikti_klasoru = os.path.join("..", "-- Weather data 2022-2023-2024-2025 daily")
    os.makedirs(cikti_klasoru, exist_ok=True)
    
    dosya_adi = f"-- {isim}_final_hava_gunluk.csv"
    
    # Clean invalid characters from the name (to avoid Windows file system errors)
    dosya_adi = "".join(c for c in dosya_adi if c.isalnum() or c in (' ', '.', '_', '-'))
    
    # NEW: Build the full path (folder + file name)
    tam_yol = os.path.join(cikti_klasoru, dosya_adi)
    
    daily_data.to_csv(tam_yol)
    print(f"\n SUCCESS! Daily data saved to '{tam_yol}'!")

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    start_date = datetime(2021, 12, 31)
    end_date = datetime(2026, 1, 1)

    print("="*65)
    print(" WEATHER AND LOCATION AUTOMATION STARTED")
    print("="*65)
    
    # Ask the user for input
    girdi = input("\nPlease paste the location information in the correct format:\n> ")
    
    cozumlenmis = girdi_cozumle(girdi)
    
    if cozumlenmis:
        hedef_isim, enlem, boylam, rakım = cozumlenmis
        print(f"\nSystem detected -> Name: {hedef_isim} | Latitude: {enlem} | Longitude: {boylam} | Elevation: {rakım}")
        
        # 1. Find the stations
        bulunan_istasyonlar = istasyon_getir(enlem, boylam, start_date, end_date, adet=3)
        
        if bulunan_istasyonlar:
            # 2. Fetch the data and write to CSV
            hava_durumu_olustur(hedef_isim, rakım, bulunan_istasyonlar, start_date, end_date)
        else:
            print(f"\nCritical Error: Operation cancelled because no valid station could be found for {hedef_isim}.")