import pandas as pd
import numpy as np
import re
import os
import ssl

# Globally disables SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# 1. TEXT PARSER (INPUT HANDLER)
# ==========================================
def girdi_cozumle(kullanici_girdisi):
    """
    Extracts the camp area name and 4 years of visitor counts from the user input.
    Example: Borçka Karagöl Tabiat Parkı - Artvin 300000 450000 600000 850000
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
        print("\nERROR: Input format not recognized! Example format: Borçka Karagöl - Artvin 300000 450000 600000 850000")
        return None, None, None, None, None

# ==========================================
# 2. MAIN MERGE AND GROUPING FUNCTION
# ==========================================
def veri_birlestir(isim, ziyaretci_2022, ziyaretci_2023, ziyaretci_2024, ziyaretci_2025):
    # ------------------------------------------
    # FILE PATH SETUP
    # ------------------------------------------
    # Reference the parent directory (DSA_PROJE_ML) of the 'ALL CODES' folder where the code runs
    ana_dizin = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths of the relevant input and output folders
    yorum_klasoru = os.path.join(ana_dizin, "--- Google Maps comments")
    hava_klasoru = os.path.join(ana_dizin, "-- Weather data 2022-2023-2024-2025 daily")
    cikti_klasoru = os.path.join(ana_dizin, "## Merged data")
    
    # The output folder (## Merged data) is created automatically if it doesn't exist
    os.makedirs(cikti_klasoru, exist_ok=True)
    
    # Input and output file paths
    yorum_dosyasi = os.path.join(yorum_klasoru, f"---{isim}.csv")
    hava_dosyasi = os.path.join(hava_klasoru, f"-- {isim}_final_hava_gunluk.csv")
    haftalik_cikti = os.path.join(cikti_klasoru, f"##{isim}_haftalık_merge.csv")

    print(f"\n--- STARTING OPERATIONS FOR {isim.upper()} ---")
    
    if not os.path.exists(yorum_dosyasi):
        print(f"Critical Error: Review file not found! Looking for: '{yorum_dosyasi}'")
        return
    if not os.path.exists(hava_dosyasi):
        print(f"Critical Error: Weather file not found! Looking for: '{hava_dosyasi}'")
        return

    df_reviews = pd.read_csv(yorum_dosyasi)
    df_weather = pd.read_csv(hava_dosyasi)

    # ------------------------------------------
    # GROUP REVIEWS ON A DAILY BASIS
    # ------------------------------------------
    df_reviews['tarih'] = pd.to_datetime(df_reviews['publishedAtDate']).dt.tz_localize(None).dt.floor('D')

    gunluk_yorumlar = df_reviews.groupby('tarih').agg(
        yorum_sayisi=('stars', 'count'),
        ortalama_puan=('stars', 'mean')
    ).reset_index()

    # ------------------------------------------
    # 4-YEAR BETA (MULTIPLIER) CALCULATION
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
    # MERGE WITH WEATHER DATA AND CLEAN UP
    # ------------------------------------------
    df_weather.rename(columns={df_weather.columns[0]: 'tarih'}, inplace=True)
    df_weather['tarih'] = pd.to_datetime(df_weather['tarih'])

    # Filter to the 2022-2025 range
    hedef_baslangic = pd.to_datetime('2022-01-01')
    hedef_bitis = pd.to_datetime('2025-12-31')
    df_weather = df_weather[(df_weather['tarih'] >= hedef_baslangic) & (df_weather['tarih'] <= hedef_bitis)]

    df_final = pd.merge(df_weather, gunluk_yorumlar, on='tarih', how='left')
    df_final['yorum_sayisi'] = df_final['yorum_sayisi'].fillna(0)

    # Added the conditions and betas for the 4 years
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
    # WEEKLY DATA TRANSFORMATION
    # ------------------------------------------
    agg_dict = {
        'gercek_ziyaretci': 'sum',          
        'yorum_sayisi': 'sum',              
        'ortalama_puan': 'mean',
        'uygulanan_beta': 'mean' 
    }
    
    # Reorganized the column sum/mean operations into a slightly cleaner structure
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
    print(f"WEEKLY data saved successfully: '{haftalik_cikti}'")
    print("-" * 55)

# ==========================================
# 3. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    girdi = input("\nPlease paste the location and 4-year (2022-2023-2024-2025) visitor information in the correct format:\n> ")
    hedef_isim, z_2022, z_2023, z_2024, z_2025 = girdi_cozumle(girdi)
    
    if hedef_isim and all(z is not None for z in [z_2022, z_2023, z_2024, z_2025]):
        veri_birlestir(hedef_isim, z_2022, z_2023, z_2024, z_2025)