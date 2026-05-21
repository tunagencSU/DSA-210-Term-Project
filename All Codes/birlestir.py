"""
Merges weekly campground data and location metadata
into a single CSV ready for machine learning.

USAGE:
    1) Place the weekly CSV files of all 16 locations into HAFTALIK_KLASORU.
       File names DO NOT need to match the prefix in the weather columns of
       the weekly file — the script automatically detects the location prefix
       from the column names inside the file.
    2) Set the path of the Camp_data file as CAMP_DATA_DOSYASI.
    3) python birlestir.py
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
import glob
import pandas as pd
import re
from datetime import date, timedelta

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
# Get the full path of the 'ALL CODES' folder where this code is located
MEVCUT_KLASOR = os.path.dirname(os.path.abspath(__file__))

# Go up one folder (DSA_PROJE_ML)
ANA_KLASOR = os.path.dirname(MEVCUT_KLASOR)

# Define paths based on ANA_KLASOR
HAFTALIK_KLASORU   = os.path.join(ANA_KLASOR, "## Merged Data")
CAMP_DATA_DOSYASI  = os.path.join(ANA_KLASOR, "Fixed Data", "AA_Camp_data_road___annual_visitors.csv")
CIKTI_DOSYASI      = os.path.join(ANA_KLASOR, "Fixed Data", "AA_Makine_Ogrenmesi_Hazir_Tum_Veri_YENI.csv")

# Column names in the weekly files (AFTER the location prefix is stripped)
HAVA_SUTUNLARI = ["temp", "prcp", "snow", "rain", "wspd", "rhum"]

# ----------------------------------------------------------------------------
# 1) TURKEY OFFICIAL PUBLIC HOLIDAYS (2022-2026)
#    If any day in the week is an official holiday, tatil_mi = 1
# ----------------------------------------------------------------------------
RESMI_TATILLER = {
    # 2022
    date(2022, 1, 1),                                                    # New Year's Day
    date(2022, 4, 23),                                                   # National Sovereignty Day
    date(2022, 5, 1),                                                    # Labor and Solidarity Day
    date(2022, 5, 2), date(2022, 5, 3), date(2022, 5, 4),                # Eid al-Fitr (Ramazan Bayramı)
    date(2022, 5, 19),                                                   # Commemoration of Atatürk, Youth and Sports Day
    date(2022, 7, 9), date(2022, 7, 10), date(2022, 7, 11), date(2022, 7, 12),  # Eid al-Adha (Kurban Bayramı)
    date(2022, 7, 15),                                                   # Democracy and National Unity Day
    date(2022, 8, 30),                                                   # Victory Day
    date(2022, 10, 29),                                                  # Republic Day
    # 2023
    date(2023, 1, 1),
    date(2023, 4, 21), date(2023, 4, 22), date(2023, 4, 23),             # Eid al-Fitr + National Sovereignty
    date(2023, 5, 1),
    date(2023, 5, 19),
    date(2023, 6, 28), date(2023, 6, 29), date(2023, 6, 30), date(2023, 7, 1),  # Eid al-Adha
    date(2023, 7, 15),
    date(2023, 8, 30),
    date(2023, 10, 29),
    # 2024
    date(2024, 1, 1),
    date(2024, 4, 9), date(2024, 4, 10), date(2024, 4, 11),              # Eid al-Fitr
    date(2024, 4, 23),
    date(2024, 5, 1),
    date(2024, 5, 19),
    date(2024, 6, 16), date(2024, 6, 17), date(2024, 6, 18), date(2024, 6, 19),  # Eid al-Adha
    date(2024, 7, 15),
    date(2024, 8, 30),
    date(2024, 10, 29),
    # 2025
    date(2025, 1, 1),
    date(2025, 3, 30), date(2025, 3, 31), date(2025, 4, 1),              # Eid al-Fitr
    date(2025, 4, 23),
    date(2025, 5, 1),
    date(2025, 5, 19),
    date(2025, 6, 6), date(2025, 6, 7), date(2025, 6, 8), date(2025, 6, 9),     # Eid al-Adha
    date(2025, 7, 15),
    date(2025, 8, 30),
    date(2025, 10, 29),
    # 2026 (weekly data ends at the beginning of January 2026)
    date(2026, 1, 1),
}


def hafta_tatil_mi(tarih_str: str) -> int:
    """ Checks whether there is a holiday in the week, from '2022.01.03 - 2022.01.09' format """
    bas, son = tarih_str.split(" - ")
    bas = date(*map(int, bas.split(".")))
    son = date(*map(int, son.split(".")))
    g = bas
    while g <= son:
        if g in RESMI_TATILLER:
            return 1
        g += timedelta(days=1)
    return 0


def hafta_orta_ay(tarih_str: str) -> int:
    """ Returns the month of the middle day of the week (Wednesday) """
    bas, son = tarih_str.split(" - ")
    bas = date(*map(int, bas.split(".")))
    son = date(*map(int, son.split(".")))
    orta = bas + (son - bas) / 2
    return orta.month


# ----------------------------------------------------------------------------
# 2) Load location metadata
# ----------------------------------------------------------------------------
camp = pd.read_csv(CAMP_DATA_DOSYASI)
print(f"Location metadata loaded: {len(camp)} locations")

# Helper to normalize location names (absorbs whitespace differences)
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

camp["_norm"] = camp["Lokasyon Adı"].apply(normalize)


# ----------------------------------------------------------------------------
# 3) Read each weekly file and match it with location metadata
# ----------------------------------------------------------------------------
haftalik_dosyalar = sorted(glob.glob(os.path.join(HAFTALIK_KLASORU, "*.csv")))
print(f"Number of weekly files: {len(haftalik_dosyalar)}")

tum_satirlar = []

for dosya in haftalik_dosyalar:
    df = pd.read_csv(dosya)

    # Detect the location prefix from the weather columns
    temp_col = [c for c in df.columns if c.endswith("_temp")]
    if not temp_col:
        print(f"  WARNING: No _temp column found in {dosya}, skipping")
        continue
    prefix = temp_col[0][:-len("_temp")]

    # Rename the weather columns
    yeniden_ad = {f"{prefix}_{x}": x for x in HAVA_SUTUNLARI}
    df = df.rename(columns=yeniden_ad)

    # Extract location name from the prefix
    candidates = [prefix, prefix.split(" - ")[0]]
    eslesen = None
    for cand in candidates:
        cand_norm = normalize(cand)
        match = camp[camp["_norm"] == cand_norm]
        if len(match) == 1:
            eslesen = match.iloc[0]
            break
    if eslesen is None:
        print(f"  WARNING: '{prefix}' did not match Camp_data, skipping")
        continue

    # Generate fields from the weekly data
    df["Lokasyon Adı"]  = eslesen["Lokasyon Adı"]
    df["ay"]            = df["tarih"].apply(hafta_orta_ay)
    df["tatil_mi"]      = df["tarih"].apply(hafta_tatil_mi)
    df["hafta_indeksi"] = range(1, len(df) + 1)

    # Add location metadata to each row
    df["Rakım (m)"]              = eslesen["Rakım (m)"]
    df["En Yakın Hastane"]       = eslesen["En Yakın Hastane"]
    df["Has. Mesafe (km)"]       = eslesen["Has. Mesafe (km)"]
    df["Ort. Eğim (%)"]          = eslesen["Ort. Eğim (%)"]
    df["Ziyaretci_2022"]         = eslesen["Ziyaretci_2022"]
    df["Ziyaretci_2023"]         = eslesen["Ziyaretci_2023"]
    df["Ziyaretci_2024"]         = eslesen["Ziyaretci_2024"]
    df["Ziyaretci_2025"]         = eslesen["Ziyaretci_2025"]
    df["Has. Varış Süresi (Dk)"] = eslesen["Has. Varış Süresi (Dk)"]
    df["_Yol"]                   = eslesen["Yol Türü & Yüzey Durumu"]
    df["_Bolge"]                 = eslesen["Bölge"]

    tum_satirlar.append(df)
    print(f"  ✓ {eslesen['Lokasyon Adı']}: {len(df)} weeks")

if not tum_satirlar:
    raise SystemExit("No rows could be merged.")

birlesik = pd.concat(tum_satirlar, ignore_index=True)
print(f"\nTotal merged rows: {len(birlesik)}")


# ----------------------------------------------------------------------------
# 4) One-hot encoding (Road Type and Region)
# ----------------------------------------------------------------------------
yol_dummies   = pd.get_dummies(birlesik["_Yol"],   prefix="Yol Türü & Yüzey Durumu").astype(int)
bolge_dummies = pd.get_dummies(birlesik["_Bolge"], prefix="Bölge").astype(int)
birlesik = pd.concat([birlesik, yol_dummies, bolge_dummies], axis=1)
birlesik = birlesik.drop(columns=["_Yol", "_Bolge", "tarih"])


# ----------------------------------------------------------------------------
# 5) Reorder columns to match the old dataset order
# ----------------------------------------------------------------------------
sira = (
    [
        "gercek_ziyaretci", "yorum_sayisi", "ortalama_puan", "uygulanan_beta",
        "temp", "prcp", "snow", "rain", "wspd", "rhum",
        "tatil_mi", "ay", "hafta_indeksi",
        "Lokasyon Adı", "Rakım (m)", "En Yakın Hastane",
        "Has. Mesafe (km)", "Ort. Eğim (%)",
        "Ziyaretci_2022", "Ziyaretci_2023", "Ziyaretci_2024", "Ziyaretci_2025",
        "Has. Varış Süresi (Dk)",
    ]
    + sorted(yol_dummies.columns)
    + sorted(bolge_dummies.columns)
)
sira = [c for c in sira if c in birlesik.columns]
birlesik = birlesik[sira]


# ----------------------------------------------------------------------------
# 6) Cast target columns to integer and save
# ----------------------------------------------------------------------------
for col in ["gercek_ziyaretci", "yorum_sayisi"]:
    birlesik[col] = birlesik[col].fillna(0).astype(int)

birlesik.to_csv(CIKTI_DOSYASI, index=False)
print(f"\n✓ SAVED: {CIKTI_DOSYASI}")
print(f"  Size: {birlesik.shape[0]} rows × {birlesik.shape[1]} columns")
print(f"\nFirst 3 rows:")
print(birlesik.head(3).to_string())
