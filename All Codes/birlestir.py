"""
Kamp alanı haftalık verilerini ve lokasyon meta verilerini
makine öğrenmesi için tek bir CSV'de birleştirir.

KULLANIM:
    1) Tüm 16 lokasyonun haftalık CSV dosyalarını HAFTALIK_KLASORU içine koy.
       Dosya isimleri ÖNEMLİ: dosya adı, haftalık dosyadaki hava sütunlarındaki
       prefix ile eşleşmek zorunda DEĞİL — script dosyanın içindeki sütun
       adlarından lokasyon prefix'ini otomatik tespit eder.
    2) Camp_data dosyasının yolunu CAMP_DATA_DOSYASI olarak ayarla.
    3) python birlestir.py
"""

import os
import glob
import pandas as pd
import re
from datetime import date, timedelta

# ----------------------------------------------------------------------------
# AYARLAR
# ----------------------------------------------------------------------------
# Kodun bulunduğu 'ALL CODES' klasörünün tam yolunu al
MEVCUT_KLASOR = os.path.dirname(os.path.abspath(__file__))

# Bir üst klasöre (DSA_PROJE_ML) çık
ANA_KLASOR = os.path.dirname(MEVCUT_KLASOR)

# Yolları ANA_KLASOR üzerinden tanımla
HAFTALIK_KLASORU   = os.path.join(ANA_KLASOR, "## Merged Data")
CAMP_DATA_DOSYASI  = os.path.join(ANA_KLASOR, "Fixed Data", "AA_Camp_data_road___annual_visitors.csv")
CIKTI_DOSYASI      = os.path.join(ANA_KLASOR, "Fixed Data", "AA_Makine_Ogrenmesi_Hazir_Tum_Veri_YENI.csv")

# Haftalık dosyalardaki sütun isimleri (lokasyon prefix'i çıkarıldıktan SONRA)
HAVA_SUTUNLARI = ["temp", "prcp", "snow", "rain", "wspd", "rhum"]

# ----------------------------------------------------------------------------
# 1) TÜRKİYE RESMİ TATİL GÜNLERİ (2022-2026)
#    Hafta içinde herhangi bir gün resmi tatilse tatil_mi = 1
# ----------------------------------------------------------------------------
RESMI_TATILLER = {
    # 2022
    date(2022, 1, 1),                                                    # Yılbaşı
    date(2022, 4, 23),                                                   # Ulusal Egemenlik
    date(2022, 5, 1),                                                    # Emek ve Dayanışma
    date(2022, 5, 2), date(2022, 5, 3), date(2022, 5, 4),                # Ramazan Bayramı
    date(2022, 5, 19),                                                   # Atatürk'ü Anma, Gençlik ve Spor
    date(2022, 7, 9), date(2022, 7, 10), date(2022, 7, 11), date(2022, 7, 12),  # Kurban Bayramı
    date(2022, 7, 15),                                                   # Demokrasi ve Milli Birlik
    date(2022, 8, 30),                                                   # Zafer Bayramı
    date(2022, 10, 29),                                                  # Cumhuriyet Bayramı
    # 2023
    date(2023, 1, 1),
    date(2023, 4, 21), date(2023, 4, 22), date(2023, 4, 23),             # Ramazan Bayramı + Egemenlik
    date(2023, 5, 1),
    date(2023, 5, 19),
    date(2023, 6, 28), date(2023, 6, 29), date(2023, 6, 30), date(2023, 7, 1),  # Kurban Bayramı
    date(2023, 7, 15),
    date(2023, 8, 30),
    date(2023, 10, 29),
    # 2024
    date(2024, 1, 1),
    date(2024, 4, 9), date(2024, 4, 10), date(2024, 4, 11),              # Ramazan Bayramı
    date(2024, 4, 23),
    date(2024, 5, 1),
    date(2024, 5, 19),
    date(2024, 6, 16), date(2024, 6, 17), date(2024, 6, 18), date(2024, 6, 19),  # Kurban Bayramı
    date(2024, 7, 15),
    date(2024, 8, 30),
    date(2024, 10, 29),
    # 2025
    date(2025, 1, 1),
    date(2025, 3, 30), date(2025, 3, 31), date(2025, 4, 1),              # Ramazan Bayramı
    date(2025, 4, 23),
    date(2025, 5, 1),
    date(2025, 5, 19),
    date(2025, 6, 6), date(2025, 6, 7), date(2025, 6, 8), date(2025, 6, 9),     # Kurban Bayramı
    date(2025, 7, 15),
    date(2025, 8, 30),
    date(2025, 10, 29),
    # 2026 (haftalık veri 2026 Ocak başında bitiyor)
    date(2026, 1, 1),
}


def hafta_tatil_mi(tarih_str: str) -> int:
    """ '2022.01.03 - 2022.01.09' formatından haftada tatil var mı kontrolü """
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
    """ Haftanın orta gününün ayını döndürür (Çarşamba) """
    bas, son = tarih_str.split(" - ")
    bas = date(*map(int, bas.split(".")))
    son = date(*map(int, son.split(".")))
    orta = bas + (son - bas) / 2
    return orta.month


# ----------------------------------------------------------------------------
# 2) Lokasyon meta verisini yükle
# ----------------------------------------------------------------------------
camp = pd.read_csv(CAMP_DATA_DOSYASI)
print(f"Lokasyon meta verisi yüklendi: {len(camp)} lokasyon")

# Lokasyon adlarını normalize etmek için yardımcı (boşluk farklarını yutar)
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

camp["_norm"] = camp["Lokasyon Adı"].apply(normalize)


# ----------------------------------------------------------------------------
# 3) Her haftalık dosyayı oku, lokasyon meta verisi ile eşleştir
# ----------------------------------------------------------------------------
haftalik_dosyalar = sorted(glob.glob(os.path.join(HAFTALIK_KLASORU, "*.csv")))
print(f"Haftalık dosya sayısı: {len(haftalik_dosyalar)}")

tum_satirlar = []

for dosya in haftalik_dosyalar:
    df = pd.read_csv(dosya)

    # Lokasyon prefix'ini hava sütunlarından tespit et
    temp_col = [c for c in df.columns if c.endswith("_temp")]
    if not temp_col:
        print(f"  UYARI: {dosya} içinde _temp sütunu yok, atlanıyor")
        continue
    prefix = temp_col[0][:-len("_temp")]

    # Hava sütunlarını yeniden adlandır
    yeniden_ad = {f"{prefix}_{x}": x for x in HAVA_SUTUNLARI}
    df = df.rename(columns=yeniden_ad)

    # prefix'ten lokasyon adını çıkar
    candidates = [prefix, prefix.split(" - ")[0]]
    eslesen = None
    for cand in candidates:
        cand_norm = normalize(cand)
        match = camp[camp["_norm"] == cand_norm]
        if len(match) == 1:
            eslesen = match.iloc[0]
            break
    if eslesen is None:
        print(f"  UYARI: '{prefix}' Camp_data ile eşleşmedi, atlanıyor")
        continue

    # Haftalık verilerden alanları üret
    df["Lokasyon Adı"]  = eslesen["Lokasyon Adı"]
    df["ay"]            = df["tarih"].apply(hafta_orta_ay)
    df["tatil_mi"]      = df["tarih"].apply(hafta_tatil_mi)
    df["hafta_indeksi"] = range(1, len(df) + 1)

    # Lokasyon meta verisini her satıra ekle
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
    print(f"  ✓ {eslesen['Lokasyon Adı']}: {len(df)} hafta")

if not tum_satirlar:
    raise SystemExit("Hiç satır birleştirilemedi.")

birlesik = pd.concat(tum_satirlar, ignore_index=True)
print(f"\nToplam birleştirilen satır: {len(birlesik)}")


# ----------------------------------------------------------------------------
# 4) One-hot encoding (Yol Türü ve Bölge)
# ----------------------------------------------------------------------------
yol_dummies   = pd.get_dummies(birlesik["_Yol"],   prefix="Yol Türü & Yüzey Durumu").astype(int)
bolge_dummies = pd.get_dummies(birlesik["_Bolge"], prefix="Bölge").astype(int)
birlesik = pd.concat([birlesik, yol_dummies, bolge_dummies], axis=1)
birlesik = birlesik.drop(columns=["_Yol", "_Bolge", "tarih"])


# ----------------------------------------------------------------------------
# 5) Sütunları eski veri seti sırasına göre düzenle
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
# 6) Hedef sütunları integer'a çevir, kaydet
# ----------------------------------------------------------------------------
for col in ["gercek_ziyaretci", "yorum_sayisi"]:
    birlesik[col] = birlesik[col].fillna(0).astype(int)

birlesik.to_csv(CIKTI_DOSYASI, index=False)
print(f"\n✓ KAYDEDİLDİ: {CIKTI_DOSYASI}")
print(f"  Boyut: {birlesik.shape[0]} satır × {birlesik.shape[1]} sütun")
print(f"\nİlk 3 satır:")
print(birlesik.head(3).to_string())