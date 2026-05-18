# Kamp Ziyaretçi Tahmin Sistemi

Karadeniz bölgesindeki 16 kamp alanı için **gelecek haftanın ziyaretçi sayısını** tahmin eden web uygulaması. Kullanıcı sadece bir lokasyon seçer; sistem Open-Meteo'dan o noktanın 7 günlük hava tahmini çeker, Türkiye tatil takvimini kontrol eder ve Random Forest modeli ile haftalık ziyaretçi tahmini döndürür.

## Mimari

```
┌─────────────────────────────────────────────────────────┐
│  Tarayıcı (index.html)                                  │
│   ├─ Kart listesi (18 lokasyon)                         │
│   └─ Tıklayınca → AJAX                                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────┐
│  Flask (app.py)                                         │
│   ├─ /api/locations         → lokasyon listesi          │
│   └─ /api/predict?id=X      ↓                           │
│         ├─ locations.py           (lat/lon/rakım)       │
│         ├─ weather_forecast.py    (Open-Meteo API)      │
│         ├─ tatil_kontrolu.py      (TR tatilleri)        │
│         └─ predictor.py           (ML model)            │
└─────────────────────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────┐
│  models/production_artifact.pkl                         │
│  (RF modeli + mevsimsel tablo + büyüme verileri)        │
└─────────────────────────────────────────────────────────┘
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
# 1. (Sadece bir kez) Modeli eğit:
python train_model.py
#    → models/production_artifact.pkl üretir

# 2. Sunucuyu başlat:
python app.py
#    → http://localhost:5001
```

İlk açılışta 16 lokasyonun kartı listelenir, hepsi tıklanabilir durumda. Bir karta tıklayınca arkada Open-Meteo'ya forecast isteği gider, ML modeli haftalık ziyaretçi tahminini hesaplar ve panel olarak gösterilir.

## Dosyalar

| Dosya | Görev |
|---|---|
| `train_model.py` | ML modelini eğitir, `models/production_artifact.pkl`'e kaydeder. Bir kez çalıştırılır. |
| `app.py` | Flask sunucu. Frontend serve eder, API endpoint'leri sağlar. |
| `predictor.py` | Pickle'dan modeli yükler, `predict_visitor()` fonksiyonu sağlar. |
| `weather_forecast.py` | Open-Meteo API wrapper. 7 günlük tahmini haftalık aggregate'e dönüştürür. |
| `tatil_kontrolu.py` | Türkiye resmi tatil takvimi (2024-2027). |
| `locations.py` | 18 kamp alanının statik kataloğu (isim, lat, lon, rakım). |
| `templates/index.html` | Frontend tek dosya HTML+CSS+JS. |
| `data/AA_Makine_Ogrenmesi_Hazir_Tum_Veri_YENI.csv` | Eğitim verisi. |
| `models/production_artifact.pkl` | Eğitilmiş model + yardımcı veriler (~13MB). |

## Model Performansı

Eğitim sonrası ölçülen:

- **Test R² = 0.868** (2025 verisi üzerinde)
- **Test MAE = ±2,148 kişi/hafta**
- OOB R² = 0.846

Eğitim train+test ile yapılır (production model 2022-2025 tümünü görür); değerlendirme metrikleri train=2022-2024 / test=2025 ile ayrı hesaplanır.

## Notlar

- **Open-Meteo internet bağlantısı gerektirir.** API key gerekmez, ücretsizdir.
- **Tatil takvimi 2024-2027 ile sınırlıdır.** Daha uzun süre için `tatil_kontrolu.py`'a yeni dini bayram tarihleri eklenmesi gerekir.
- **Production deployment için:** `debug=False` yap ve `gunicorn` veya `uwsgi` ile çalıştır.
