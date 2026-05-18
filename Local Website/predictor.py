"""
Campsite visitor prediction module.
Loads the trained artifact and provides the prediction function.

This module is the production-ready version of the gelecek_tahmin() function
in the original ml_v2 code. The only difference: the model and auxiliary data
are loaded from a pickle file, no in-memory computation.
"""

import os
import pickle
import numpy as np
import pandas as pd


# Module-level cache — loaded once at Flask startup
_ARTIFACT = None
_ARTIFACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models", "production_artifact.pkl")


def load_artifact(path: str = None):
    """Loads the model artifact from the pickle and stores it in the module cache."""
    global _ARTIFACT
    p = path or _ARTIFACT_PATH
    with open(p, "rb") as f:
        _ARTIFACT = pickle.load(f)
    return _ARTIFACT


def get_artifact():
    """Returns the cached artifact; loads it if not present."""
    if _ARTIFACT is None:
        load_artifact()
    return _ARTIFACT


def _acik_maskesi(acik_orani: float) -> float:
    """Linear ramp mask for winter closure weeks."""
    return float(np.clip((acik_orani - 0.1) / 0.4, 0, 1))


def predict_visitor(
    lokasyon_adi: str,
    yil: int,
    ay: int,
    yil_ici_hafta: int,
    temp: float,
    prcp: float,
    rain: float,
    snow: float,
    wspd: float,
    rhum: float,
    tatil_mi: int,
) -> dict:
    """
    Makes a weekly visitor prediction for the given location and weather.
    
    Args:
        lokasyon_adi:   Location name in the CSV (e.g., "Yedigöller Milli Parkı")
        yil, ay:        Prediction date
        yil_ici_hafta:  Week of the year (1-52)
        temp..rhum:     Weekly aggregate weather values
        tatil_mi:       0 or 1
    
    Returns:
        {
            'lokasyon': str,
            'tahmin_ziyaretci': int,
            'yillik_ort': int,
            'yillik_ort_projeksiyon': int,
            'buyume_carpani': float,
            'goreceli_talep': float,
            'mevsimsel_sinyal_ort': float | None,
            'mevsimsel_belirsizlik': float | None,
            'acik_orani': float,
            'acik_carpan': float,
        }
    
    Raises:
        ValueError: If the location is not recognized.
    """
    art = get_artifact()

    if lokasyon_adi not in art["all_lokasyonlar"]:
        raise ValueError(
            f"'{lokasyon_adi}' is not in the training set. "
            f"Available: {sorted(art['all_lokasyonlar'])}"
        )

    selected_features = art["selected_features"]
    mevsimsel_tablo   = art["mevsimsel_tablo"]
    yillik_ort_map    = art["yillik_ort_map"]
    lok_ref           = art["lokasyon_ref"][lokasyon_adi]
    yorum_ort_map     = art["yorum_ort_map"]
    yol_bolge_cols    = art["yol_bolge_cols"]
    mevsim_map        = art["mevsim_map"]
    cfg               = art["config"]
    rf_production     = art["model"]

    # ─── Time features ───
    yil_ici_hafta = max(1, min(52, int(yil_ici_hafta)))
    ay_sin    = np.sin(2 * np.pi * ay / 12)
    ay_cos    = np.cos(2 * np.pi * ay / 12)
    hafta_sin = np.sin(2 * np.pi * yil_ici_hafta / 52)
    hafta_cos = np.cos(2 * np.pi * yil_ici_hafta / 52)
    mevsim    = mevsim_map[ay]

    # ─── Seasonal pattern (location × yil_ici_hafta) ───
    mevsim_satir = mevsimsel_tablo[
        (mevsimsel_tablo["Lokasyon Adı"] == lokasyon_adi) &
        (mevsimsel_tablo["yil_ici_hafta"] == yil_ici_hafta)
    ]
    if len(mevsim_satir) > 0:
        ahgo = float(mevsim_satir["ayni_hafta_gecmis_ort"].values[0])
        ahgs = float(mevsim_satir["ayni_hafta_gecmis_std"].values[0])
        syah = float(mevsim_satir["son_yil_ayni_hafta"].values[0])
        kho  = float(mevsim_satir["komsu_hafta_ort"].values[0])
        acik_orani = float(mevsim_satir["acik_orani_gecmis"].values[0])
    else:
        # No history for this week — location average
        lok_data = mevsimsel_tablo[mevsimsel_tablo["Lokasyon Adı"] == lokasyon_adi]
        ahgo = float(lok_data["ayni_hafta_gecmis_ort"].mean())
        ahgs = float(lok_data["ayni_hafta_gecmis_std"].mean())
        syah = float(lok_data["son_yil_ayni_hafta"].mean())
        kho  = float(lok_data["komsu_hafta_ort"].mean())
        acik_orani = 1.0

    # NaN safety
    ahgo = ahgo if not np.isnan(ahgo) else 1.0
    ahgs = ahgs if not np.isnan(ahgs) else 0.0
    syah = syah if not np.isnan(syah) else 1.0
    kho  = kho  if not np.isnan(kho)  else 1.0
    acik_orani = acik_orani if not np.isnan(acik_orani) else 1.0

    # ─── Review defaults (location average) ───
    yorum_ort = float(yorum_ort_map.get(lokasyon_adi, 10.0))

    # ─── Create input vector ───
    yil_trendi = yil - 2022
    girdi = {
        "temp": temp, "prcp": prcp, "snow": snow, "rain": rain,
        "wspd": wspd, "rhum": rhum, "tatil_mi": int(tatil_mi),
        "ay": ay, "mevsim": mevsim,
        "ay_sin": ay_sin, "ay_cos": ay_cos,
        "hafta_sin": hafta_sin, "hafta_cos": hafta_cos,
        "yil_ici_hafta": yil_ici_hafta,
        "temp_kare":       temp ** 2,
        "sicak_yagmur":    temp * rain,
        "kar_soguk":       snow * int(temp < 0),
        "sicaklik_konfor": -((temp - 20) ** 2),
        "nem_sicaklik":    rhum * temp / 100,
        "kotu_hava":       rain * wspd,
        "yil_trendi":      yil_trendi,
        "buyume_egimi":    lok_ref["buyume_egimi"],
        "lokasyon_olcek":  lok_ref["lokasyon_olcek"],
        "buyume_x_trend":  lok_ref["buyume_egimi"] * yil_trendi,
        "tatil_yaz":       int(tatil_mi) * int(mevsim == 3),
        "tatil_ilkbahar":  int(tatil_mi) * int(mevsim == 2),
        "Rakım (m)":              lok_ref["Rakım (m)"],
        "Has. Mesafe (km)":       lok_ref["Has. Mesafe (km)"],
        "Ort. Eğim (%)":          lok_ref["Ort. Eğim (%)"],
        "Has. Varış Süresi (Dk)": lok_ref["Has. Varış Süresi (Dk)"],
        "ortalama_puan":          lok_ref["ortalama_puan"],
        "yorum_rolling4":         yorum_ort,
        "yorum_rolling8":         yorum_ort,
        "ayni_hafta_gecmis_ort":  ahgo,
        "ayni_hafta_gecmis_std":  ahgs,
        "son_yil_ayni_hafta":     syah,
        "komsu_hafta_ort":        kho,
        "acik_orani_gecmis":      acik_orani,
    }
    # One-hot columns
    for col in yol_bolge_cols:
        girdi[col] = int(lok_ref.get(col, 0))

    # Convert model input to DataFrame (in selected feature order)
    girdi_df = pd.DataFrame([girdi])[selected_features]
    log_pred = rf_production.predict(girdi_df)[0]
    rel_pred = float(np.expm1(log_pred))

    # ─── Growth scaling ───
    yillik_ort = float(yillik_ort_map[lokasyon_adi])
    egim_sinirli = float(np.clip(
        lok_ref["buyume_egimi"],
        -cfg["MAX_YEARLY_GROWTH"], cfg["MAX_YEARLY_GROWTH"]
    ))
    buyume_carpani = float(np.exp(egim_sinirli * (yil - cfg["REFERENCE_YEAR"])))
    yillik_ort_proj = yillik_ort * buyume_carpani

    # ─── Absolute prediction + open-ratio mask ───
    abs_tahmin_ham = rel_pred * yillik_ort_proj
    acik_carpan    = _acik_maskesi(acik_orani)
    tahmin_ziyaretci = max(0, int(abs_tahmin_ham * acik_carpan))

    return {
        "lokasyon": lokasyon_adi,
        "tahmin_ziyaretci": tahmin_ziyaretci,
        "yillik_ort": int(yillik_ort),
        "yillik_ort_projeksiyon": int(yillik_ort_proj),
        "buyume_carpani": round(buyume_carpani, 3),
        "goreceli_talep": round(rel_pred, 2),
        "mevsimsel_sinyal_ort":  round(ahgo, 2),
        "mevsimsel_belirsizlik": round(ahgs, 2),
        "acik_orani": round(acik_orani, 2),
        "acik_carpan": round(acik_carpan, 2),
    }


if __name__ == "__main__":
    # Test (with examples from the training data)
    art = load_artifact()
    print(f"✓ Artifact loaded: {len(art['all_lokasyonlar'])} locations")
    print(f"  Test R² (measured during training): {art['eval_metrics']['test_r2']:.4f}")
    print(f"  Test MAE: {art['eval_metrics']['test_mae']:,.0f} people")

    # Example prediction
    print("\n--- Yedigöller, July 2026 (summer week) ---")
    sonuc = predict_visitor(
        lokasyon_adi="Yedigöller Milli Parkı",
        yil=2026, ay=7, yil_ici_hafta=28,
        temp=20.0, prcp=5.0, rain=5.0, snow=0.0,
        wspd=8.0, rhum=70.0, tatil_mi=0,
    )
    for k, v in sonuc.items():
        print(f"  {k}: {v}")

    print("\n--- Horma Kanyonu, January 2026 (winter week) ---")
    sonuc = predict_visitor(
        lokasyon_adi="Horma Kanyonu",
        yil=2026, ay=1, yil_ici_hafta=3,
        temp=-2.0, prcp=20.0, rain=0.0, snow=20.0,
        wspd=12.0, rhum=85.0, tatil_mi=0,
    )
    for k, v in sonuc.items():
        print(f"  {k}: {v}")