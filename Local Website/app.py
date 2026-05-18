"""
Flask backend: Campsite Visitor Prediction System.

Endpoints:
  GET  /                       → Frontend (index.html)
  GET  /api/locations          → List of campsites (JSON)
  GET  /api/predict?id=X       → Makes a prediction:
                                 1. Fetches lat/lon/elevation from Location ID
                                 2. Gets next week's forecast from Open-Meteo
                                 3. Checks for holidays in Turkey
                                 4. Calls the ML model
                                 5. Returns JSON response

Run: python app.py  (then from browser http://localhost:5000)
"""

import os
import sys
import traceback
from datetime import date, timedelta
from flask import Flask, jsonify, render_template, request

from locations import get_locations_dict, get_location_by_id
from weather_forecast import get_weekly_forecast
from tatil_kontrolu import hafta_icinde_tatil_var_mi, tatil_gun_sayisi
from predictor import load_artifact, predict_visitor, get_artifact

app = Flask(__name__)

# Load ML model at startup (so it doesn't reload on every request)
print("Loading model artifact...")
load_artifact()
art = get_artifact()
TRAINED_LOCATIONS = set(art["all_lokasyonlar"])
print(f"✓ Model ready for {len(TRAINED_LOCATIONS)} locations")
print(f"  Test R²:  {art['eval_metrics']['test_r2']:.4f}")
print(f"  Test MAE: {art['eval_metrics']['test_mae']:,.0f} people")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/locations")
def api_locations():
    """Returns all campsites; marks those without training data."""
    locs = get_locations_dict()
    for loc in locs:
        loc["model_available"] = loc["name"] in TRAINED_LOCATIONS
    return jsonify({
        "locations": locs,
        "model_info": {
            "test_r2":  round(art["eval_metrics"]["test_r2"], 4),
            "test_mae": int(art["eval_metrics"]["test_mae"]),
            "n_train":  art["eval_metrics"]["n_train"],
        }
    })


@app.route("/api/predict")
def api_predict():
    """
    Generates prediction from Location ID.
    Query: ?id=X  (location id, 0-based)
    """
    try:
        loc_id = int(request.args.get("id", -1))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid id"}), 400

    loc = get_location_by_id(loc_id)
    if loc is None:
        return jsonify({"error": f"Location not found: id={loc_id}"}), 404

    if loc["name"] not in TRAINED_LOCATIONS:
        return jsonify({
            "error": f"No training data for '{loc['name']}'. The model cannot make predictions for this location.",
            "location": loc,
        }), 422

    # 1. Next week's forecast from Open-Meteo
    try:
        weather = get_weekly_forecast(
            lat=loc["latitude"],
            lon=loc["longitude"],
            elevation=loc["elevation_m"],
            start_offset_days=7,  # 1 week later
        )
    except Exception as e:
        return jsonify({
            "error": "Could not get weather forecast (Open-Meteo).",
            "detail": str(e),
        }), 502

    # 2. Date information
    start_date = date.fromisoformat(weather["start_date"])
    end_date   = date.fromisoformat(weather["end_date"])
    ay = start_date.month
    yil_ici_hafta = max(1, min(52, start_date.isocalendar().week))
    yil = start_date.year

    # 3. Holiday check
    tatil_var = hafta_icinde_tatil_var_mi(start_date, gun_sayisi=7)
    tatil_gun = tatil_gun_sayisi(start_date, gun_sayisi=7)

    # 4. ML prediction
    try:
        sonuc = predict_visitor(
            lokasyon_adi=loc["name"],
            yil=yil, ay=ay, yil_ici_hafta=yil_ici_hafta,
            temp=weather["temp"],
            prcp=weather["prcp"],
            rain=weather["rain"],
            snow=weather["snow"],
            wspd=weather["wspd"],
            rhum=weather["rhum"],
            tatil_mi=tatil_var,
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Error occurred during prediction.",
            "detail": str(e),
        }), 500

    return jsonify({
        "location": loc,
        "date_range": {
            "start": weather["start_date"],
            "end":   weather["end_date"],
            "year":  yil,
            "month": ay,
            "week_of_year": yil_ici_hafta,
        },
        "weather": {
            "temp": weather["temp"],
            "prcp": weather["prcp"],
            "rain": weather["rain"],
            "snow": weather["snow"],
            "wspd": weather["wspd"],
            "rhum": weather["rhum"],
            "daily": weather["days"],
        },
        "holiday": {
            "has_holiday": bool(tatil_var),
            "holiday_days": tatil_gun,
        },
        "prediction": sonuc,
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


if __name__ == "__main__":
    # In production run with debug=False, using gunicorn.
    # Defaulting to port 5001 because port 5000 is used by AirPlay Receiver on macOS.
    app.run(host="0.0.0.0", port=5001, debug=True)