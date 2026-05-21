"""
Open-Meteo Forecast API wrapper.

Usage:
    >>> from weather_forecast import get_weekly_forecast
    >>> data = get_weekly_forecast(lat=40.93, lon=31.74, start_offset_days=7)
    >>> # Aggregate of the week from day 7 to day 14

Open-Meteo:
  - Free, no API key required
  - 16-day forecast (ECMWF + ICON + other models)
  - Has daily and hourly endpoints

Output (weekly aggregate, in ML model format):
  - temp:  weekly average temperature (°C)
  - prcp:  weekly total precipitation (mm)
  - rain:  weekly total rain (mm)
  - snow:  weekly total snow (mm WATER EQUIVALENT — directly from API)
  - wspd:  weekly average wind speed (km/h — average of hourly wind_speed_10m)
  - rhum:  weekly average relative humidity (%)

Training/inference consistency:
  - It is recommended to choose a fixed model with the `model` parameter 
    (e.g., "ecmwf_ifs025"). Otherwise, Open-Meteo uses best_match and 
    a different model may be selected depending on the location.
  - Aggregated from hourly data for wind and snow; this ensures consistency 
    with the "average wind" / "mm water equivalent snow" definitions in the 
    training set.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from datetime import date, timedelta
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Forecast endpoint provides 16 days. For a 7-day window,
# start_offset must be between 0..9 (if starting on day 9, 7 days are completed by day 15).
MAX_START_OFFSET = 9


def _session_with_retries(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """Requests session with retry/backoff: resilient to temporary 5xx and connection errors."""
    sess = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def _safe_mean(values):
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _safe_sum(values):
    return sum(v for v in values if v is not None)


def get_weekly_forecast(lat: float, lon: float,
                        elevation: Optional[float] = None,
                        start_offset_days: int = 7,
                        timezone: str = "Europe/Istanbul",
                        model: Optional[str] = None,
                        timeout: int = 15) -> dict:
    """
    Fetches the aggregate weather forecast for the 7-day week starting 
    `start_offset_days` days from today for the given location.

    Args:
        lat, lon:           Coordinates.
        elevation:          Elevation (optional). Open-Meteo uses this for 
                            temperature downscaling.
        start_offset_days:  Start of the forecast week (between 0..9; default 7).
        timezone:           Timezone (e.g., Europe/Istanbul for TR).
        model:              Open-Meteo model (e.g., "ecmwf_ifs025", "icon_eu").
                            If None, best_match (automatic) is used.
                            Fixing a single model is recommended for ML 
                            training/inference consistency.
        timeout:            HTTP timeout (seconds).

    Returns:
        {
            "temp", "prcp", "rain", "snow", "wspd", "rhum": float,
            "start_date": "YYYY-MM-DD",
            "end_date":   "YYYY-MM-DD",
            "model":      str,
            "days": [{...daily details...}, ...],
        }

    Raises:
        ValueError:                  Invalid argument or missing forecast.
        requests.HTTPError:          If the API returns an error code.
        requests.RequestException:   Network error (after retries are exhausted).
    """
    # ---- Input validation ----------------------------------------------
    if not isinstance(start_offset_days, int):
        raise TypeError("start_offset_days must be int.")
    if not (0 <= start_offset_days <= MAX_START_OFFSET):
        raise ValueError(
            f"start_offset_days={start_offset_days} is out of bounds. "
            f"Open-Meteo provides 16 days; for a 7-day window, it "
            f"must be between 0..{MAX_START_OFFSET}."
        )
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")

    # ---- Forecast window -----------------------------------------------
    start_date = date.today() + timedelta(days=start_offset_days)
    end_date = start_date + timedelta(days=6)

    # ---- API request parameters ----------------------------------------
    # Daily: temperature, precipitation, humidity aggregates (canonical names — underscored)
    # Hourly: for wind and snow water equivalent (we will do our own aggregation)
    #
    # wspd: we take the daily and weekly average of hourly wind_speed_10m.
    #       This is consistent with the average wind speed definition in the training data.
    #       (Original code used daily max — semantic mismatch.)
    #
    # snow: hourly snowfall_water_equivalent (mm) directly gives water equivalent.
    #       Thus, approximate conversions like "1 cm snow ≈ 1 mm water" are not needed.
    params = {
        "latitude":  lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",                  # cm — for info, will not be used instead of snow
            "wind_speed_10m_max",            # for info
            "relative_humidity_2m_mean",
        ]),
        "hourly": ",".join([
            "wind_speed_10m",                # km/h (with wind_speed_unit)
            "snowfall_water_equivalent",     # mm — actual water equivalent
        ]),
        "start_date":      start_date.isoformat(),
        "end_date":        end_date.isoformat(),
        "timezone":        timezone,
        "wind_speed_unit": "kmh",
    }
    if elevation is not None:
        params["elevation"] = elevation
    if model is not None:
        params["models"] = model

    # ---- HTTP request --------------------------------------------------
    session = _session_with_retries()
    resp = session.get(OPEN_METEO_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if "daily" not in data:
        raise ValueError(f"Open-Meteo response is missing 'daily': {data}")
    if "hourly" not in data:
        raise ValueError(f"Open-Meteo response is missing 'hourly': {data}")

    daily = data["daily"]
    hourly = data["hourly"]

    n_days = len(daily["time"])
    if n_days < 7:
        raise ValueError(f"Insufficient forecast data: {n_days} days (7 expected)")

    # ---- Hourly -> daily grouping --------------------------------------
    # Open-Meteo returns hours in "YYYY-MM-DDTHH:MM" format according to timezone.
    # Grouping hours by date.
    hourly_times = hourly["time"]
    wind_hourly = hourly.get("wind_speed_10m", [])
    snow_we_hourly = hourly.get("snowfall_water_equivalent", [])

    wind_by_day: dict = {}
    snow_we_by_day: dict = {}
    for i, t in enumerate(hourly_times):
        day = t.split("T")[0]
        if i < len(wind_hourly) and wind_hourly[i] is not None:
            wind_by_day.setdefault(day, []).append(wind_hourly[i])
        if i < len(snow_we_hourly) and snow_we_hourly[i] is not None:
            snow_we_by_day.setdefault(day, []).append(snow_we_hourly[i])

    # ---- Daily details (for debug) -------------------------------------
    daily_rows = []
    for i in range(n_days):
        day_str = daily["time"][i]
        daily_rows.append({
            "date":         day_str,
            "temp_max":     daily["temperature_2m_max"][i],
            "temp_min":     daily["temperature_2m_min"][i],
            "temp_mean":    daily["temperature_2m_mean"][i],
            "prcp":         daily["precipitation_sum"][i],
            "rain":         daily["rain_sum"][i],
            "snow_cm":      daily["snowfall_sum"][i],                       # info: snow depth (cm)
            "snow_mm_we":   sum(snow_we_by_day.get(day_str, [])),           # actual value (mm water eq.)
            "wspd_max":     daily["wind_speed_10m_max"][i],                 # info: daily max
            "wspd_mean":    _safe_mean(wind_by_day.get(day_str, [])),       # for ML: daily average
            "rhum":         daily["relative_humidity_2m_mean"][i],
        })

    # ---- Weekly aggregate (ML model format) ----------------------------
    weekly = {
        "temp": round(_safe_mean([r["temp_mean"]  for r in daily_rows]), 2),
        "prcp": round(_safe_sum( [r["prcp"]       for r in daily_rows]), 2),
        "rain": round(_safe_sum( [r["rain"]       for r in daily_rows]), 2),
        "snow": round(_safe_sum( [r["snow_mm_we"] for r in daily_rows]), 2),
        "wspd": round(_safe_mean([r["wspd_mean"]  for r in daily_rows]), 2),
        "rhum": round(_safe_mean([r["rhum"]       for r in daily_rows]), 2),
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "model":      model or "best_match",
        "days":       daily_rows,
    }

    return weekly


if __name__ == "__main__":
    # Test (requires internet connection)
    print("Test: Fetching next week's forecast for Yedigöller (Bolu)...")
    try:
        w = get_weekly_forecast(40.937, 31.741, elevation=900)
        print(f"  Date:                 {w['start_date']} → {w['end_date']}")
        print(f"  Model:                {w['model']}")
        print(f"  Average temperature:  {w['temp']}°C")
        print(f"  Total precipitation:  {w['prcp']} mm")
        print(f"  Total rain:           {w['rain']} mm")
        print(f"  Total snow (w.e.):    {w['snow']} mm")
        print(f"  Average wind:         {w['wspd']} km/h")
        print(f"  Average humidity:     {w['rhum']}%")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
