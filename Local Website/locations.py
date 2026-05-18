"""
Campsite catalog.
16 campsites in the Black Sea region; all locations with historical data 
in the training set are here.
"""

# Each location: (name, city, lat, lon, elevation_m)
KAMP_ALANLARI = [
    ("Borçka Karagöl Tabiat Parkı",     "Artvin",    41.3868298981645,   41.85432063282449, 1471),
    ("Elevit Yaylası",                  "Rize",      40.85574975606613,  41.01496137115783, 1944),
    ("Şaşvat karagöl",                  "Artvin",    41.30857261266539,  42.48350063237095, 1630),
    ("Hıdırnebi yaylası",               "Trabzon",   40.954930969360014, 39.4317744903852,  1413),
    ("Kuzalan şelalesi Tabiat Parkı",   "Giresun",   40.63228823575366,  38.390547083727334, 807),
    ("Kümbet yaylası",                  "Giresun",   40.559880959408574, 38.437963480566744, 1705),
    ("Perşembe Yaylası",                "Ordu",      40.62810003954787,  37.29670445499646, 1444),
    ("Ulugöl Tabiat Parkı",             "Ordu",      40.62920406691162,  37.5466277261601,  1224),
    ("Şahinkaya Kanyonu",               "Samsun",    41.279517520297404, 35.42883284667845,  188),
    ("Erfelek Tatlıca Şelaleleri",      "Sinop",     41.840495852841016, 34.779778739694976, 491),
    ("Horma Kanyonu",                   "Kastamonu", 41.63418796514924,  33.143329797360096, 630),
    ("Valla Kanyonu",                   "Kastamonu", 41.72009305981009,  33.07314276852683,  498),
    ("Yedigöller Milli Parkı",          "Bolu",      40.93742864391557,  31.740793112680358, 900),
    ("Abant Gölü Tabiat Parkı",         "Bolu",      40.61213667457994,  31.278819154995922, 1335),
    ("Güzeldere selalesi Tabiat Parkı", "Düzce",     40.72351477470955,  31.04999354336356,  651),
    ("Gölcük Tabiat Parkı",             "Bolu",      40.658060951840355, 31.629687654997547, 1217),
]


def get_locations_dict():
    """Returns all locations as a list of dicts (for the frontend)."""
    return [
        {
            "id": idx,
            "name": name,
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elev,
        }
        for idx, (name, city, lat, lon, elev) in enumerate(KAMP_ALANLARI)
    ]


def get_location_by_id(loc_id):
    """Returns location details by ID."""
    if 0 <= loc_id < len(KAMP_ALANLARI):
        name, city, lat, lon, elev = KAMP_ALANLARI[loc_id]
        return {
            "id": loc_id,
            "name": name,
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elev,
        }
    return None