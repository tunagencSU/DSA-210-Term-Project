"""
Official public holidays in Turkey.
Fixed holidays are programmatic, religious holidays (hijri calendar) are hardcoded.
Scope: 2024-2027.
"""

from datetime import date, timedelta

# Fixed non-religious public holidays (same every year)
_SABIT_TATILLER = [
    (1, 1),    # New Year's Day
    (4, 23),   # National Sovereignty and Children's Day
    (5, 1),    # Labor and Solidarity Day
    (5, 19),   # Commemoration of Atatürk, Youth and Sports Day
    (7, 15),   # Democracy and National Unity Day
    (8, 30),   # Victory Day
    (10, 29),  # Republic Day
]

# Religious holidays (Official calendar of the Directorate of Religious Affairs)
# Format: (year, month, day, duration_in_days, name)
_DINI_BAYRAMLAR = [
    # 2024
    (2024, 4, 10, 3, "Eid al-Fitr"),
    (2024, 6, 16, 4, "Eid al-Adha"),
    # 2025
    (2025, 3, 30, 3, "Eid al-Fitr"),
    (2025, 6, 6,  4, "Eid al-Adha"),
    # 2026
    (2026, 3, 20, 3, "Eid al-Fitr"),
    (2026, 5, 27, 4, "Eid al-Adha"),
    # 2027
    (2027, 3, 9,  3, "Eid al-Fitr"),
    (2027, 5, 16, 4, "Eid al-Adha"),
]


def _tum_tatil_tarihleri(yil):
    """Returns all holiday dates for a given year as a set."""
    tatiller = set()
    # Fixed holidays
    for ay, gun in _SABIT_TATILLER:
        tatiller.add(date(yil, ay, gun))
    # Religious holidays
    for byil, bay, bgun, sure, _ in _DINI_BAYRAMLAR:
        if byil == yil:
            for i in range(sure):
                tatiller.add(date(byil, bay, bgun) + timedelta(days=i))
    return tatiller


def hafta_icinde_tatil_var_mi(baslangic_tarihi, gun_sayisi=7):
    """
    Is there a public holiday within `gun_sayisi` days starting from the given date?
    Weekends (Sat/Sun) ARE NOT CONSIDERED HOLIDAYS - only official public holidays.
    
    Returns:
        1 (holiday exists) or 0 (no holiday)
    """
    for i in range(gun_sayisi):
        gun = baslangic_tarihi + timedelta(days=i)
        if gun in _tum_tatil_tarihleri(gun.year):
            return 1
    return 0


def tatil_gun_sayisi(baslangic_tarihi, gun_sayisi=7):
    """Returns the number of holiday days in the given range (for detailed info)."""
    sayi = 0
    for i in range(gun_sayisi):
        gun = baslangic_tarihi + timedelta(days=i)
        if gun in _tum_tatil_tarihleri(gun.year):
            sayi += 1
    return sayi


if __name__ == "__main__":
    # Test
    from datetime import date
    test = date(2025, 6, 6)
    print(f"7 days from June 6, 2025: is there a holiday? {hafta_icinde_tatil_var_mi(test)}")
    print(f"Number of holiday days: {tatil_gun_sayisi(test)}")
    
    test2 = date(2026, 7, 1)
    print(f"\n7 days from July 1, 2026: is there a holiday? {hafta_icinde_tatil_var_mi(test2)}")
    print(f"Number of holiday days: {tatil_gun_sayisi(test2)}")