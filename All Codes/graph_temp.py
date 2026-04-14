import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Her kamp alanı için sıcaklık grafiği 2024 ve 2025 (bar chart)

# Grafikler için genel estetik ayarları (Tamamen temiz)
plt.style.use('default')

def sicaklik_grafikleri_olustur(hedef_klasor):
    cikti_klasoru = "Kamp_Sicaklik_Grafikleri"
    if not os.path.exists(cikti_klasoru):
        os.makedirs(cikti_klasoru)

    print(f"\nVeriler '{hedef_klasor}' dizininde taranıyor...")

    try:
        tum_dosyalar = os.listdir(hedef_klasor)
    except FileNotFoundError:
        print(f"\nHATA: Girdiğiniz '{hedef_klasor}' adında bir klasör bilgisayarda bulunamadı. Yolu kontrol edin.")
        return

    hava_dosyalar = []
    for dosya in tum_dosyalar:
        if dosya.lower().endswith(".csv"):
            # Arama filtresi "hava" olarak güncellendi
            if "hava" in dosya.lower() or "merge" in dosya.lower():
                tam_yol = os.path.join(hedef_klasor, dosya)
                hava_dosyalar.append(tam_yol)

    if len(hava_dosyalar) == 0:
        print(f"\nHATA: Belirtilen klasörde '.csv' uzantılı ve isminde 'hava' veya 'merge' geçen dosya bulunamadı!")
        print("Python'un bu klasörde gördüğü dosyalar şunlar (İlk 10):")
        for d in tum_dosyalar[:10]:
            print(f" -> {d}")
        return

    print(f"Toplam {len(hava_dosyalar)} kamp alanı bulundu. Sıcaklık çizimleri başlıyor...\n")

    for dosya_yolu in hava_dosyalar:
        dosya_adi = os.path.basename(dosya_yolu)
        
        # --- İSİM TEMİZLEME MANTIĞI YENİ FORMATA GÖRE GÜNCELLENDİ ---
        if "hava" in dosya_adi.lower():
            # Örnek: "- Abant Gölü Tabiat Parkı - Bolu_final_hava.csv"
            isim = dosya_adi.replace("_final_hava.csv", "").strip()
            # Baştaki tireyi ve boşluğu at
            if isim.startswith("-"):
                isim = isim[1:].strip()
        else:
            isim = dosya_adi.replace("##", "").split("_haftal")[0].strip()

        df = pd.read_csv(dosya_yolu)

        tarih_sutunu = df.columns[0] 
        if 'tarih' in df.columns:
            tarih_sutunu = 'tarih'

        if df[tarih_sutunu].dtype == object and ' - ' in str(df[tarih_sutunu].iloc[0]):
            df['Gercek_Tarih'] = pd.to_datetime(df[tarih_sutunu].str.split(' - ').str[0])
        else:
            df['Gercek_Tarih'] = pd.to_datetime(df[tarih_sutunu])

        temp_col = None
        for col in df.columns:
            if 'temp' in col.lower():
                temp_col = col
                break

        if temp_col is None:
            print(f" -> {isim}: Sıcaklık verisi (temp) bulunamadı, atlanıyor.")
            continue

        df.set_index('Gercek_Tarih', inplace=True)
        haftalik_df = df.resample('W').mean(numeric_only=True).reset_index()

        fig, ax = plt.subplots(figsize=(16, 6))

        ax.bar(haftalik_df['Gercek_Tarih'], haftalik_df[temp_col], width=5, color='#e74c3c', edgecolor='black', alpha=0.85)

        ax.axhline(0, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7)

        plt.title(f"{isim} - Haftalık Ortalama Sıcaklık (°C) (5 Yıllık Trend)", fontsize=16, fontweight='bold', pad=25)
        plt.xlabel("")
        plt.ylabel("Ortalama Sıcaklık (°C)", fontsize=12, fontweight='bold')

        ax.xaxis.set_major_locator(mdates.YearLocator()) 
        ax.set_xticklabels([]) 
        ax.grid(visible=False, axis='both')
        ax.set_facecolor('white')
        fig.set_facecolor('white')
        ax.grid(visible=True, axis='y', color='gray', linestyle='-', alpha=0.15)

        haftalik_df['yil'] = haftalik_df['Gercek_Tarih'].dt.year
        unique_years = haftalik_df['yil'].unique()
        
        for yil in unique_years:
            year_data = haftalik_df[haftalik_df['yil'] == yil]
            if not year_data.empty:
                start_date = year_data['Gercek_Tarih'].min()
                end_date = year_data['Gercek_Tarih'].max()
                x_center = start_date + (end_date - start_date) / 2
                
                ax.text(x_center, -0.04, f"{yil}", 
                        transform=ax.get_xaxis_transform(), 
                        ha='center', va='top', fontsize=12, fontweight='bold', color='black')

        plt.subplots_adjust(bottom=0.15)
        
        temiz_isim = "".join(c for c in isim if c.isalnum() or c in (' ', '_', '-'))
        kayit_yolu = os.path.join(cikti_klasoru, f"{temiz_isim}_Sicaklik.png")
        
        plt.savefig(kayit_yolu, dpi=300) 
        plt.close(fig) 
        
        print(f" -> {isim} sıcaklık grafiği başarıyla oluşturuldu.")

    print(f"\nİŞLEM TAMAMLANDI! Bütün sıcaklık grafikleri '{cikti_klasoru}' klasörüne kaydedildi.")

if __name__ == "__main__":
    klasor_yolu = input("\nLütfen 5 yıllık verilerin bulunduğu klasörün tam yolunu yapıştırın:\n> ")
    klasor_yolu = klasor_yolu.strip('"').strip("'")
    sicaklik_grafikleri_olustur(klasor_yolu)
