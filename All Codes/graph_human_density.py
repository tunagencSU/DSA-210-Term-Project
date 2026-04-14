import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os

# Grafikler için genel estetik ayarları
plt.style.use('default')

def grafikleri_temiz_olustur(hedef_klasor):
    cikti_klasoru = "Kamp_Grafikleri_Temiz"
    if not os.path.exists(cikti_klasoru):
        os.makedirs(cikti_klasoru)
    
    ay_adlari = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
        7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
    }

    print(f"\nVeriler '{hedef_klasor}' dizininde taranıyor...")
    
    arama_yolu = os.path.join(hedef_klasor, "*.csv")
    tum_csvler = glob.glob(arama_yolu)
    haftalik_dosyalar = [dosya for dosya in tum_csvler if "merge" in dosya.lower() and "haftal" in dosya.lower()]
    
    if len(haftalik_dosyalar) == 0:
        print(f"\nHATA: Belirtilen klasörde veri bulunamadı!")
        return
        
    print(f"Toplam {len(haftalik_dosyalar)} kamp alanı bulundu. Çizimler başlıyor...\n")
    
    for dosya_yolu in haftalik_dosyalar:
        dosya_adi = os.path.basename(dosya_yolu)
        isim = dosya_adi.replace("##", "").split("_haftal")[0].strip()
        
        df = pd.read_csv(dosya_yolu)
        df['tarih_baslangic'] = pd.to_datetime(df['tarih'].str.split(' - ').str[0])
        df = df.sort_values('tarih_baslangic')
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        ax.bar(df['tarih_baslangic'], df['gercek_ziyaretci'], width=5, color='#2c7bb6', edgecolor='black', alpha=0.85)
        
        plt.title(f"{isim} - Haftalık Tahmini Ziyaretçi Sayısı (2024-2025)", fontsize=16, fontweight='bold', pad=25)
        plt.xlabel("")
        plt.ylabel("Tahmini Toplam Ziyaretçi", fontsize=12, fontweight='bold')
        
        # 1. Standart x ekseni yazılarını sil
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
        ax.set_xticklabels([]) 
        
        # 2. Grid ve arka plan temizliği
        ax.grid(visible=False, axis='both')
        ax.set_facecolor('white')
        fig.set_facecolor('white')
        ax.grid(visible=True, axis='y', color='gray', linestyle='-', alpha=0.15)
        
        # --- MANÜEL AY İŞARETÇİLERİ (DİKEY & ŞARTLI YIL) ---
        df['yil'] = df['tarih_baslangic'].dt.year
        df['ay'] = df['tarih_baslangic'].dt.month
        unique_months = df[['yil', 'ay']].drop_duplicates()
        
        for _, row in unique_months.iterrows():
            yil = row['yil']
            ay = row['ay']
            month_data = df[(df['yil'] == yil) & (df['ay'] == ay)]
            
            if not month_data.empty:
                start_date = month_data['tarih_baslangic'].min()
                end_date = month_data['tarih_baslangic'].max()
                x_center = start_date + (end_date - start_date) / 2
                
                # Sadece Ocak aylarında yılı belirt
                # Sadece Ocak aylarında yılı belirt
                if ay == 1:
                    month_label = f"{ay_adlari[ay]} {yil}"
                else:
                    month_label = f"{ay_adlari[ay]}"
                
                # rotation=90 ile yazıyı dikey hale getiriyoruz
                ax.text(x_center, -0.02, month_label, 
                        transform=ax.get_xaxis_transform(), 
                        ha='center', va='top', rotation=75, fontsize=10, color='black')
        
        # Dikey yazılar sığsın diye alt marjı biraz artırdık (0.20)
        plt.subplots_adjust(bottom=0.20)
        
        # Dosyayı kaydet
        temiz_isim = "".join(c for c in isim if c.isalnum() or c in (' ', '_', '-'))
        kayit_yolu = os.path.join(cikti_klasoru, f"{temiz_isim}_Temiz_Ziyaretci.png")
        
        plt.savefig(kayit_yolu, dpi=300) 
        plt.close(fig) 
        
        print(f" -> {isim} grafiği başarıyla temizlendi.")
        
    print(f"\nİŞLEM TAMAMLANDI! Bütün temiz grafikler '{cikti_klasoru}' klasörüne kaydedildi.")

if __name__ == "__main__":
    klasor_yolu = input("\nLütfen haftalık merge verilerinin bulunduğu klasörün tam yolunu yapıştırın:\n> ")
    klasor_yolu = klasor_yolu.strip('"').strip("'")
    grafikleri_temiz_olustur(klasor_yolu)