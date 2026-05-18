import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os

# Bar chart of weekly estimated visitor counts for the years 2022, 2023, 2024 and 2025, per camp area

# General aesthetic settings for the plots
plt.style.use('default')

def grafikleri_temiz_olustur(hedef_klasor):
    # Since hedef_klasor holds the "## Merged data" path, one level up will be "DSA_PROJE_ML".
    ana_klasor = os.path.dirname(hedef_klasor) 
    
    # Create the output folder directly inside DSA_PROJE_ML.
    cikti_klasoru = os.path.join(ana_klasor, "EDA_GRAPHS_HUMAN_DENSİTY")
    
    if not os.path.exists(cikti_klasoru):
        os.makedirs(cikti_klasoru)
    
    ay_adlari = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }

    print(f"\nScanning data in '{hedef_klasor}' directory...")
    
    arama_yolu = os.path.join(hedef_klasor, "*.csv")
    tum_csvler = glob.glob(arama_yolu)
    haftalik_dosyalar = [dosya for dosya in tum_csvler if "merge" in dosya.lower() and "haftal" in dosya.lower()]
    
    if len(haftalik_dosyalar) == 0:
        print(f"\nERROR: No data found in the specified folder!")
        return
        
    print(f"Total of {len(haftalik_dosyalar)} camp areas found. Plotting starts...\n")
    
    for dosya_yolu in haftalik_dosyalar:
        dosya_adi = os.path.basename(dosya_yolu)
        isim = dosya_adi.replace("##", "").split("_haftal")[0].strip()
        
        df = pd.read_csv(dosya_yolu)
        df['tarih_baslangic'] = pd.to_datetime(df['tarih'].str.split(' - ').str[0])
        df = df.sort_values('tarih_baslangic')
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        ax.bar(df['tarih_baslangic'], df['gercek_ziyaretci'], width=5, color='#2c7bb6', edgecolor='black', alpha=0.85)
        
        plt.title(f"{isim} - Weekly Estimated Visitor Count (2024-2025)", fontsize=16, fontweight='bold', pad=25)
        plt.xlabel("")
        plt.ylabel("Estimated Total Visitors", fontsize=12, fontweight='bold')
        
        # 1. Remove the standard x-axis tick labels
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
        ax.set_xticklabels([]) 
        
        # 2. Clean up the grid and background
        ax.grid(visible=False, axis='both')
        ax.set_facecolor('white')
        fig.set_facecolor('white')
        ax.grid(visible=True, axis='y', color='gray', linestyle='-', alpha=0.15)
        
        # --- MANUAL MONTH MARKERS (VERTICAL & CONDITIONAL YEAR) ---
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
                
                # Only show the year on January
                if ay == 1:
                    month_label = f"{ay_adlari[ay]} {yil}"
                else:
                    month_label = f"{ay_adlari[ay]}"
                
                # Make the text vertical with rotation=90
                ax.text(x_center, -0.02, month_label, 
                        transform=ax.get_xaxis_transform(), 
                        ha='center', va='top', rotation=75, fontsize=10, color='black')
        
        # Increase the bottom margin a bit (0.20) so the vertical labels fit
        plt.subplots_adjust(bottom=0.20)
        
        # Save the file
        temiz_isim = "".join(c for c in isim if c.isalnum() or c in (' ', '_', '-'))
        kayit_yolu = os.path.join(cikti_klasoru, f"{temiz_isim}.png")
        
        plt.savefig(kayit_yolu, dpi=300) 
        plt.close(fig) 
        
        print(f" -> {isim} graph cleaned successfully.")
        
    print(f"\nOPERATION COMPLETED! All cleaned graphs saved to '{cikti_klasoru}' folder.")

if __name__ == "__main__":
    klasor_yolu = input("\nPlease paste the full path of the folder containing the weekly merge data:\n> ")
    klasor_yolu = klasor_yolu.strip('"').strip("'")
    grafikleri_temiz_olustur(klasor_yolu)