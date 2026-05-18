import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Generates EDA (Exploratory Data Analysis) graphs — 4 graphs per camp area

# 1. Build file paths dynamically
# Find the directory where the code is running (ALL CODES folder)
mevcut_klasor = os.path.dirname(os.path.abspath(__file__))

# Go up one directory (DSA_PROJE_ML folder)
ana_klasor = os.path.dirname(mevcut_klasor)

# Build paths for the input and output folders
hedef_klasor = os.path.join(ana_klasor, "## Merged Data")
output_dir = os.path.join(ana_klasor, "EDA_Grafikleri")

# Check whether the folder exists
if not os.path.exists(hedef_klasor):
    print(f"Error: '{hedef_klasor}' folder not found. Please check the folder structure.")
else:
    # Find *_haftalık_merge.csv files in the target folder
    arama_deseni = os.path.join(hedef_klasor, "*_haftalık_merge.csv")
    dosya_yollari = glob.glob(arama_deseni)

    if len(dosya_yollari) == 0:
        print("No files with '_haftalık_merge.csv' extension found in the specified folder.")
    else:
        print(f"Total of {len(dosya_yollari)} camp areas found. Starting analysis...\n")

        # Create a main folder inside DSA_PROJE_ML to keep the outputs organized
        os.makedirs(output_dir, exist_ok=True)

        for dosya in dosya_yollari:
            # Dynamically extract the camp name from the file name
            kamp_adi_ham = os.path.basename(dosya).replace("_haftalık_merge.csv", "")
            kamp_prefix = kamp_adi_ham.replace("##", "")

            print(f"Processing {kamp_prefix}...")

            # Load the data
            df = pd.read_csv(dosya)

            # Process the date column
            df['baslangic_tarihi'] = df['tarih'].apply(lambda x: str(x).split(' - ')[0])
            df['baslangic_tarihi'] = pd.to_datetime(df['baslangic_tarihi'], format='%Y.%m.%d')
            df = df.sort_values('baslangic_tarihi')

            # Create a dedicated subfolder for this camp area
            kamp_klasoru = os.path.join(output_dir, kamp_prefix)
            os.makedirs(kamp_klasoru, exist_ok=True)

            # -------------------------------------------------------------------
            # 1. Visitor count over time
            # -------------------------------------------------------------------
            plt.figure(figsize=(12, 5))
            plt.plot(df['baslangic_tarihi'], df['gercek_ziyaretci'], marker='o', linestyle='-', color='b', markersize=4)
            plt.title(f'Weekly Actual Visitor Count Over Time ({kamp_prefix})')
            plt.xlabel('Date')
            plt.ylabel('Visitor Count')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(kamp_klasoru, '1_ziyaretci_zaman_serisi.png'))
            plt.close()

            # -------------------------------------------------------------------
            # 2. Weather vs visitor count comparison (dual y-axis)
            # -------------------------------------------------------------------
            fig, ax1 = plt.subplots(figsize=(12, 5))

            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Visitor Count', color=color)
            ax1.plot(df['baslangic_tarihi'], df['gercek_ziyaretci'], color=color, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.tick_params(axis='x', rotation=45)

            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Temperature (°C)', color=color)
            
            temp_col = f'{kamp_prefix}_temp'
            if temp_col in df.columns:
                ax2.plot(df['baslangic_tarihi'], df[temp_col], color=color, linestyle='--', linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title(f'Weekly Visitor Count vs Temperature ({kamp_prefix})')
            fig.tight_layout()
            plt.savefig(os.path.join(kamp_klasoru, '2_ziyaretci_sicaklik_karsilastirma.png'))
            plt.close()

            # -------------------------------------------------------------------
            # 3. Correlation Heatmap
            # -------------------------------------------------------------------
            beklenen_sutunlar = {
                'gercek_ziyaretci': 'Visitors',
                'yorum_sayisi': 'Reviews',
                'ortalama_puan': 'Rating',
                f'{kamp_prefix}_temp': 'Temp',
                f'{kamp_prefix}_prcp': 'Precip(Tot)',
                f'{kamp_prefix}_snow': 'Snow',
                f'{kamp_prefix}_rain': 'Rain',
                f'{kamp_prefix}_wspd': 'Wind',
                f'{kamp_prefix}_rhum': 'Humidity'
            }

            # Keep only columns that actually exist in the CSV
            mevcut_sutunlar = {k: v for k, v in beklenen_sutunlar.items() if k in df.columns}
            corr_df = df[list(mevcut_sutunlar.keys())].rename(columns=mevcut_sutunlar)
            corr = corr_df.corr()

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
            plt.title(f'Correlation Matrix Between Variables ({kamp_prefix})')
            plt.tight_layout()
            plt.savefig(os.path.join(kamp_klasoru, '3_korelasyon_matrisi.png'))
            plt.close()

            # -------------------------------------------------------------------
            # 4. Visitor distribution and scatter plots
            # -------------------------------------------------------------------
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            sns.histplot(df['gercek_ziyaretci'], bins=20, kde=True, ax=axes[0], color='purple')
            axes[0].set_title('Visitor Count Distribution')
            axes[0].set_xlabel('Weekly Visitors')
            axes[0].set_ylabel('Frequency')

            if temp_col in df.columns:
                sns.scatterplot(x=df[temp_col], y=df['gercek_ziyaretci'], ax=axes[1])
                axes[1].set_title('Temperature vs Visitors')
                axes[1].set_xlabel('Temperature (°C)')
                axes[1].set_ylabel('Visitor Count')

            prcp_col = f'{kamp_prefix}_prcp'
            if prcp_col in df.columns:
                sns.scatterplot(x=df[prcp_col], y=df['gercek_ziyaretci'], ax=axes[2])
                axes[2].set_title('Precipitation vs Visitors')
                axes[2].set_xlabel('Total Precipitation')
                axes[2].set_ylabel('Visitor Count')

            plt.suptitle(f'Distribution and Relationship Graphs ({kamp_prefix})', y=1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(kamp_klasoru, '4_dagilim_ve_scatter.png'))
            plt.close()

        print(f"\nAll analyses completed! Graphs saved to '{output_dir}' folder.")