import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# EDA (Exploratory Data Analysis) Grafikleri Oluşturma, her kamp alanı için 4 adet grafik

# ziyaretçi grafiği
# ziyaretçi - sıcaklık grafiği (çift eksenli)
# korelasyon matrisi (heatmap)
# ziyaretçi dağılımı histogramı ve scatter plotlar (sıcaklık, yağış vs ziyaretçi)


# 1. Kullanıcıdan veri klasörünün yolunu al
hedef_klasor = input("Lütfen CSV dosyalarının bulunduğu klasörün yolunu (örneğin: C:\\Dosyalar\\Kamp_Verileri) girin: ")

# Tırnak işaretleriyle kopyalama ihtimaline karşı temizlik yap
hedef_klasor = hedef_klasor.strip('"').strip("'")

# Klasörün var olup olmadığını kontrol et
if not os.path.exists(hedef_klasor):
    print("Hata: Belirtilen klasör bulunamadı. Lütfen yolu kontrol edip tekrar deneyin.")
else:
    # Hedef klasördeki *_haftalık_merge.csv dosyalarını bul
    arama_deseni = os.path.join(hedef_klasor, "*_haftalık_merge.csv")
    dosya_yollari = glob.glob(arama_deseni)

    if len(dosya_yollari) == 0:
        print("Belirtilen klasörde '_haftalık_merge.csv' uzantılı dosya bulunamadı.")
    else:
        print(f"Toplam {len(dosya_yollari)} kamp alanı bulundu. Analiz başlatılıyor...\n")

        # Çıktıları düzenli tutmak için veri klasörünün içine bir ana klasör oluştur
        output_dir = os.path.join(hedef_klasor, "EDA_Grafikleri")
        os.makedirs(output_dir, exist_ok=True)

        for dosya in dosya_yollari:
            # Dosya adından kamp ismini dinamik olarak çıkar 
            kamp_adi_ham = os.path.basename(dosya).replace("_haftalık_merge.csv", "")
            kamp_prefix = kamp_adi_ham.replace("##", "")

            print(f"{kamp_prefix} işleniyor...")

            # Veriyi yükle
            df = pd.read_csv(dosya)

            # Tarih sütununu işle
            df['baslangic_tarihi'] = df['tarih'].apply(lambda x: str(x).split(' - ')[0])
            df['baslangic_tarihi'] = pd.to_datetime(df['baslangic_tarihi'], format='%Y.%m.%d')
            df = df.sort_values('baslangic_tarihi')

            # Bu kamp alanına özel bir alt klasör oluştur
            kamp_klasoru = os.path.join(output_dir, kamp_prefix)
            os.makedirs(kamp_klasoru, exist_ok=True)

            # -------------------------------------------------------------------
            # 1. Ziyaretçi Sayısının Zaman İçindeki Değişimi
            # -------------------------------------------------------------------
            plt.figure(figsize=(12, 5))
            plt.plot(df['baslangic_tarihi'], df['gercek_ziyaretci'], marker='o', linestyle='-', color='b', markersize=4)
            plt.title(f'Zaman İçinde Haftalık Gerçek Ziyaretçi Sayısı ({kamp_prefix})')
            plt.xlabel('Tarih')
            plt.ylabel('Ziyaretçi Sayısı')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(kamp_klasoru, '1_ziyaretci_zaman_serisi.png'))
            plt.close()

            # -------------------------------------------------------------------
            # 2. Hava Durumu ve Ziyaretçi Sayısı Karşılaştırması (Çift Eksenli)
            # -------------------------------------------------------------------
            fig, ax1 = plt.subplots(figsize=(12, 5))

            color = 'tab:blue'
            ax1.set_xlabel('Tarih')
            ax1.set_ylabel('Ziyaretçi Sayısı', color=color)
            ax1.plot(df['baslangic_tarihi'], df['gercek_ziyaretci'], color=color, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.tick_params(axis='x', rotation=45)

            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Sıcaklık (°C)', color=color)
            
            temp_col = f'{kamp_prefix}_temp'
            if temp_col in df.columns:
                ax2.plot(df['baslangic_tarihi'], df[temp_col], color=color, linestyle='--', linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title(f'Haftalık Ziyaretçi Sayısı ve Sıcaklık İlişkisi ({kamp_prefix})')
            fig.tight_layout()
            plt.savefig(os.path.join(kamp_klasoru, '2_ziyaretci_sicaklik_karsilastirma.png'))
            plt.close()

            # -------------------------------------------------------------------
            # 3. Korelasyon Isı Haritası (Heatmap)
            # -------------------------------------------------------------------
            beklenen_sutunlar = {
                'gercek_ziyaretci': 'Ziyaretçi',
                'yorum_sayisi': 'Yorum',
                'ortalama_puan': 'Puan',
                f'{kamp_prefix}_temp': 'Sıcaklık',
                f'{kamp_prefix}_prcp': 'Yağış(Top)',
                f'{kamp_prefix}_snow': 'Kar',
                f'{kamp_prefix}_rain': 'Yağmur',
                f'{kamp_prefix}_wspd': 'Rüzgar',
                f'{kamp_prefix}_rhum': 'Nem'
            }

            # Sadece CSV'de var olan sütunları al
            mevcut_sutunlar = {k: v for k, v in beklenen_sutunlar.items() if k in df.columns}
            corr_df = df[list(mevcut_sutunlar.keys())].rename(columns=mevcut_sutunlar)
            corr = corr_df.corr()

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
            plt.title(f'Değişkenler Arası Korelasyon Matrisi ({kamp_prefix})')
            plt.tight_layout()
            plt.savefig(os.path.join(kamp_klasoru, '3_korelasyon_matrisi.png'))
            plt.close()

            # -------------------------------------------------------------------
            # 4. Ziyaretçi Dağılımı ve Scatter Plotlar
            # -------------------------------------------------------------------
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            sns.histplot(df['gercek_ziyaretci'], bins=20, kde=True, ax=axes[0], color='purple')
            axes[0].set_title('Ziyaretçi Sayısı Dağılımı')
            axes[0].set_xlabel('Haftalık Ziyaretçi')
            axes[0].set_ylabel('Frekans')

            if temp_col in df.columns:
                sns.scatterplot(x=df[temp_col], y=df['gercek_ziyaretci'], ax=axes[1])
                axes[1].set_title('Sıcaklık vs Ziyaretçi')
                axes[1].set_xlabel('Sıcaklık (°C)')
                axes[1].set_ylabel('Ziyaretçi Sayısı')

            prcp_col = f'{kamp_prefix}_prcp'
            if prcp_col in df.columns:
                sns.scatterplot(x=df[prcp_col], y=df['gercek_ziyaretci'], ax=axes[2])
                axes[2].set_title('Yağış vs Ziyaretçi')
                axes[2].set_xlabel('Toplam Yağış')
                axes[2].set_ylabel('Ziyaretçi Sayısı')

            plt.suptitle(f'Dağılım ve İlişki Grafikleri ({kamp_prefix})', y=1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(kamp_klasoru, '4_dagilim_ve_scatter.png'))
            plt.close()

        print(f"\nTüm analizler tamamlandı! Grafikler '{output_dir}' klasörüne kaydedildi.")