import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy import stats

# 1. Kullanıcıdan veri klasörünün yolunu al
hedef_klasor = input("Lütfen CSV dosyalarının bulunduğu klasörün yolunu girin: ").strip('"').strip("'")

if not os.path.exists(hedef_klasor):
    print("Klasör bulunamadı! Lütfen yolu doğru girdiğinizden emin olun.")
else:
    # 2. Statik Veriyi (Temizlenmis_Kamp_Verileri.csv) Yükle
    statik_veri_yolu = os.path.join(hedef_klasor, "Temizlenmis_Kamp_Verileri.csv")
    
    if not os.path.exists(statik_veri_yolu):
        print(f"Hata: 'Temizlenmis_Kamp_Verileri.csv' dosyası bulunamadı!")
    else:
        df_statik = pd.read_csv(statik_veri_yolu)
        
        # Makine öğrenmesi ve testler için yol türünü basitleştirelim
        def yol_kategorize_et(yol_metni):
            yol_metni = str(yol_metni).lower()
            if "tamamı asfalt" in yol_metni or "asfalt" in yol_metni.split('+')[0].strip():
                return "Asfalt Ağırlıklı"
            else:
                return "Toprak/Karma"
                
        df_statik['Yol_Turu_Basit'] = df_statik['Yol Türü & Yüzey Durumu'].apply(yol_kategorize_et)

        dosya_yollari = glob.glob(os.path.join(hedef_klasor, "*_haftalık_merge.csv"))
        master_df_list = []

        print("\nVeriler otomatik eşleştirilip birleştiriliyor...")
        for dosya in dosya_yollari:
            kamp_adi_ham = os.path.basename(dosya).replace("_haftalık_merge.csv", "")
            kamp_prefix = kamp_adi_ham.replace("##", "")
            
            try:
                df = pd.read_csv(dosya, sep=None, engine='python')
                df.columns = df.columns.str.strip()
                
                sutun_isimleri = list(df.columns)
                sutun_isimleri[0] = 'tarih' 
                df.columns = sutun_isimleri
                
                df['tarih_baslangic'] = df['tarih'].apply(lambda x: str(x).split(' - ')[0])

                rename_dict = {}
                for col in df.columns:
                    if '_temp' in col.lower(): rename_dict[col] = 'Sicaklik'
                    elif '_prcp' in col.lower(): rename_dict[col] = 'Yagis_Top'
                    elif '_snow' in col.lower(): rename_dict[col] = 'Kar'
                    elif '_rain' in col.lower(): rename_dict[col] = 'Yagmur'
                    elif '_wspd' in col.lower(): rename_dict[col] = 'Ruzgar'
                    elif '_rhum' in col.lower(): rename_dict[col] = 'Nem'
                
                df.rename(columns=rename_dict, inplace=True)
                df['Kamp_Alani'] = kamp_prefix
                
                eslesen_satir = df_statik[df_statik['Lokasyon Adı'].apply(lambda x: str(x) in kamp_prefix or kamp_prefix in str(x))]
                
                if not eslesen_satir.empty:
                    df['Rakim'] = eslesen_satir['Rakım (m)'].values[0]
                    df['Eğim_Yuzde'] = eslesen_satir['Ort. Eğim (%)'].values[0]
                    df['Hastane_Mesafe_km'] = eslesen_satir['Has. Mesafe (km)'].values[0]
                    df['Hastane_Varis_dk'] = eslesen_satir['Has. Varış Süresi (Dk)'].values[0]
                    df['Yol_Turu_Basit'] = eslesen_satir['Yol_Turu_Basit'].values[0]
                    df['Bolge'] = eslesen_satir['Bölge'].values[0]
                else:
                    df['Rakim'] = None
                    df['Eğim_Yuzde'] = None
                    df['Hastane_Mesafe_km'] = None
                    df['Hastane_Varis_dk'] = None
                    df['Yol_Turu_Basit'] = None
                    df['Bolge'] = None
                    
                master_df_list.append(df)
                print(f" [+] {kamp_prefix} başarıyla işlendi.")

            except Exception as e:
                print(f" [!!!] HATA: '{kamp_prefix}' dosyası bozuk veya okunamıyor. Atlanıyor! (Sebep: {e})")
                continue

        if len(master_df_list) == 0:
            print("\nHiçbir dosya okunamadı! Lütfen dosyalarınızın içeriğini kontrol edin.")
        else:
            master_df = pd.concat(master_df_list, ignore_index=True)
            
            # --- KRİTİK VERİ TİPİ DÜZELTMESİ (HATA BURADAN ÇIKIYORDU) ---
            # Sayısal hesaplama yapılacak sütunları zorla sayı formatına (float) dönüştürüyoruz.
            master_df['gercek_ziyaretci'] = pd.to_numeric(master_df['gercek_ziyaretci'], errors='coerce')
            master_df['Hastane_Varis_dk'] = pd.to_numeric(master_df['Hastane_Varis_dk'], errors='coerce')
            # -------------------------------------------------------------

            output_dir = os.path.join(hedef_klasor, "Makro_Analiz_Sonuclari")
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n{len(master_df_list)} kamp alanı başarıyla birleştirildi. Görselleştirmeler hazırlanıyor...")
            
            plt.figure(figsize=(14, 8))
            kamp_toplam = master_df.groupby('Kamp_Alani')['gercek_ziyaretci'].sum().sort_values(ascending=False)
            # Seaborn uyarısı giderildi (hue eklendi, legend kapatıldı)
            sns.barplot(x=kamp_toplam.values, y=kamp_toplam.index, hue=kamp_toplam.index, legend=False, palette="viridis")
            plt.title('Karadeniz Bölgesi Kamp Alanları Toplam Ziyaretçi Sayıları')
            plt.xlabel('Toplam Ziyaretçi (Son 2 Yıl)')
            plt.ylabel('Kamp Alanı')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '1_Kamp_Populerlik_Kiyaslama.png'))
            plt.close()

            yol_df = master_df.dropna(subset=['Yol_Turu_Basit', 'gercek_ziyaretci'])
            if not yol_df.empty:
                plt.figure(figsize=(8, 6))
                # Seaborn uyarısı giderildi
                sns.boxplot(x='Yol_Turu_Basit', y='gercek_ziyaretci', data=yol_df, hue='Yol_Turu_Basit', legend=False, palette="Set2")
                plt.title('Yol Türüne Göre Haftalık Ziyaretçi Dağılımı')
                plt.ylabel('Haftalık Ziyaretçi')
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, '2_Yol_Turu_Etkisi.png'))
                plt.close()

            uzaklik_df = master_df.groupby('Kamp_Alani').agg({
                'Hastane_Varis_dk': 'mean',
                'gercek_ziyaretci': 'mean',
                'Bolge': 'first'
            }).dropna()

            if not uzaklik_df.empty:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='Hastane_Varis_dk', y='gercek_ziyaretci', hue='Bolge', s=100, data=uzaklik_df, palette="deep")
                plt.title('Hastaneye Varış Süresi vs Ortalama Ziyaretçi')
                plt.xlabel('Hastaneye Varış Süresi (Dakika)')
                plt.ylabel('Haftalık Ortalama Ziyaretçi')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, '3_Hastane_Uzaklik_Etkisi.png'))
                plt.close()

            with open(os.path.join(output_dir, "Hipotez_Test_Sonuclari.txt"), "w", encoding="utf-8") as f:
                f.write("DSA210 PROJESI - EDA VE HIPOTEZ TEST BULGULARI\n")
                f.write("="*50 + "\n\n")
                
                f.write("HİPOTEZ 1: Asfalt yola sahip kamp alanları, toprak/karma yola sahip olanlardan istatistiksel olarak farklı sayıda ziyaretçi alır.\n")
                if not yol_df.empty:
                    asfalt = yol_df[yol_df['Yol_Turu_Basit'] == 'Asfalt Ağırlıklı']['gercek_ziyaretci'].dropna()
                    toprak = yol_df[yol_df['Yol_Turu_Basit'] == 'Toprak/Karma']['gercek_ziyaretci'].dropna()
                    
                    if len(asfalt) > 0 and len(toprak) > 0:
                        t_stat, p_val = stats.ttest_ind(asfalt, toprak, equal_var=False)
                        f.write(f"T-İstatistiği: {t_stat:.4f}, P-Değeri: {p_val:.4f}\n")
                        if p_val < 0.05:
                            f.write("Sonuç: H0 reddedildi. Yol türü ziyaretçi sayısında istatistiksel olarak ANLAMLI bir fark yaratmaktadır.\n\n")
                        else:
                            f.write("Sonuç: H0 reddedilemedi. Yol türü istatistiksel olarak anlamlı bir fark YARATMAMAKTADIR.\n\n")

                f.write("HİPOTEZ 2: Kamp alanının hastaneye uzaklığı (dk) ile ziyaretçi sayısı arasında negatif bir korelasyon vardır.\n")
                if not uzaklik_df.empty:
                    corr_katsayisi, p_val2 = stats.pearsonr(uzaklik_df['Hastane_Varis_dk'], uzaklik_df['gercek_ziyaretci'])
                    f.write(f"Pearson Korelasyon Katsayısı: {corr_katsayisi:.4f}, P-Değeri: {p_val2:.4f}\n")
                    if p_val2 < 0.05:
                        f.write("Sonuç: Uzaklık ve ziyaretçi sayısı arasında istatistiksel olarak ANLAMLI bir ilişki vardır.\n")
                    else:
                        f.write("Sonuç: Uzaklık ve ziyaretçi sayısı arasında istatistiksel olarak anlamlı bir ilişki YOKTUR.\n")

            master_df.to_csv(os.path.join(output_dir, "Master_Kamp_Verisi_Temiz.csv"), index=False)
            print(f"\nİşlem Başarılı! Bütün makro analizler tamamlandı.")