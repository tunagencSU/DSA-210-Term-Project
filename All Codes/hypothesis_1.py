import pandas as pd
import os
import glob
from scipy.stats import pearsonr, ttest_ind

# Hipotez testi 1

# ==========================================
# KARADENİZ KAMP ALANLARI HİPOTEZ TESTİ BETİĞİ
# ==========================================

# 1. Kullanıcıdan Windows dosya yolunu al ve temizle
hedef_klasor = input("Lütfen '_haftalık_merge.csv' dosyalarının bulunduğu klasörün yolunu girin (Örn: C:\\Users\\Administrator\\Desktop\\DSA): ")
hedef_klasor = hedef_klasor.strip('"').strip("'")

if not os.path.exists(hedef_klasor):
    print("HATA: Belirtilen klasör bulunamadı. Lütfen dosya yolunu kontrol edin.")
else:
    # Hedef klasördeki tüm haftalık merge dosyalarını bul
    arama_deseni = os.path.join(hedef_klasor, "*_haftalık_merge.csv")
    dosya_yollari = glob.glob(arama_deseni)

    if len(dosya_yollari) == 0:
        print("Belirtilen klasörde '_haftalık_merge.csv' uzantılı dosya bulunamadı.")
    else:
        print(f"Toplam {len(dosya_yollari)} kamp alanı bulundu. Hipotez testleri uygulanıyor...\n")
        
        sonuclar = []

        for dosya in dosya_yollari:
            # Dosya isminden kamp adını temizle
            kamp_adi = os.path.basename(dosya).replace("_haftalık_merge.csv", "").replace("##", "")
            
            df = pd.read_csv(dosya)
            
            # Dinamik sütun tespiti (Önceki isimlendirme hatalarını by-pass etmek için)
            temp_col = next((col for col in df.columns if col.endswith('_temp')), None)
            prcp_col = next((col for col in df.columns if col.endswith('_prcp')), None)
            ziyaretci_col = 'gercek_ziyaretci'
            
            if ziyaretci_col not in df.columns:
                continue
                
            kamp_sonucu = {'Kamp Alanı': kamp_adi}
            
            # -----------------------------------------------------------
            # TEST 1: Sıcaklık vs Ziyaretçi Sayısı (Pearson Korelasyonu)
            # H0: Sıcaklık ile ziyaretçi sayısı arasında anlamlı bir ilişki yoktur.
            # -----------------------------------------------------------
            if temp_col:
                valid_temp = df[[temp_col, ziyaretci_col]].dropna()
                if len(valid_temp) > 2:
                    corr, p_val_temp = pearsonr(valid_temp[temp_col], valid_temp[ziyaretci_col])
                    anlamli_mi_temp = "Evet" if p_val_temp < 0.05 else "Hayır"
                    kamp_sonucu['Sıcaklık (P-Value)'] = f"{p_val_temp:.4f}"
                    kamp_sonucu['Sıc. Anlamlı?'] = anlamli_mi_temp
                else:
                    kamp_sonucu['Sıcaklık (P-Value)'] = "Veri Yetersiz"
                    kamp_sonucu['Sıc. Anlamlı?'] = "-"
            else:
                kamp_sonucu['Sıcaklık (P-Value)'] = "Sütun Yok"
                kamp_sonucu['Sıc. Anlamlı?'] = "-"
                
            # -----------------------------------------------------------
            # TEST 2: Yağış vs Ziyaretçi Sayısı (Bağımsız Çift Örneklem T-Testi)
            # H0: Yağışlı haftalar ile yağışsız haftaların ziyaretçi ortalamaları eşittir.
            # -----------------------------------------------------------
            if prcp_col:
                valid_prcp = df[[prcp_col, ziyaretci_col]].dropna()
                yagisli = valid_prcp[valid_prcp[prcp_col] > 1.0][ziyaretci_col]
                yagissiz = valid_prcp[valid_prcp[prcp_col] <= 1.0][ziyaretci_col]
                
                # T-Testi için her iki grupta da en az 2 veri noktası olmalı
                if len(yagisli) > 1 and len(yagissiz) > 1:
                    # equal_var=False kullanarak Welch's t-test uyguluyoruz (daha güvenilirdir)
                    t_stat, p_val_prcp = ttest_ind(yagissiz, yagisli, equal_var=False)
                    anlamli_mi_prcp = "Evet" if p_val_prcp < 0.05 else "Hayır"
                    kamp_sonucu['Yağış (P-Value)'] = f"{p_val_prcp:.4f}"
                    kamp_sonucu['Yağış Anlamlı?'] = anlamli_mi_prcp
                else:
                    kamp_sonucu['Yağış (P-Value)'] = "Örneklem Yetersiz"
                    kamp_sonucu['Yağış Anlamlı?'] = "-"
            else:
                kamp_sonucu['Yağış (P-Value)'] = "Sütun Yok"
                kamp_sonucu['Yağış Anlamlı?'] = "-"
                
            sonuclar.append(kamp_sonucu)

        # -----------------------------------------------------------
        # SONUÇLARI PANDAS DATAFRAME İLE TABLOYA DÖNÜŞTÜR VE KAYDET
        # -----------------------------------------------------------
        df_ozet = pd.DataFrame(sonuclar)
        
        print("-" * 80)
        print("HİPOTEZ TESTİ ÖZET TABLOSU")
        print("-" * 80)
        print(df_ozet.to_string(index=False))
        print("-" * 80)
        
        # Analizleri GitHub repona kolayca ekleyebilmen için CSV olarak kaydet
        cikti_yolu = os.path.join(hedef_klasor, "Hipotez_Testleri_Ozet_Tablosu.csv")
        df_ozet.to_csv(cikti_yolu, index=False, encoding='utf-8-sig')
        print(f"\n✅ Tablo başarıyla CSV olarak kaydedildi: '{cikti_yolu}'")
