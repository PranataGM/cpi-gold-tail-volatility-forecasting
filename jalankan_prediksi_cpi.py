import os
import sys
import pandas as pd
from cpi_gold_volatility_model import CPIGoldVolatilityModel

# Pastikan konsol Windows rapi dan tahan karakter Unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def print_banner():
    print("=" * 72)
    print("       MODEL KUANTITATIF PREDIKSI VOLATILITAS EMAS (CPI XAU/USD)        ")
    print("=" * 72)


def main():
    print_banner()

    dataset_path = os.path.join("data", "cpi_gold_merged_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"[!] File dataset {dataset_path} belum ditemukan.")
        print("[*] Menjalankan build_cpi_dataset.py terlebih dahulu...")
        import build_cpi_dataset
        print("[OK] Dataset berhasil dibangun.")

    df_data = pd.read_csv(dataset_path)
    model = CPIGoldVolatilityModel(pip_threshold=350.0, top_k=5)
    model.fit(df_data)

    print("[*] Model berhasil dilatih pada 80 event historis CPI (2020 - 2026).")
    print("\nPILIH MODE OPERASI:")
    print("  1. Uji Coba Otomatis pada Rilis CPI Terbaru (September 2026)")
    print("  2. Masukkan Angka Rilis CPI Baru secara Langsung (Hari H Trading)")

    choice = input("\nPilihan Anda (1 atau 2, default: 1): ").strip()
    if choice != "2":
        # Mode 1: Event Terakhir (September 2026)
        latest = df_data.iloc[-1]
        core_act = float(latest['core_mom_act'])
        core_est = float(latest['core_mom_est'])
        yoy_act = float(latest['cpi_yoy_act'])
        yoy_est = float(latest['cpi_yoy_est'])
        label = f"Rilis {latest['release_date']} ({latest['period']})"
    else:
        # Mode 2: Input Cepat saat Trading (Hanya butuh 2 angka utama dari ForexFactory)
        print("\n--- INPUT DATA RILIS CPI (DARI FOREXFACTORY / INVESTING) ---")
        print("Tip: Cukup masukkan 2 angka utama (Core CPI m/m). Angka lain bersifat opsional.")
        try:
            core_act_in = input("1. Core CPI m/m Actual   (contoh: 0.3) : ").strip()
            core_est_in = input("2. Core CPI m/m Forecast (contoh: 0.2) : ").strip()
            core_act = float(core_act_in) if core_act_in else 0.3
            core_est = float(core_est_in) if core_est_in else 0.2

            yoy_act_in = input("3. CPI y/y Actual   [Opsional - tekan Enter jika tidak ada]: ").strip()
            yoy_est_in = input("4. CPI y/y Forecast [Opsional - tekan Enter jika tidak ada]: ").strip()
            if yoy_act_in and yoy_est_in:
                yoy_act = float(yoy_act_in)
                yoy_est = float(yoy_est_in)
            else:
                # Default selaras dengan deviasi Core
                yoy_act = round(2.5 + (core_act - core_est), 2)
                yoy_est = 2.5
            label = "Input Live Trading"
        except ValueError:
            print("[!] Format angka salah. Menggunakan nilai default.")
            core_act, core_est, yoy_act, yoy_est = 0.3, 0.2, 2.6, 2.5
            label = "Default Demo"

    # Jalankan Prediksi
    res = model.predict(core_act, core_est, yoy_act, yoy_est, period_label=label)

    print("\n" + "=" * 72)
    print(f" HASIL ANALISIS VOLATILITAS & BIAS ARAH EMAS (XAU/USD) - {label} ")
    print("=" * 72)
    print(f"Status Sinyal Model         : {res['status_sinyal']}")
    print(f"Probabilitas Gerak Ekstrem  : {res['prob_extreme']}% (Peluang rentang >= 350 pips)")
    print(f"Estimasi Rentang Harga Emas : +/- {res['expected_pips']} Pips (+/- $ {res['expected_dollar']} USD)")
    print(f"Bias Arah Pergerakan (Skew) : {res['directional_bias']} (Keyakinan: {res['directional_confidence']}%)")
    print(f"Alasan Analisis Makro       : {res['bias_reason']}")
    print("-" * 72)
    print("Rekomendasi Eksekusi MT4/MT5:")
    print(res['rekomendasi'])
    print("-" * 72)
    print("Top-5 Rilis Historis dengan Vektor Kejutan Serupa:")
    for _, r in res['similar_events'].iterrows():
        print(f"  - {r['release_date']} ({r['period']}) | Core MoM: {r['core_mom_act']}% vs {r['core_mom_est']}% | Rentang: {r['gold_pips_range']:>6.1f} pips | {r['gold_direction']}")
    print("=" * 72)

    # Simpan ke riwayat log
    log_file = "riwayat_prediksi_cpi.csv"
    log_data = {
        'Waktu_Eksekusi': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Periode': label,
        'Core_MoM_Act': core_act,
        'Core_MoM_Est': core_est,
        'Headline_YoY_Act': yoy_act,
        'Headline_YoY_Est': yoy_est,
        'Prob_Ekstrem': res['prob_extreme'],
        'Estimasi_Pips': res['expected_pips'],
        'Bias_Arah': res['directional_bias'],
        'Status_Sinyal': res['status_sinyal']
    }
    df_log = pd.DataFrame([log_data])
    if os.path.exists(log_file):
        df_log.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_log.to_csv(log_file, mode='w', header=True, index=False)
    print(f"[*] Hasil eksekusi telah tersimpan ke: {log_file}")


if __name__ == '__main__':
    main()
