"""
CPI Macro Surprise & Tail Volatility Forecasting Model: XAU/USD (Gold)
Institutional quantitative framework for predicting extreme volatility risks (>= 350 pips)
and asymmetric directional skew upon US CPI macroeconomic releases.
"""

import os
import sys
import re
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Configure stdout for UTF-8 support on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Enable ANSI escape code processing in Windows Terminal / CMD
os.system('')

# ANSI Color Palette
GOLD = '\033[38;5;220m'
YELLOW = '\033[1;33m'
GREEN = '\033[1;32m'
CYAN = '\033[1;36m'
BLUE = '\033[1;34m'
MAGENTA = '\033[1;35m'
RED = '\033[1;31m'
GRAY = '\033[90m'
WHITE = '\033[1;37m'
RESET = '\033[0m'
BOLD = '\033[1m'


def strip_ansi(s):
    return re.sub(r'\033\[[0-9;]*m', '', s)


def box_row(content, width=58):
    vis = len(strip_ansi(content))
    pad = max(0, width - vis)
    return f"{GREEN}│{RESET}{content}{' ' * pad}{GREEN}│{RESET}"


def print_gold_header(dataset_count=80):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ASCII Art Logo (XAU/USD Gold Ingot)
    logo = [
        r"          .______________________.          ",
        r"         /                      /|          ",
        r"        /      X A U / U S D   / |          ",
        r"       /      G O L D  999.9  /  |          ",
        r"      /______________________/   |          ",
        r"      |                      |   |          ",
        r"      |   .--------------.   |   |          ",
        r"      |   |  FINE  GOLD  |   |   |          ",
        r"      |   |   1000 g     |   |   /          ",
        r"      |   '--------------'   |  /           ",
        r"      |     NET LIQUIDITY    | /            ",
        r"      |______________________|/             ",
        r"             [ XAUUSD ]                     ",
        r"         Institutional Desk                 ",
        r"                                            ",
        r"                                            "
    ]

    row_cat   = f" {CYAN}CATALYST  {RESET}  US CPI INFLATION RELEASE (BLS / FOREXFACTORY)"
    row_mod   = f" {CYAN}MODEL     {RESET}  MACRO SURPRISE & TAIL VOLATILITY (>= 350 PIPS)"
    row_ds    = f" {CYAN}DATABASE  {RESET}  {dataset_count} HISTORICAL CPI EVENTS (2020 - 2026)"
    row_eng   = f" {CYAN}STATUS    {RESET}  ONLINE // RAPID PARSER READY"
    row_sc    = f" {CYAN}SHORTCUT  {RESET}  TYPE '3, 2' FOR CORE ACT 0.3%, EST 0.2%"
    row_time  = f" {CYAN}TIME      {RESET}  {now_str} WIB"

    right_panel = [
        f"{BOLD}{WHITE}Hey, Trader{RESET}",
        f"{GREEN}┌── CPI Macro Volatility Engine ───────────────────────────┐{RESET}",
        box_row(row_cat),
        box_row(row_mod),
        box_row(row_ds),
        f"{GREEN}├── Rapid Input Protocol ───────────────────────────────────┤{RESET}",
        box_row(row_eng),
        box_row(row_sc),
        box_row(row_time),
        f"{GREEN}└───────────────────────────────────────────────────────────┘{RESET}",
        f"{WHITE}Quantitative Forecasting Engine Online.{RESET}",
        f"{RED}●{RESET} {GREEN}●{RESET} {YELLOW}●{RESET} {BLUE}●{RESET} {MAGENTA}●{RESET} {CYAN}●{RESET} {WHITE}●{RESET}",
        ""
    ]

    print("\n")
    for l, r in zip(logo, right_panel):
        print(f"{YELLOW}{l}{RESET}  {r}")
    print("")


class CPIGoldVolatilityModel:
    """
    Quantitative framework predicting Gold (XAU/USD) volatility and directional skew
    driven by macroeconomic surprise vectors in US CPI releases (Actual vs. Forecast).
    """

    def __init__(self, pip_threshold=350.0, top_k=5):
        self.pip_threshold = pip_threshold
        self.top_k = top_k
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(class_weight='balanced', random_state=42)
        self.is_fitted = False
        self.df_train = None
        self.feature_cols = ['core_mom_surprise', 'cpi_yoy_surprise', 'abs_core_surprise', 'abs_yoy_surprise', 'is_divergent']

    def _engineer_features(self, df):
        df_feat = df.copy()
        if 'core_mom_surprise' not in df_feat.columns:
            df_feat['core_mom_surprise'] = df_feat['core_mom_act'] - df_feat['core_mom_est']
        if 'cpi_yoy_surprise' not in df_feat.columns:
            df_feat['cpi_yoy_surprise'] = df_feat['cpi_yoy_act'] - df_feat['cpi_yoy_est']

        df_feat['abs_core_surprise'] = np.abs(df_feat['core_mom_surprise'])
        df_feat['abs_yoy_surprise'] = np.abs(df_feat['cpi_yoy_surprise'])
        
        # Divergence Flag: Core MoM up but Headline YoY down or vice versa
        df_feat['is_divergent'] = (
            (df_feat['core_mom_surprise'] * df_feat['cpi_yoy_surprise'] < 0)
        ).astype(int)

        return df_feat

    def fit(self, df):
        """Fit model on historical CPI releases and Gold volatility movements."""
        df_feat = self._engineer_features(df)
        self.df_train = df_feat.copy()

        X = df_feat[self.feature_cols]
        y = (df_feat['gold_pips_range'] >= self.pip_threshold).astype(int)

        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, core_mom_act, core_mom_est, cpi_yoy_act, cpi_yoy_est, period_label=""):
        """
        Generate tail volatility probability, expected pip range, directional skew,
        and top-K historical analogue releases.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling .predict()")

        core_surp = round(core_mom_act - core_mom_est, 2)
        yoy_surp = round(cpi_yoy_act - cpi_yoy_est, 2)

        input_df = pd.DataFrame([{
            'core_mom_act': core_mom_act,
            'core_mom_est': core_mom_est,
            'cpi_yoy_act': cpi_yoy_act,
            'cpi_yoy_est': cpi_yoy_est,
            'core_mom_surprise': core_surp,
            'cpi_yoy_surprise': yoy_surp
        }])

        input_feat = self._engineer_features(input_df)
        X_new = input_feat[self.feature_cols]
        X_new_scaled = self.scaler.transform(X_new)

        # 1. Extreme Tail Volatility Probability (>= 350 pips)
        prob_extreme = float(self.clf.predict_proba(X_new_scaled)[0, 1])

        # 2. Nearest Neighbors Historical Surprise Analogue Search
        X_train_scaled = self.scaler.transform(self.df_train[self.feature_cols])
        dists = euclidean_distances(X_new_scaled, X_train_scaled)[0]
        
        sims = 1.0 / (1.0 + dists)
        top_idx = np.argsort(sims)[::-1][:self.top_k]
        
        top_sims = sims[top_idx]
        weights = top_sims / np.sum(top_sims)
        
        matched_events = self.df_train.iloc[top_idx].copy()
        matched_events['similarity_score'] = np.round(top_sims, 4)
        
        # 3. Weighted Expected Pip & Dollar Range
        expected_pips = float(np.sum(weights * matched_events['gold_pips_range'].values))
        expected_dollar = expected_pips / 10.0

        # 4. Asymmetric Directional Bias
        if core_surp >= 0.1:
            directional_bias = "BEARISH (DUMP)"
            bias_reason = f"Core CPI MoM surprised hot (+{core_surp}%). Fed rate-cut expectations priced out."
            dir_confidence = min(88.0, 70.0 + abs(core_surp) * 60)
        elif core_surp <= -0.1:
            directional_bias = "BULLISH (SURGE)"
            bias_reason = f"Core CPI MoM surprised cool ({core_surp}%). Fed easing narrative accelerates, DXY drops."
            dir_confidence = min(90.0, 72.0 + abs(core_surp) * 60)
        else:
            if input_feat['is_divergent'].iloc[0] == 1:
                directional_bias = "WHIPSAW / REVERSAL"
                bias_reason = "Core CPI inline but divergent with Headline. Watch for aggressive 2-way wicks."
                dir_confidence = 55.0
            else:
                directional_bias = "NEUTRAL / SIDEWAYS"
                bias_reason = "CPI prints aligned with consensus. Fast mean-reversion expected."
                dir_confidence = 50.0

        # 5. Operational Playbook Status
        if prob_extreme >= 0.55 or expected_pips >= 400:
            recommendation = (
                f"{RED}[!] HIGH VOLATILITY ALERT: TAIL RISK DETECTED{RESET}\n"
                f"    - Pasang STRADDLE PENDING BRACKET (Buy Stop & Sell Stop berjarak 25-30 pips dari spot).\n"
                f"    - Stop Loss: 150-200 pips. Jika eksekusi searah bias ({directional_bias}), tunggu candle 5 menit pertama konfirmasi."
            )
            status_signal = f"{RED}[!] BAHAYA: VOLATILITAS TINGGI (RENTANG >= 350 PIPS){RESET}"
        else:
            recommendation = (
                f"{GREEN}[OK] NORMAL VOLATILITY: INLINE MARKET REACTION{RESET}\n"
                f"    - Hindari mengejar lonjakan sesaat menit awal (fakeouts).\n"
                f"    - Tunggu penutupan candle 15m untuk konfirmasi, target profit wajar 25-35 pips."
            )
            status_signal = f"{GREEN}[OK] AMAN: VOLATILITAS NORMAL / SEDANG{RESET}"

        return {
            'period': period_label,
            'core_surprise': core_surp,
            'yoy_surprise': yoy_surp,
            'prob_extreme': round(prob_extreme * 100, 1),
            'expected_pips': round(expected_pips, 1),
            'expected_dollar': round(expected_dollar, 2),
            'directional_bias': directional_bias,
            'directional_confidence': round(dir_confidence, 1),
            'bias_reason': bias_reason,
            'status_signal': status_signal,
            'recommendation': recommendation,
            'similar_events': matched_events[[
                'release_date', 'period', 'core_mom_act', 'core_mom_est', 
                'cpi_yoy_act', 'cpi_yoy_est', 'gold_pips_range', 'gold_direction', 'similarity_score'
            ]]
        }


def parse_cpi_token(token):
    """
    Parses a single number token with resilient character stripping.
    If the user enters integers without decimal point (e.g., '3', '2', '-1'):
    they are automatically converted to tenths (0.3, 0.2, -0.1).
    If a decimal point is explicitly provided (e.g. '1.0', '0.3', '2.5'):
    it is parsed directly as float.
    """
    token = token.strip().replace('"', '').replace("'", '').replace('\ufeff', '').replace('\u200b', '').replace('%', '').strip()
    if not token:
        return None
    try:
        if '.' in token:
            return float(token)
        else:
            return float(token) / 10.0
    except ValueError:
        return None


def parse_cpi_input(raw_input):
    """
    Parses single-line user input separated by comma, space, or slash.
    Examples:
      '3, 2'           -> [0.3, 0.2]
      '3,2'            -> [0.3, 0.2]
      '3 2'            -> [0.3, 0.2]
      '4, 2, 2.6, 2.5' -> [0.4, 0.2, 2.6, 2.5]
      '1.0, 0.8'       -> [1.0, 0.8]
      ''               -> None (Benchmark Mode)
    """
    raw = raw_input.strip().replace('"', '').replace("'", '')
    if not raw:
        return None
    
    parts = [p.strip() for p in raw.replace(';', ',').replace('/', ',').split(',') if p.strip()]
    if len(parts) == 1:
        space_parts = parts[0].split()
        if len(space_parts) > 1:
            parts = space_parts

    values = [parse_cpi_token(p) for p in parts if p.strip()]
    values = [v for v in values if v is not None]
    return values if len(values) > 0 else None


def run_prediction_flow(model, df_data):
    print("=" * 82)
    print(f" {BOLD}{WHITE}RAPID CPI INPUT PROTOCOL{RESET}")
    print("=" * 82)
    print(f"  Masukkan angka rilis sekaligus dipisahkan tanda koma: {YELLOW}Actual, Forecast{RESET}")
    print(f"  {GRAY}* Ketik {WHITE}'3, 2'{GRAY}           -> Core Actual: 0.3% | Forecast: 0.2% (tanpa perlu ketik '0.')")
    print(f"  * Ketik {WHITE}'1.0, 0.8'{GRAY}       -> Core Actual: 1.0% | Forecast: 0.8% (jika data >= 1 gunakan titik)")
    print(f"  * Ketik {WHITE}'4, 2, 2.6, 2.5'{GRAY} -> Core MoM: 0.4% vs 0.2% | Headline YoY: 2.6% vs 2.5%")
    print(f"  * Tekan {WHITE}[ENTER]{GRAY} langsung -> Otomatis uji coba rilis historis terbaru (Benchmark Mode){RESET}")
    print("-" * 82)

    user_raw = input(f"{CYAN}>> Input CPI Data [Actual, Forecast]: {RESET}").strip()
    tokens = parse_cpi_input(user_raw)

    if tokens is None or len(tokens) == 0:
        # Mode Benchmark: Rilis Terakhir di Dataset
        latest = df_data.iloc[-1]
        core_act = float(latest['core_mom_act'])
        core_est = float(latest['core_mom_est'])
        yoy_act = float(latest['cpi_yoy_act'])
        yoy_est = float(latest['cpi_yoy_est'])
        label = f"Benchmark Mode: Rilis {latest['release_date']} ({latest['period']})"
    else:
        core_act = tokens[0] if tokens[0] is not None else 0.3
        core_est = tokens[1] if (len(tokens) > 1 and tokens[1] is not None) else 0.2
        if len(tokens) >= 4 and tokens[2] is not None and tokens[3] is not None:
            yoy_act = tokens[2]
            yoy_est = tokens[3]
        else:
            # Otomatis estimasi YoY selaras deviasi Core
            yoy_act = round(2.5 + (core_act - core_est), 2)
            yoy_est = 2.5
        label = f"Live Input: Core MoM {core_act:.1f}% (Act) vs {core_est:.1f}% (Est)"

    # Jalankan inferensi kuantitatif
    res = model.predict(core_act, core_est, yoy_act, yoy_est, period_label=label)

    print("\n" + "=" * 82)
    print(f"       {BOLD}{GOLD}HASIL ANALISIS VOLATILITAS & BIAS ARAH EMAS (XAU/USD){RESET}       ")
    print("=" * 82)
    print(f"  Skenario Rilis       : {WHITE}{label}{RESET}")
    print(f"  Surprise Delta (MoM) : {YELLOW}{res['core_surprise']:+.2f}%{RESET} (Actual - Forecast)")
    print("-" * 82)
    print(f"  Status Sinyal        : {res['status_signal']}")
    print(f"  Probabilitas Ekstrem : {BOLD}{res['prob_extreme']}%{RESET} (Peluang rentang gerak >= 350 pips)")
    print(f"  Estimasi Rentang Pips: {BOLD}+/- {res['expected_pips']} Pips{RESET} (+/- $ {res['expected_dollar']} USD)")
    print(f"  Bias Arah (Skew)     : {BOLD}{res['directional_bias']}{RESET} (Keyakinan: {res['directional_confidence']}%)")
    print(f"  Katalis Makro        : {res['bias_reason']}")
    print("-" * 82)
    print(f"{BOLD}Protokol Eksekusi MetaTrader (MT4 / MT5):{RESET}")
    print(res['recommendation'])
    print("-" * 82)
    print(f"{BOLD}Top-5 Rilis Historis dengan Karakter Serupa:{RESET}")
    for _, r in res['similar_events'].iterrows():
        dir_color = GREEN if "SURGE" in str(r['gold_direction']) else RED
        print(f"  - {r['release_date']} ({r['period']}) | Core: {r['core_mom_act']}% vs {r['core_mom_est']}% | Rentang: {r['gold_pips_range']:>6.1f} pips | {dir_color}{r['gold_direction']}{RESET}")
    print("=" * 82)


def main():
    dataset_path = os.path.join("data", "cpi_gold_merged_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"[!] Dataset file '{dataset_path}' not found.")
        print("[!] Ensure historical files are placed inside 'data/'.")
        return

    df_data = pd.read_csv(dataset_path)
    model = CPIGoldVolatilityModel(pip_threshold=350.0, top_k=5)
    model.fit(df_data)

    print_gold_header(dataset_count=len(df_data))
    run_prediction_flow(model, df_data)


if __name__ == '__main__':
    main()
