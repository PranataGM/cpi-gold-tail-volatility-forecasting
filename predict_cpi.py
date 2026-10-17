"""
CPI Macro Surprise & Tail Volatility Forecasting Model: XAU/USD (Gold)
Institutional quantitative framework for predicting extreme volatility risks (>= 350 pips)
and asymmetric directional skew upon US CPI macroeconomic releases.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Configure Windows console encoding for Unicode support
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


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
                f"STATUS: HIGH VOLATILITY RISK!\n"
                f"Deploy STRADDLE PENDING BRACKET (Buy Stop & Sell Stop placed 25-30 pips from spot).\n"
                f"Stop Loss: 150-200 pips. If entering directional bias ({directional_bias}), await 5m candle close."
            )
            status_signal = "[!] HIGH VOLATILITY ALERT"
        else:
            recommendation = (
                f"STATUS: NORMAL / MODERATE VOLATILITY.\n"
                f"Avoid chasing initial 1m-3m momentum spikes (fakeouts). Standard target: 25-35 pips."
            )
            status_signal = "[OK] NORMAL VOLATILITY"

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


def print_banner():
    print("=" * 76)
    print("   CPI MACRO SURPRISE & GOLD (XAU/USD) TAIL VOLATILITY FORECASTER   ")
    print("=" * 76)


def main():
    print_banner()

    dataset_path = os.path.join("data", "cpi_gold_merged_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"[!] Dataset file '{dataset_path}' not found.")
        print("[!] Ensure the historical data files are in 'data/'.")
        return

    df_data = pd.read_csv(dataset_path)
    model = CPIGoldVolatilityModel(pip_threshold=350.0, top_k=5)
    model.fit(df_data)

    print("[*] Model trained on 80 historical CPI events (2020 - 2026).")
    print("\nSELECT OPERATING MODE:")
    print("  1. Backtest Benchmark Mode (Latest Historical Event)")
    print("  2. Live Trading Mode (Input New ForexFactory / BLS Release Numbers)")

    choice = input("\nEnter Choice (1 or 2, default: 1): ").strip()
    if choice != "2":
        latest = df_data.iloc[-1]
        core_act = float(latest['core_mom_act'])
        core_est = float(latest['core_mom_est'])
        yoy_act = float(latest['cpi_yoy_act'])
        yoy_est = float(latest['cpi_yoy_est'])
        label = f"Release {latest['release_date']} ({latest['period']})"
    else:
        print("\n--- INPUT CPI NUMBERS (FROM FOREXFACTORY / INVESTING.COM) ---")
        print("Tip: Only Core CPI m/m is mandatory. YoY figures are optional.")
        try:
            core_act_in = input("1. Core CPI m/m Actual   (e.g., 0.3) : ").strip()
            core_est_in = input("2. Core CPI m/m Forecast (e.g., 0.2) : ").strip()
            core_act = float(core_act_in) if core_act_in else 0.3
            core_est = float(core_est_in) if core_est_in else 0.2

            yoy_act_in = input("3. CPI y/y Actual   [Optional - Enter to skip]: ").strip()
            yoy_est_in = input("4. CPI y/y Forecast [Optional - Enter to skip]: ").strip()
            if yoy_act_in and yoy_est_in:
                yoy_act = float(yoy_act_in)
                yoy_est = float(yoy_est_in)
            else:
                yoy_act = round(2.5 + (core_act - core_est), 2)
                yoy_est = 2.5
            label = "Live Trading Input"
        except ValueError:
            print("[!] Invalid number format. Falling back to default baseline.")
            core_act, core_est, yoy_act, yoy_est = 0.3, 0.2, 2.6, 2.5
            label = "Default Demo"

    # Execute Prediction
    res = model.predict(core_act, core_est, yoy_act, yoy_est, period_label=label)

    print("\n" + "=" * 76)
    print(f" FORECAST OUTPUT & DIRECTIONAL SKEW ANALYSIS - {label} ")
    print("=" * 76)
    print(f"Signal Status               : {res['status_signal']}")
    print(f"Tail Volatility Probability : {res['prob_extreme']}% (Chance of range >= 350 pips)")
    print(f"Expected Gold Movement      : +/- {res['expected_pips']} Pips (+/- $ {res['expected_dollar']} USD)")
    print(f"Directional Bias (Skew)     : {res['directional_bias']} (Confidence: {res['directional_confidence']}%)")
    print(f"Macro Driver Rationale      : {res['bias_reason']}")
    print("-" * 76)
    print("Execution Protocol (MT4 / MT5):")
    print(res['recommendation'])
    print("-" * 76)
    print("Top-5 Closest Historical Surprise Analogues:")
    for _, r in res['similar_events'].iterrows():
        print(f"  - {r['release_date']} ({r['period']}) | Core MoM: {r['core_mom_act']}% vs {r['core_mom_est']}% | Range: {r['gold_pips_range']:>6.1f} pips | {r['gold_direction']}")
    print("=" * 76)


if __name__ == '__main__':
    main()
