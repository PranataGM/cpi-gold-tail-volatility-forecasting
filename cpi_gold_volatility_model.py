import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


class CPIGoldVolatilityModel:
    """
    Model Kuantitatif Institusional untuk Memprediksi Volatilitas dan Arah Asimetris Emas (XAU/USD)
    berdasarkan Kejutan Makroekonomi Rilis CPI (Actual vs. Forecast).
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
        
        # Sinyal Divergensi: Core naik tapi Headline turun atau sebaliknya (Pemicu Whipsaw Ekstrem)
        df_feat['is_divergent'] = (
            (df_feat['core_mom_surprise'] * df_feat['cpi_yoy_surprise'] < 0)
        ).astype(int)

        return df_feat

    def fit(self, df):
        """Melatih model pada data historis CPI dan pergerakan emas."""
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
        Menghasilkan estimasi probabilitas volatilitas ekstrem, rentang pips yang diharapkan,
        bias arah, serta event historis paling mirip.
        """
        if not self.is_fitted:
            raise ValueError("Model harus dilatih terlebih dahulu dengan memanggil .fit()")

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

        # 1. Probabilitas Klasifikasi Volatilitas Ekstrem
        prob_extreme = float(self.clf.predict_proba(X_new_scaled)[0, 1])

        # 2. Pencarian Top-K Event Historis Paling Mirip (Nearest Neighbors Surprise Vector)
        X_train_scaled = self.scaler.transform(self.df_train[self.feature_cols])
        dists = euclidean_distances(X_new_scaled, X_train_scaled)[0]
        
        # Konversi jarak euclidean menjadi bobot similaritas (1 / (1 + dist))
        sims = 1.0 / (1.0 + dists)
        top_idx = np.argsort(sims)[::-1][:self.top_k]
        
        top_sims = sims[top_idx]
        weights = top_sims / np.sum(top_sims)
        
        matched_events = self.df_train.iloc[top_idx].copy()
        matched_events['similarity_score'] = np.round(top_sims, 4)
        
        # 3. Estimasi Rentang Pips Tertimbang
        expected_pips = float(np.sum(weights * matched_events['gold_pips_range'].values))
        expected_dollar = expected_pips / 10.0

        # 4. Asymmetric Directional Bias
        # Core CPI MoM adalah raja penentu arah:
        # Core > Est -> Inflasi panas -> Dolar Menguat -> Emas Jatuh (Bearish)
        # Core < Est -> Inflasi dingin -> Dolar Melemah -> Emas Melonjak (Bullish)
        if core_surp >= 0.1:
            directional_bias = "BEARISH (DUMP)"
            bias_reason = f"Core CPI MoM meleset panas (+{core_surp}%). Ekspektasi suku bunga The Fed naik agresif."
            dir_confidence = min(88.0, 70.0 + abs(core_surp) * 60)
        elif core_surp <= -0.1:
            directional_bias = "BULLISH (SURGE)"
            bias_reason = f"Core CPI MoM mendingin ({core_surp}%). Tekanan The Fed melonggar, DXY melemah tajam."
            dir_confidence = min(90.0, 72.0 + abs(core_surp) * 60)
        else:
            if input_feat['is_divergent'].iloc[0] == 1:
                directional_bias = "WHIPSAW / REVERSAL"
                bias_reason = "Core CPI netral namun ada divergensi dengan Headline. Waspadai ekor candle palsu!"
                dir_confidence = 55.0
            else:
                directional_bias = "NETRAL / SIDEWAYS"
                bias_reason = "Angka CPI selaras dengan konsensus pasar. Volatilitas cepat mereda."
                dir_confidence = 50.0

        # 5. Rekomendasi Eksekusi Trading
        if prob_extreme >= 0.55 or expected_pips >= 400:
            rekomendasi = (
                f"Sinyal BAHAYA: Volatilitas Tinggi!\n"
                f"Pasang STRADDLE PENDING ORDER (Buy Stop & Sell Stop berjarak 25-30 pips dari harga running).\n"
                f"Jika eksekusi searah bias ({directional_bias}), tunggu candle 5 menit pertama konfirmasi."
            )
            status_sinyal = "[!] BAHAYA: VOLATILITAS TINGGI"
        else:
            rekomendasi = (
                f"Sinyal AMAN: Volatilitas Normal/Sedang.\n"
                f"Hindari mengejar lonjakan sesaat (fakeout). Target profit wajar 25-35 pips."
            )
            status_sinyal = "[OK] AMAN: VOLATILITAS NORMAL"

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
            'status_sinyal': status_sinyal,
            'rekomendasi': rekomendasi,
            'similar_events': matched_events[[
                'release_date', 'period', 'core_mom_act', 'core_mom_est', 
                'cpi_yoy_act', 'cpi_yoy_est', 'gold_pips_range', 'gold_direction', 'similarity_score'
            ]]
        }
