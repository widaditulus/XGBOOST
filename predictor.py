# predictor.py (VERSI REFACTOR TOTAL - FUNGSI LENGKAP & STABIL)
# -*- coding: utf-8 -*-
import os
import shutil
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from sklearn.preprocessing import LabelEncoder
from constants import (MODELS_DIR, MARKET_CONFIGS, DRIFT_THRESHOLD,
                     ACCURACY_THRESHOLD_FOR_RETRAIN, ADAPTIVE_LEARNING_CONFIG, ENSEMBLE_CONFIG, CONTINUAL_LEARNING_CONFIG)
from utils import logger, error_handler, drift_logger
from model_config import TRAINING_CONFIG_OPTIONS
from exceptions import TrainingError, PredictionError, DataFetchingError
from data_fetcher import DataFetcher
from evaluation import calculate_brier_score, calculate_ece
from ensemble_helper import train_ensemble_models, ensemble_predict_proba
from continual_learner import ContinualLearner

class DataManager:
    """Mengelola pengambilan dan validasi data."""
    def __init__(self, pasaran):
        self.pasaran = pasaran; self.df = None; self.fetcher = DataFetcher(pasaran)
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: raise DataFetchingError("Dataframe kosong.")
        df = df.copy(); df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if c in ['date', 'tanggal']), None)
        result_col = next((c for c in df.columns if c in ['result', 'nomor']), None)
        if not date_col or not result_col: raise DataFetchingError(f"Kolom wajib tidak ditemukan.")
        df.rename(columns={date_col: 'date', result_col: 'result'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date', 'result'], inplace=True)
        df.drop_duplicates(subset=['date'], keep='first', inplace=True)
        df['result'] = df['result'].astype(str).str.strip().str.split('.').str[0].str.zfill(4)
        df = df[df['result'].str.match(r'^\d{4}$')].copy()
        if df.empty: raise DataFetchingError("Tidak ada data valid setelah dibersihkan.")
        return df.sort_values("date").reset_index(drop=True)
    def get_data(self, force_refresh: bool = False, force_github: bool = False) -> pd.DataFrame:
        if self.df is None or force_refresh:
            self.df = self._validate_data(self.fetcher.fetch_data(force_github=force_github))
        return self.df.copy()

class FeatureProcessor:
    """Logika rekayasa fitur yang diperkuat dan 100% konsisten."""
    def __init__(self, timesteps, feature_config):
        self.timesteps = timesteps; self.config = feature_config; self.digits = ["as", "kop", "kepala", "ekor"]
    
    def create_features(self, df_input: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = df_input.copy()
        for i, digit in enumerate(self.digits):
            valid_rows = df['result'].notna()
            df.loc[valid_rows, digit] = pd.to_numeric(df.loc[valid_rows, "result"].str[i], errors='coerce')

        # --- MEMPERTAHANKAN SEMUA FITUR CANGGIH DENGAN CARA YANG AMAN ---
        df['dayofweek'] = df['date'].dt.dayofweek
        if self.config.get("add_cyclical_features"):
            df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 6.0)
            df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 6.0)

        # Fitur turunan & rolling HANYA dari data masa lalu (H-1) untuk mencegah kebocoran
        s_as, s_kop, s_kep, s_eko = (df['as'].shift(1), df['kop'].shift(1), df['kepala'].shift(1), df['ekor'].shift(1))
        df['prev_sum_all'] = s_as + s_kop + s_kep + s_eko
        if self.config.get("add_interaction_features"):
            for d_str, series in zip(self.digits, [s_as, s_kop, s_kep, s_eko]):
                window = self.config.get("volatility_window", 10)
                df[f'prev_{d_str}_rolling_mean_{window}'] = series.rolling(window=window).mean()
                df[f'prev_{d_str}_rolling_std_{window}'] = series.rolling(window=window).std()

        # Fitur lag (selalu aman)
        for d in self.digits:
            for i in range(1, self.timesteps + 1):
                df[f'{d}_lag_{i}'] = df[d].shift(i)
        
        feature_names = [c for c in df.columns if c not in ['date', 'result'] + self.digits]
        df[feature_names] = df[feature_names].fillna(0).astype('float32')
        return df, feature_names

class ModelPredictor:
    def __init__(self, pasaran: str):
        self.pasaran = pasaran.lower()
        self.config = self._get_config()
        self.model_dir_base = os.path.join(MODELS_DIR, self.pasaran)
        self.digits = ["as", "kop", "kepala", "ekor"]
        self.data_manager = DataManager(pasaran)
        self.feature_processor = FeatureProcessor(self.config["strategy"]["timesteps"], self.config["feature_engineering"])
        self.models: Dict[str, Any] = {d: None for d in self.digits}; self.rf_models: Dict[str, Any] = {d: None for d in self.digits}; self.lgbm_models: Dict[str, Any] = {d: None for d in self.digits}
        self.label_encoders: Dict[str, LabelEncoder] = {d: LabelEncoder() for d in self.digits}
        self.feature_names: List[str] = []
        self.continual_learner = ContinualLearner(self) if CONTINUAL_LEARNING_CONFIG.get("ENABLED") else None
        self.models_ready = self.load_models()

    def _get_config(self) -> Dict[str, Any]:
        default = MARKET_CONFIGS["default"].copy()
        market_specific = MARKET_CONFIGS.get(self.pasaran, {})
        # Deep merge
        default['strategy'].update(market_specific.get('strategy', {}))
        default['feature_engineering'].update(market_specific.get('feature_engineering', {}))
        return default

    def load_models(self) -> bool:
        if not os.path.exists(self.model_dir_base): return False
        try:
            self.feature_names = joblib.load(os.path.join(self.model_dir_base, "features.pkl"))
            for d in self.digits:
                self.models[d] = joblib.load(os.path.join(self.model_dir_base, f"model_{d}.pkl"))
                self.label_encoders[d] = joblib.load(os.path.join(self.model_dir_base, f"encoder_{d}.pkl"))
                if ENSEMBLE_CONFIG.get("USE_ENSEMBLE"):
                    if os.path.exists(p := os.path.join(self.model_dir_base, f"rf_model_{d}.pkl")): self.rf_models[d] = joblib.load(p)
                    if os.path.exists(p := os.path.join(self.model_dir_base, f"lgbm_model_{d}.pkl")): self.lgbm_models[d] = joblib.load(p)
            logger.info(f"Model untuk {self.pasaran} berhasil dimuat."); return True
        except Exception: return False

    @error_handler(logger)
    def train_model(self, training_mode: str, use_recency_bias: bool, custom_data: Optional[pd.DataFrame] = None):
        os.makedirs(self.model_dir_base, exist_ok=True)
        params = TRAINING_CONFIG_OPTIONS[training_mode]['xgb_params']
        df_full = custom_data if custom_data is not None else self.data_manager.get_data(force_refresh=True, force_github=True)
        
        # Alur tunggal untuk rekayasa fitur
        processed_df, features = self.feature_processor.create_features(df_full)
        self.feature_names = features
        joblib.dump(self.feature_names, os.path.join(self.model_dir_base, "features.pkl"))

        train_df = processed_df.dropna(subset=self.feature_names + self.digits).copy()
        min_samples = self.config["strategy"]["min_training_samples"]
        if len(train_df) < min_samples: raise TrainingError(f"Data tidak cukup: Perlu {min_samples}, tersedia {len(train_df)}.")

        for d in self.digits:
            X = train_df[self.feature_names]; y = train_df[d]
            y_encoded = self.label_encoders[d].fit_transform(y)
            model = xgb.XGBClassifier(**params[d]); model.fit(X, y_encoded)
            joblib.dump(model, os.path.join(self.model_dir_base, f"model_{d}.pkl"))
            joblib.dump(self.label_encoders[d], os.path.join(self.model_dir_base, f"encoder_{d}.pkl"))
            
            # Simpan feature importance untuk deteksi drift
            imp_df = pd.DataFrame(model.get_booster().get_score(importance_type='weight').items(), columns=['feature', 'weight']).sort_values('weight', ascending=False)
            imp_df.to_csv(os.path.join(self.model_dir_base, f"feature_importance_{d}.csv"), index=False)
            
            if ENSEMBLE_CONFIG.get("USE_ENSEMBLE"): train_ensemble_models(X, y_encoded, self.model_dir_base, d)

        self.models_ready = self.load_models()
        if not self.models_ready: raise TrainingError("Training selesai tapi model gagal dimuat ulang.")

    @error_handler(logger)
    def predict_next_day(self, target_date_str: str, for_evaluation: bool = False) -> Dict[str, Any]:
        if not self.models_ready: raise PredictionError("Model belum siap. Lakukan training.")
        target_date = pd.to_datetime(target_date_str)
        
        # Siapkan data: historis + 1 baris masa depan
        df_hist = self.data_manager.get_data()
        start_date = target_date - timedelta(days=self.config["strategy"]["timesteps"] + 50) # Buffer
        df_window = df_hist[df_hist['date'] >= start_date].copy()
        future_row = pd.DataFrame([{'date': target_date, 'result': np.nan}])
        df_process = pd.concat([df_window, future_row], ignore_index=True)
        
        # Proses dengan alur yang SAMA PERSIS seperti training
        processed_df, _ = self.feature_processor.create_features(df_process)
        feature_vector = processed_df.iloc[-1:][self.feature_names]
        if feature_vector.empty or feature_vector.isnull().values.any(): raise PredictionError("Gagal membuat feature vector.")

        # Lakukan prediksi dan kembalikan semua fungsi
        predictions, all_candidates = {}, []
        result = {"probabilities": {}, "label_encoders": {}} if for_evaluation else {}
        for d in self.digits:
            models = {'xgb': self.models[d], 'rf': self.rf_models.get(d), 'lgbm': self.lgbm_models.get(d)}
            probas = ensemble_predict_proba({k: v for k, v in models.items() if v}, feature_vector)[0]
            if for_evaluation: result["probabilities"][d], result["label_encoders"][d] = probas, self.label_encoders[d]
            top_indices = np.argsort(probas)[-2:][::-1]
            top_digits = self.label_encoders[d].inverse_transform(top_indices)
            predictions[d] = [str(int(digit)) for digit in top_digits]
            all_candidates.extend([(str(int(d)), p) for d, p in zip(top_digits, probas[top_indices])])

        angka_main = ", ".join(sorted(list(set(c[0] for c in all_candidates)))[:4])
        colok_bebas = max(all_candidates, key=lambda item: item[1])[0] if all_candidates else ""
        
        result.update({
            "prediction_date": target_date.strftime("%Y-%m-%d"),
            "final_4d_prediction": f"{predictions['as'][0]}{predictions['kop'][0]}{predictions['kepala'][0]}{predictions['ekor'][0]}",
            "kandidat_as": ", ".join(predictions['as']), "kandidat_kop": ", ".join(predictions['kop']),
            "kandidat_kepala": ", ".join(predictions['kepala']), "kandidat_ekor": ", ".join(predictions['ekor']),
            "angka_main": angka_main, "colok_bebas": colok_bebas
        })
        return result

    @error_handler(logger)
    def evaluate_performance(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        df = self.data_manager.get_data(force_github=True)
        eval_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        if eval_df.empty: return {"summary": {"error": "Tidak ada data pada periode ini."}, "results": []}
        
        results_list = []
        for _, row in eval_df.iterrows():
            try:
                pred = self.predict_next_day(row['date'].strftime('%Y-%m-%d'), for_evaluation=False)
                if pred: results_list.append({ "date": row['date'].strftime('%Y-%m-%d'), "actual": row['result'], **pred })
            except Exception: continue
        if not results_list: return {"summary": {"error": "Gagal menghasilkan prediksi untuk dievaluasi."}, "results": []}
        
        summary_df = pd.DataFrame(results_list)
        def check_hit(p, a): return a in p.replace(' ','').split(',') if isinstance(p, str) else False
        summary = {
            "total_days_evaluated": len(summary_df),
            "as_accuracy": summary_df.apply(lambda r: check_hit(r['kandidat_as'], r['actual'][0]), axis=1).mean(),
            "kop_accuracy": summary_df.apply(lambda r: check_hit(r['kandidat_kop'], r['actual'][1]), axis=1).mean(),
            "kepala_accuracy": summary_df.apply(lambda r: check_hit(r['kandidat_kepala'], r['actual'][2]), axis=1).mean(),
            "ekor_accuracy": summary_df.apply(lambda r: check_hit(r['kandidat_ekor'], r['actual'][3]), axis=1).mean(),
            "am_accuracy": summary_df.apply(lambda r: any(d in r['actual'] for d in r['angka_main'].replace(' ','').split(',')), axis=1).mean(),
            "cb_accuracy": summary_df.apply(lambda r: r['colok_bebas'] in r['actual'], axis=1).mean(),
            "retraining_recommended": False, "retraining_reason": "N/A"
        }
        drift = any(self._check_for_drift(d) for d in self.digits)
        if summary["kepala_accuracy"] < ACCURACY_THRESHOLD_FOR_RETRAIN:
            summary.update({"retraining_recommended": True, "retraining_reason": f"Akurasi Kepala ({summary['kepala_accuracy']:.1%}) di bawah ambang batas."})
        if self.continual_learner: self.continual_learner.check_and_retrain(summary, drift)
        return {"summary": summary, "results": results_list}

    @error_handler(drift_logger)
    def _check_for_drift(self, digit: str) -> bool:
        new_path = os.path.join(self.model_dir_base, f"feature_importance_{digit}.csv")
        base_path = os.path.join(self.model_dir_base, f"feature_importance_{digit}_baseline.csv")
        if not os.path.exists(new_path): return False
        if not os.path.exists(base_path): shutil.copy(new_path, base_path); return False
        new_df = pd.read_csv(new_path).set_index('feature'); base_df = pd.read_csv(base_path).set_index('feature')
        all_feats = base_df.index.union(new_df.index)
        new_dist = new_df.reindex(all_feats, fill_value=0)['weight']; base_dist = base_df.reindex(all_feats, fill_value=0)['weight']
        new_dist /= new_dist.sum(); base_dist /= base_dist.sum()
        base_dist[base_dist == 0] = 1e-6; new_dist[new_dist == 0] = 1e-6
        psi = np.sum((new_dist - base_dist) * np.log(new_dist / base_dist))
        if psi > 0.25:
            drift_logger.warning(f"DRIFT DETECTED for {self.pasaran}-{digit}! PSI ({psi:.4f}) > 0.25. Updating baseline.")
            shutil.copy(new_path, base_path); return True
        return False
        
    def get_feature_importance_data(self) -> Dict[str, Any]:
        imp_data = {}
        for d in self.digits:
            path = os.path.join(self.model_dir_base, f"feature_importance_{d}.csv")
            if os.path.exists(path): imp_data[d] = pd.read_csv(path).head(10).to_dict('records')
            else: imp_data[d] = []
        return imp_data