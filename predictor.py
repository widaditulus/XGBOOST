# predictor.py (Final - Dengan Strategi CB Hibrida dan Integrasi Tuning)

# -*- coding: utf-8 -*-
import os
import shutil
import joblib
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from constants import (MODELS_DIR, MARKET_CONFIGS, DRIFT_THRESHOLD,
                     ACCURACY_THRESHOLD_FOR_RETRAIN, ADAPTIVE_LEARNING_CONFIG,
                     ENSEMBLE_CONFIG, TRAINING_PENALTY_CONFIG, HYBRID_SCORING_CONFIG,
                     CB_STRATEGY_CONFIG)
from utils import logger, error_handler, drift_logger
from model_config import TRAINING_CONFIG_OPTIONS, XGB_PARAMS_CB
from exceptions import TrainingError, PredictionError, DataFetchingError
from data_fetcher import DataFetcher
from evaluation import calculate_brier_score, calculate_ece
from ensemble_helper import train_ensemble_models, ensemble_predict_proba
from continual_learner import ContinualLearner

class DataManager:
    def __init__(self, pasaran):
        self.pasaran = pasaran
        self.df = None
        self.lock = threading.RLock()
        self.fetcher = DataFetcher(pasaran)

    def _clean_and_pad_result(self, series: pd.Series) -> pd.Series:
        cleaned_series = series.astype(str).str.strip()
        cleaned_series = cleaned_series.str.split('.').str[0]
        cleaned_series = cleaned_series.str.zfill(4)
        return cleaned_series

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: raise DataFetchingError("DataFrame sumber kosong atau tidak ada.")
        required = ['date', 'result']
        if not all(col in df.columns for col in required): raise DataFetchingError(f"Kolom wajib '{', '.join(required)}' tidak ditemukan di data.")
        df = df.copy()
        if df['date'].duplicated().any():
            df.drop_duplicates(subset=['date'], keep='first', inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df['result'] = self._clean_and_pad_result(df['result'])
        valid_results = df['result'].str.match(r'^\d{4}$')
        if not valid_results.all():
            invalid_count = (~valid_results).sum()
            logger.warning(f"Ditemukan {invalid_count} format 'result' yang tidak valid (bukan 4-digit) SETELAH pembersihan. Baris ini akan dihapus.")
            df = df[valid_results]
        if df.empty: raise DataFetchingError("Tidak ada data valid yang tersisa setelah proses pembersihan.")
        return df

    @error_handler(logger)
    def get_data(self, force_refresh: bool = False, force_github: bool = False) -> pd.DataFrame:
        with self.lock:
            if self.df is None or force_refresh:
                df_raw = self.fetcher.fetch_data(force_github=force_github)
                df_validated = self._validate_data(df_raw)
                df_sorted = df_validated.sort_values("date").reset_index(drop=True)
                self.df = df_sorted
            return self.df.copy()

    # UPDATED: Metode baru untuk memeriksa kesegaran data antara lokal dan remote
    @error_handler(logger)
    def check_data_freshness(self) -> Dict[str, Any]:
        """Membandingkan tanggal data terbaru dari DB lokal dan GitHub."""
        local_df = self.get_data(force_refresh=False, force_github=False)
        if local_df is None or local_df.empty:
            return {"status": "stale", "message": "Data lokal tidak ditemukan."}

        local_latest_date = local_df['date'].max()

        try:
            remote_df = self.fetcher.fetch_data(force_github=True)
            if remote_df is None or remote_df.empty:
                return {"status": "error", "message": "Gagal mengambil data remote."}
            
            remote_latest_date = self._validate_data(remote_df)['date'].max()

            if local_latest_date < remote_latest_date:
                status = "stale"
            else:
                status = "latest"

            return {
                "status": status,
                "local_date": local_latest_date.strftime('%Y-%m-%d'),
                "remote_date": remote_latest_date.strftime('%Y-%m-%d')
            }
        except Exception as e:
            logger.error(f"Error saat memeriksa kesegaran data: {e}", exc_info=True)
            return {"status": "error", "message": "Terjadi kesalahan saat perbandingan data."}

class FeatureProcessor:
    def __init__(self, timesteps, feature_config):
        self.timesteps = timesteps
        self.feature_config = feature_config
        self.digits = ["as", "kop", "kepala", "ekor"]
        self.cb_target_cols = [f'cb_target_{i}' for i in range(10)]
        self.feature_names = []

    def _calculate_streak(self, series):
        change_markers = series.ne(series.shift())
        cumulative_group_id = change_markers.cumsum()
        return series.groupby(cumulative_group_id).cumcount() + 1

    def _detect_trend_changes(self, series, short_window=5, long_window=15):
        short_ma = series.rolling(window=short_window, min_periods=1).mean()
        long_ma = series.rolling(window=long_window, min_periods=1).mean()
        crossover = ((short_ma > long_ma) & (short_ma.shift(1) < long_ma.shift(1))) | ((short_ma < long_ma) & (short_ma.shift(1) > long_ma.shift(1)))
        return crossover.astype(int)

    def fit_transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()
        df['date'] = pd.to_datetime(df['date'])
        for i, digit in enumerate(self.digits):
            df[digit] = df["result"].str[i].astype('int8')
        
        for i in range(10):
            df[f'cb_target_{i}'] = df['result'].str.contains(str(i)).astype(int)
        
        new_features = {}
        digit_cols = df[self.digits]
        new_features['digit_sum_prev'] = digit_cols.sum(axis=1).shift(1)
        new_features['even_count_prev'] = (digit_cols % 2 == 0).sum(axis=1).shift(1)
        new_features['odd_count_prev'] = (digit_cols % 2 != 0).sum(axis=1).shift(1)
        new_features['low_count_prev'] = (digit_cols < 5).sum(axis=1).shift(1)
        new_features['high_count_prev'] = (digit_cols >= 5).sum(axis=1).shift(1)

        new_features['dayofweek'] = df['date'].dt.dayofweek
        new_features['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
        new_features['day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 6.0)
        new_features['day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 6.0)
        new_features['as_kop_sum_prev'] = (df['as'] + df['kop']).shift(1)
        new_features['kepala_ekor_diff_prev'] = (df['kepala'] - df['ekor']).shift(1)
        new_features['as_mul_kop_prev'] = (df['as'] * df['kop']).shift(1)
        new_features['kepala_mul_ekor_prev'] = (df['kepala'] * df['ekor']).shift(1)
        vol_window = self.feature_config.get("volatility_window", 10)
        adv_window = 30
        for d in self.digits:
            shifted_d = df[d].shift(1)
            new_features[f'{d}_skew_{vol_window}_prev'] = shifted_d.rolling(vol_window).skew()
            new_features[f'{d}_kurt_{vol_window}_prev'] = shifted_d.rolling(vol_window).kurt()
            rolling_mean = shifted_d.rolling(window=adv_window, min_periods=1).mean()
            rolling_std = shifted_d.rolling(window=adv_window, min_periods=1).std().replace(0, 1)
            new_features[f'{d}_zscore_prev'] = (shifted_d - rolling_mean) / rolling_std
            new_features[f'{d}_streak_prev'] = self._calculate_streak(shifted_d)
            new_features[f'{d}_trend_change_prev'] = self._detect_trend_changes(shifted_d)
        for d in self.digits:
            for i in range(1, self.timesteps + 1):
                new_features[f'{d}_lag_{i}'] = df[d].shift(i)
        df_features = pd.DataFrame(new_features, index=df.index)
        training_df = pd.concat([df, df_features], axis=1)
        
        self.feature_names = [col for col in training_df.columns if col not in ['date', 'result'] + self.digits + self.cb_target_cols]
        
        training_df.dropna(subset=self.feature_names + self.digits, inplace=True)
        training_df[self.feature_names] = training_df[self.feature_names].fillna(0)
        
        return training_df

    def transform_for_prediction(self, historical_df: pd.DataFrame, target_date: datetime) -> pd.DataFrame:
        if len(historical_df) < self.timesteps:
            raise PredictionError(f"Data historis tidak cukup ({len(historical_df)} baris) untuk membuat prediksi.")
        last_known_data = historical_df.iloc[-self.timesteps:].copy()
        for i, digit in enumerate(self.digits):
            last_known_data[digit] = last_known_data["result"].str[i].astype('int8')
        pred_vector = {}
        last_row = last_known_data.iloc[-1]

        last_digits = last_row[self.digits].values
        pred_vector['digit_sum_prev'] = last_digits.sum()
        pred_vector['even_count_prev'] = np.sum(last_digits % 2 == 0)
        pred_vector['odd_count_prev'] = np.sum(last_digits % 2 != 0)
        pred_vector['low_count_prev'] = np.sum(last_digits < 5)
        pred_vector['high_count_prev'] = np.sum(last_digits >= 5)
        
        pred_vector['dayofweek'] = target_date.dayofweek
        pred_vector['is_weekend'] = 1 if target_date.dayofweek in [5, 6] else 0
        pred_vector['day_sin'] = np.sin(2 * np.pi * target_date.dayofweek / 6.0)
        pred_vector['day_cos'] = np.cos(2 * np.pi * target_date.dayofweek / 6.0)
        pred_vector['as_kop_sum_prev'] = last_row['as'] + last_row['kop']
        pred_vector['kepala_ekor_diff_prev'] = last_row['kepala'] - last_row['ekor']
        pred_vector['as_mul_kop_prev'] = last_row['as'] * last_row['kop']
        pred_vector['kepala_mul_ekor_prev'] = last_row['kepala'] * last_row['ekor']
        vol_window = self.feature_config.get("volatility_window", 10)
        for d in self.digits:
            pred_vector[f'{d}_skew_{vol_window}_prev'] = last_known_data[d].rolling(vol_window).skew().iloc[-1]
            pred_vector[f'{d}_kurt_{vol_window}_prev'] = last_known_data[d].rolling(vol_window).kurt().iloc[-1]
        adv_window = 30
        for d in self.digits:
            rolling_mean = last_known_data[d].rolling(window=adv_window, min_periods=1).mean().iloc[-1]
            rolling_std = last_known_data[d].rolling(window=adv_window, min_periods=1).std().replace(0, 1).iloc[-1]
            pred_vector[f'{d}_zscore_prev'] = (last_row[d] - rolling_mean) / rolling_std
            pred_vector[f'{d}_streak_prev'] = self._calculate_streak(last_known_data[d]).iloc[-1]
            pred_vector[f'{d}_trend_change_prev'] = self._detect_trend_changes(last_known_data[d]).iloc[-1]
        for d in self.digits:
            for i in range(1, self.timesteps + 1):
                pred_vector[f'{d}_lag_{i}'] = last_known_data[d].iloc[-i]
        
        df_pred = pd.DataFrame([pred_vector])
        return df_pred.fillna(0)

class ModelTrainer:
    def __init__(self, digits, training_params, pasaran):
        self.digits = digits
        self.training_params = training_params
        self.pasaran = pasaran

    @error_handler(logger)
    def train_digit_model(self, X_full, y_full, digit, existing_model=None, feature_subset=None, sample_weights=None):
        try:
            xgb_params = self.training_params

            le = LabelEncoder()
            y_encoded = le.fit_transform(y_full)
            if len(le.classes_) < 2:
                logger.warning(f"Hanya ada satu kelas unik untuk digit {digit}. Skipping training.")
                return None, None, None, None
            X_train = X_full[feature_subset] if feature_subset else X_full
            
            if TRAINING_PENALTY_CONFIG["ENABLED"]:
                class_counts = y_full.value_counts()
                total_samples = len(y_full)
                class_weights = total_samples / (len(le.classes_) * class_counts.reindex(le.classes_, fill_value=0.001))
                class_weights = class_weights / class_weights.sum() * len(le.classes_)
                dynamic_weights = pd.Series([class_weights.get(c, 1.0) for c in y_full], index=y_full.index)
                if sample_weights is not None:
                    sample_weights = sample_weights * dynamic_weights
                else:
                    sample_weights = dynamic_weights

            model = xgb.XGBClassifier(**xgb_params)

            logger.info(f"Training model untuk {digit}. Warm-start: {'YES' if existing_model else 'NO'}.")
            
            model.fit(X_train, y_encoded, eval_set=[(X_train, y_encoded)], verbose=False, xgb_model=existing_model, sample_weight=sample_weights)

            importance_scores = model.get_booster().get_score(importance_type='weight')
            feature_importance = {feature: importance_scores.get(feature, 0) for feature in X_train.columns}
            return model, le, feature_importance, y_encoded
        except Exception as e:
            logger.error(f"Error training model untuk digit {digit}: {e}", exc_info=True)
            return None, None, None, None
    
    @error_handler(logger)
    def train_cb_digit_model(self, X_full, y_full, digit, sample_weights=None):
        try:
            model = xgb.XGBClassifier(**self.training_params) # Use passed params
            
            if 'scale_pos_weight' not in self.training_params:
                scale_pos_weight = (y_full == 0).sum() / (y_full == 1).sum() if (y_full == 1).sum() > 0 else 1
                model.set_params(scale_pos_weight=scale_pos_weight)

            logger.info(f"Training CB model untuk digit '{digit}'.")
            
            model.fit(X_full, y_full, eval_set=[(X_full, y_full)], verbose=False, sample_weight=sample_weights)
            
            return model
        except Exception as e:
            logger.error(f"Error training CB model untuk digit {digit}: {e}", exc_info=True)
            return None

class ModelPredictor:
    def __init__(self, pasaran: str):
        self.pasaran = pasaran.lower()
        self.config = self._get_config()
        self.model_dir_base = os.path.join(MODELS_DIR, self.pasaran)
        self.digits = ["as", "kop", "kepala", "ekor"]
        self.data_manager = DataManager(pasaran)
        self.feature_processor = FeatureProcessor(self.config["strategy"]["timesteps"], self.config["feature_engineering"])
        self.continual_learner = ContinualLearner(self)
        
        self.models: Dict[str, Optional[xgb.XGBClassifier]] = {d: None for d in self.digits}
        self.cb_models: Dict[str, Optional[xgb.XGBClassifier]] = {str(i): None for i in range(10)}
        self.rf_models: Dict[str, Optional[RandomForestClassifier]] = {d: None for d in self.digits}
        self.lgbm_models: Dict[str, Optional[lgb.LGBMClassifier]] = {d: None for d in self.digits}
        self.label_encoders: Dict[str, Optional[LabelEncoder]] = {d: None for d in self.digits}
        self.feature_names: Optional[List[str]] = None
        
        try:
            self.data_manager.get_data()
        except DataFetchingError as e:
            logger.warning(f"Gagal memuat data awal untuk {self.pasaran}: {e}")
        self.models_ready = self.load_models()

    def _get_config(self) -> Dict[str, Any]:
        market_config = MARKET_CONFIGS.get(self.pasaran)
        if not market_config:
            raise KeyError(f"Konfigurasi untuk pasaran '{self.pasaran}' tidak ditemukan.")
        
        config = market_config.copy()
        config["training_params"] = TRAINING_CONFIG_OPTIONS.get('OPTIMIZED')
        return config

    @error_handler(logger)
    def load_models(self) -> bool:
        if not os.path.exists(self.model_dir_base): return False
        features_path = os.path.join(self.model_dir_base, "features.pkl")
        if not os.path.exists(features_path): return False
            
        self.feature_names = joblib.load(features_path)
        
        xgb_loaded, cb_loaded = 0, 0
        for d in self.digits:
            model_path = os.path.join(self.model_dir_base, f"model_{d}.pkl")
            if os.path.exists(model_path):
                self.models[d] = joblib.load(model_path)
                xgb_loaded += 1
            encoder_path = os.path.join(self.model_dir_base, f"encoder_{d}.pkl")
            if os.path.exists(encoder_path):
                self.label_encoders[d] = joblib.load(encoder_path)
        
        for i in range(10):
            cb_model_path = os.path.join(self.model_dir_base, f"model_cb_{i}.pkl")
            if os.path.exists(cb_model_path):
                self.cb_models[str(i)] = joblib.load(cb_model_path)
                cb_loaded += 1
        
        logger.info(f"Model loaded: 4D_XGB({xgb_loaded}/{len(self.digits)}), CB_XGB({cb_loaded}/10)")
        return xgb_loaded == len(self.digits) and cb_loaded == 10

    @error_handler(logger)
    def train_model(self, training_mode: str = 'OPTIMIZED', use_recency_bias: bool = True, custom_data: Optional[pd.DataFrame] = None) -> bool:
        logger.info(f"Memulai training untuk {self.pasaran} mode {training_mode}.")
        
        os.makedirs(self.model_dir_base, exist_ok=True)
        df_full = custom_data if custom_data is not None else self.data_manager.get_data(force_refresh=True, force_github=True)
        training_df = self.feature_processor.fit_transform(df_full)
        self.feature_names = self.feature_processor.feature_names
        
        min_samples = self.config["strategy"]["min_training_samples"]
        if len(training_df) < min_samples:
            raise TrainingError(f"Data tidak cukup untuk training. Perlu {min_samples}, tersedia {len(training_df)}.")
        
        joblib.dump(self.feature_names, os.path.join(self.model_dir_base, "features.pkl"))
        
        sample_weights = None
        if use_recency_bias and ADAPTIVE_LEARNING_CONFIG["USE_RECENCY_WEIGHTING"]:
            days_since_latest = (training_df['date'].max() - training_df['date']).dt.days
            decay_rate = np.log(2) / ADAPTIVE_LEARNING_CONFIG["RECENCY_HALF_LIFE_DAYS"]
            sample_weights = pd.Series(np.exp(-decay_rate * days_since_latest), index=training_df.index)

        X_full = training_df[self.feature_names]

        logger.info("--- Memulai Training Model 4D ---")
        for digit in self.digits:
            training_params = {}
            if training_mode == 'AUTO':
                best_params_path = os.path.join(self.model_dir_base, f"best_params_{digit}.json")
                if os.path.exists(best_params_path):
                    logger.info(f"Mode AUTO: Menggunakan parameter hasil tuning untuk 4D-{digit}.")
                    with open(best_params_path, 'r') as f:
                        training_params = json.load(f)
                else:
                    logger.warning(f"Mode AUTO: File best_params_{digit}.json tidak ditemukan. Fallback ke mode OPTIMIZED.")
                    training_params = TRAINING_CONFIG_OPTIONS['OPTIMIZED'].get("xgb_params", {}).get(digit, {})
            else:
                config_options = TRAINING_CONFIG_OPTIONS.get(training_mode, TRAINING_CONFIG_OPTIONS['OPTIMIZED'])
                training_params = config_options.get("xgb_params", {}).get(digit, {})

            model_trainer = ModelTrainer(self.digits, training_params, self.pasaran)
            y_full = training_df[digit]
            if y_full.nunique() < 2: continue
            
            model, le, importance, _ = model_trainer.train_digit_model(X_full, y_full, digit, sample_weights=sample_weights)
            if model and le and importance:
                joblib.dump(model, os.path.join(self.model_dir_base, f"model_{digit}.pkl"))
                joblib.dump(le, os.path.join(self.model_dir_base, f"encoder_{digit}.pkl"))
                pd.DataFrame(importance.items(), columns=['feature', 'weight']).sort_values('weight', ascending=False).to_csv(os.path.join(self.model_dir_base, f"feature_importance_{digit}.csv"), index=False)
                self._check_for_drift(digit)

        logger.info("--- Memulai Training Model Colok Bebas (CB) ---")
        for i in range(10):
            digit = str(i)
            cb_params = {}
            if training_mode == 'AUTO':
                best_params_path = os.path.join(self.model_dir_base, f"best_params_cb_{digit}.json")
                if os.path.exists(best_params_path):
                    logger.info(f"Mode AUTO: Menggunakan parameter hasil tuning untuk CB-{digit}.")
                    with open(best_params_path, 'r') as f:
                        cb_params = json.load(f)
                else:
                    logger.warning(f"Mode AUTO: File best_params_cb_{digit}.json tidak ditemukan. Fallback ke parameter default.")
                    cb_params = XGB_PARAMS_CB
            else:
                cb_params = XGB_PARAMS_CB

            cb_trainer = ModelTrainer(self.digits, cb_params, self.pasaran)
            y_cb = training_df[f'cb_target_{digit}']
            if y_cb.nunique() < 2: continue
            
            cb_model = cb_trainer.train_cb_digit_model(X_full, y_cb, digit, sample_weights=sample_weights)
            if cb_model:
                joblib.dump(cb_model, os.path.join(self.model_dir_base, f"model_cb_{digit}.pkl"))

        self.models_ready = self.load_models()
        return self.models_ready

    def _determine_colok_bebas_dedicated(self, latest_features: pd.DataFrame) -> str:
        cb_probas = {}
        for i in range(10):
            model = self.cb_models.get(str(i))
            if model:
                proba = model.predict_proba(latest_features)[0][1]
                cb_probas[str(i)] = proba
        return max(cb_probas, key=cb_probas.get) if cb_probas else ""

    def _determine_colok_bebas_aggregated(self, all_4d_probas: Dict[str, np.ndarray]) -> str:
        total_probs = {str(i): 0.0 for i in range(10)}
        for digit_pos, probabilities in all_4d_probas.items():
            encoder = self.label_encoders.get(digit_pos)
            if not encoder: continue
            for i, class_label in enumerate(encoder.classes_):
                total_probs[str(class_label)] += probabilities[i]
        return max(total_probs, key=total_probs.get) if any(p > 0 for p in total_probs.values()) else ""

    def _determine_angka_main(self, predictions: Dict[str, List[str]]) -> List[str]:
        am_candidates = OrderedDict()
        am_candidates[predictions['as'][0]] = None
        am_candidates[predictions['kop'][0]] = None
        am_candidates[predictions['kepala'][0]] = None
        am_candidates[predictions['ekor'][0]] = None
        
        if len(am_candidates) < 4:
            backup_pool = [
                predictions['as'][1], predictions['kop'][1], 
                predictions['kepala'][1], predictions['ekor'][1]
            ]
            for cand in backup_pool:
                if len(am_candidates) >= 4:
                    break
                am_candidates[cand] = None
        
        return sorted(list(am_candidates.keys()))[:4]

    @error_handler(drift_logger)
    def predict_next_day(self, target_date_str: Optional[str] = None, for_evaluation: bool = False) -> Dict[str, Any]:
        if not self.models_ready:
            raise PredictionError("Model tidak siap. Silakan jalankan training.")
        
        target_date = pd.to_datetime(target_date_str) if target_date_str else datetime.now() + timedelta(days=1)
        base_df = self.data_manager.get_data()
        latest_features = self.feature_processor.transform_for_prediction(base_df, target_date)
        latest_features = latest_features.reindex(columns=self.feature_names, fill_value=0)
        
        predictions = {}
        all_4d_probas = {}
        for d in self.digits:
            encoder = self.label_encoders[d]
            if not encoder: raise PredictionError(f"Encoder untuk '{d}' tidak tersedia.")
            
            raw_probabilities = self.models[d].predict_proba(latest_features)[0]
            all_4d_probas[d] = raw_probabilities
            top_indices = np.argsort(raw_probabilities)[::-1]
            top_three_digits = encoder.inverse_transform(top_indices[:3])
            predictions[d] = [str(digit) for digit in top_three_digits]

        strategy = CB_STRATEGY_CONFIG.get(self.pasaran, "dedicated")
        logger.info(f"Menggunakan strategi CB '{strategy}' untuk pasaran {self.pasaran}.")

        if strategy == "aggregated":
            colok_bebas = self._determine_colok_bebas_aggregated(all_4d_probas)
        else:
            colok_bebas = self._determine_colok_bebas_dedicated(latest_features)
        
        angka_main = self._determine_angka_main(predictions)

        return {
            "prediction_date": target_date.strftime("%Y-%m-%d"),
            "final_4d_prediction": f"{predictions['as'][0]}{predictions['kop'][0]}{predictions['kepala'][0]}{predictions['ekor'][0]}",
            "kandidat_as": ", ".join(predictions['as']),
            "kandidat_kop": ", ".join(predictions['kop']),
            "kandidat_kepala": ", ".join(predictions['kepala']),
            "kandidat_ekor": ", ".join(predictions['ekor']),
            "angka_main": ", ".join(angka_main),
            "colok_bebas": colok_bebas
        }
    
    @error_handler(drift_logger)
    def evaluate_performance(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        df = self.data_manager.get_data(force_github=True)
        eval_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        if eval_df.empty: return {"summary": {"error": "Tidak ada data pada periode yang diminta"}, "results": []}
        results_list = []
        for _, row in eval_df.iterrows():
            try:
                pred_result = self.predict_next_day(row['date'].strftime('%Y-%m-%d'), for_evaluation=True)
                actual_result = row['result']
                if pred_result and actual_result:
                    results_list.append({ "date": row['date'].strftime('%Y-%m-%d'), "actual": actual_result, **pred_result })
            except PredictionError as e:
                logger.warning(f"Skipping evaluasi untuk {row['date'].strftime('%Y-%m-%d')}: {e}")
        if not results_list: return {"summary": {"error": "Gagal menghasilkan prediksi."}, "results": []}
        eval_summary_df = pd.DataFrame(results_list)
        def check_hit_digits(p, a): return a in p.replace(' ','').split(',') if isinstance(p, str) else False
        as_accuracy = eval_summary_df.apply(lambda r: check_hit_digits(r['kandidat_as'], r['actual'][0]), axis=1).mean()
        kop_accuracy = eval_summary_df.apply(lambda r: check_hit_digits(r['kandidat_kop'], r['actual'][1]), axis=1).mean()
        kepala_accuracy = eval_summary_df.apply(lambda r: check_hit_digits(r['kandidat_kepala'], r['actual'][2]), axis=1).mean()
        ekor_accuracy = eval_summary_df.apply(lambda r: check_hit_digits(r['kandidat_ekor'], r['actual'][3]), axis=1).mean()
        def check_am(a, p): return any(d in a for d in p.replace(' ','').split(',')) if isinstance(p, str) and a else False
        def check_cb(a, p): return p in a if isinstance(p, str) and a else False
        am_accuracy = eval_summary_df.apply(lambda r: check_am(r['actual'], r['angka_main']), axis=1).mean()
        cb_accuracy = eval_summary_df.apply(lambda r: check_cb(r['actual'], r['colok_bebas']), axis=1).mean()
        
        summary = {
            "total_days_evaluated": len(eval_summary_df),
            "as_accuracy": as_accuracy, "kop_accuracy": kop_accuracy, "kepala_accuracy": kepala_accuracy, "ekor_accuracy": ekor_accuracy,
            "am_accuracy": am_accuracy, "cb_accuracy": cb_accuracy,
            "retraining_recommended": False, "retraining_reason": "N/A"
        }
        if (summary["kepala_accuracy"] < ACCURACY_THRESHOLD_FOR_RETRAIN or summary["ekor_accuracy"] < ACCURACY_THRESHOLD_FOR_RETRAIN):
            summary["retraining_recommended"] = True
            summary["retraining_reason"] = f"Akurasi Kepala/Ekor ({summary['kepala_accuracy']:.1%}/{summary['ekor_accuracy']:.1%}) di bawah ambang batas."
        return {"summary": summary, "results": results_list}

    @error_handler(drift_logger)
    def _check_for_drift(self, digit: str) -> bool:
        new_importance_path = os.path.join(self.model_dir_base, f"feature_importance_{digit}.csv")
        baseline_path = os.path.join(self.model_dir_base, f"feature_importance_{digit}_baseline.csv")
        if not os.path.exists(new_importance_path): return False
        if not os.path.exists(baseline_path):
            shutil.copy(new_importance_path, baseline_path)
            drift_logger.info(f"BASELINE CREATED for {self.pasaran}-{digit}.")
            return False
        try:
            new_df = pd.read_csv(new_importance_path)
            baseline_df = pd.read_csv(baseline_path)
            top_features = set(baseline_df.head(50)['feature']).union(set(new_df.head(50)['feature']))
            new_df = new_df.set_index('feature').reindex(top_features, fill_value=0)
            baseline_df = baseline_df.set_index('feature').reindex(top_features, fill_value=0)
            baseline_dist = baseline_df['weight'] / baseline_df['weight'].sum()
            new_dist = new_df['weight'] / new_df['weight'].sum()
            jaccard_sim = len(set(baseline_df.head(20).index) & set(new_df.head(20).index)) / len(set(baseline_df.head(20).index) | set(new_df.head(20).index))
            baseline_dist[baseline_dist == 0] = 1e-6
            new_dist[new_dist == 0] = 1e-6
            psi = np.sum((new_dist - baseline_dist) * np.log(new_dist / baseline_dist))
            kl_div = np.sum(new_dist * np.log(new_dist / baseline_dist))
            drift_logger.info(f"Drift Check for {self.pasaran}-{digit}: Jaccard={jaccard_sim:.3f}, PSI={psi:.4f}, KL-Div={kl_div:.4f}")
            if jaccard_sim < DRIFT_THRESHOLD or psi > 0.25:
                drift_logger.warning(f"DRIFT DETECTED for {self.pasaran}-{digit}! Jaccard ({jaccard_sim:.3f}) < {DRIFT_THRESHOLD} or PSI ({psi:.4f}) > 0.25. Updating baseline.")
                shutil.copy(new_importance_path, baseline_path)
                return True
            return False
        except Exception as e:
            drift_logger.error(f"Gagal saat memeriksa drift untuk {self.pasaran}-{digit}: {e}")
            return False