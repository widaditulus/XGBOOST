# predictor.py (Final - Ditambahkan Fitur Angka Berpasangan & Pengetahuan Lainnya)
# BEJO
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
# PERBAIKAN: Impor yang hilang untuk mengatasi NameError
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from constants import (MODELS_DIR, MARKET_CONFIGS, DRIFT_THRESHOLD,
                     ACCURACY_THRESHOLD_FOR_RETRAIN, ADAPTIVE_LEARNING_CONFIG,
                     ENSEMBLE_CONFIG, TRAINING_PENALTY_CONFIG,
                     CB_STRATEGY_CONFIG, CRITICAL_ACCURACY_THRESHOLD)
from utils import logger, error_handler, drift_logger
from model_config import TRAINING_CONFIG_OPTIONS, XGB_PARAMS_CB
from exceptions import TrainingError, PredictionError, DataFetchingError
from data_fetcher import DataFetcher
from evaluation import calculate_brier_score, calculate_ece
from ensemble_helper import train_ensemble_models, ensemble_predict_proba, train_temporary_ensemble_models
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
    def get_data(self, force_refresh: bool = False, force_github: bool = False) -> Optional[pd.DataFrame]:
        with self.lock:
            if self.df is None or force_refresh:
                df_raw = self.fetcher.fetch_data(force_github=force_github)
                df_validated = self._validate_data(df_raw)
                df_sorted = df_validated.sort_values("date").reset_index(drop=True)
                self.df = df_sorted
            return self.df.copy()

    @error_handler(logger)
    def check_data_freshness(self) -> Dict[str, Any]:
        local_df = self.get_data(force_refresh=False, force_github=False)
        if local_df is None or local_df.empty:
            return {"status": "stale", "message": "Data lokal tidak ditemukan."}
        local_latest_date = local_df['date'].max()
        try:
            remote_df = self.fetcher.fetch_data(force_github=True)
            if remote_df is None or remote_df.empty:
                return {"status": "error", "message": "Gagal mengambil data remote."}
            remote_latest_date = self._validate_data(remote_df)['date'].max()
            status = "stale" if local_latest_date < remote_latest_date else "latest"
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
        
        new_features['as_kop_sum_prev'] = (df['as'] + df['kop']).shift(1)
        new_features['as_kop_diff_prev'] = (df['as'] - df['kop']).shift(1)
        new_features['as_kop_mul_prev'] = (df['as'] * df['kop']).shift(1)
        new_features['kepala_ekor_sum_prev'] = (df['kepala'] + df['ekor']).shift(1)
        new_features['kepala_ekor_diff_prev'] = (df['kepala'] - df['ekor']).shift(1)
        new_features['kepala_ekor_mul_prev'] = (df['kepala'] * df['ekor']).shift(1)
        new_features['as_ekor_sum_prev'] = (df['as'] + df['ekor']).shift(1)
        new_features['kop_kepala_sum_prev'] = (df['kop'] + df['kepala']).shift(1)
        
        new_features['digit_sum_prev'] = digit_cols.sum(axis=1).shift(1)
        new_features['even_count_prev'] = (digit_cols % 2 == 0).sum(axis=1).shift(1)
        new_features['odd_count_prev'] = (digit_cols % 2 != 0).sum(axis=1).shift(1)
        new_features['low_count_prev'] = (digit_cols < 5).sum(axis=1).shift(1)
        new_features['high_count_prev'] = (digit_cols >= 5).sum(axis=1).shift(1)

        new_features['dayofweek'] = df['date'].dt.dayofweek
        new_features['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
        new_features['day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 6.0)
        new_features['day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 6.0)
        
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
        
        for d in self.digits:
            for num in range(10):
                is_num = (df[d] == num)
                new_features[f'{d}_{num}_freq_30d'] = is_num.shift(1).rolling(window=30, min_periods=1).sum()
                
                not_is_num = (df[d] != num)
                days_since_seen_series = not_is_num.groupby(not_is_num.cumsum()).cumcount()
                new_features[f'{d}_{num}_days_since_seen'] = days_since_seen_series.shift(1)

        shifted_digits = df[self.digits].shift(1)
        new_features['unique_digits_prev'] = shifted_digits.nunique(axis=1)
        new_features['range_prev'] = shifted_digits.max(axis=1) - shifted_digits.min(axis=1)
        new_features['twin_count_prev'] = shifted_digits.apply(lambda row: row.value_counts().gt(1).sum(), axis=1)

        for d in self.digits:
            short_ma = df[d].shift(1).rolling(window=7, min_periods=1).mean()
            long_ma = df[d].shift(1).rolling(window=30, min_periods=1).mean()
            new_features[f'{d}_delta_ma_7_30'] = short_ma - long_ma

        df_features = pd.DataFrame(new_features, index=df.index)
        training_df = pd.concat([df, df_features], axis=1)
        
        self.feature_names = [col for col in training_df.columns if col not in ['date', 'result'] + self.digits + self.cb_target_cols]
        
        training_df.dropna(subset=self.feature_names + self.digits, inplace=True)
        training_df[self.feature_names] = training_df[self.feature_names].fillna(0)
        
        return training_df

    def transform_for_prediction(self, historical_df: pd.DataFrame, target_date: datetime) -> pd.DataFrame:
        required_len = max(self.timesteps, 30) + 1 
        if len(historical_df) < required_len:
            raise PredictionError(f"Data historis tidak cukup ({len(historical_df)}) untuk prediksi. Perlu {required_len}.")

        pred_row = pd.DataFrame([{'date': target_date, 'result': '0000'}])
        df_for_calc = pd.concat([historical_df.iloc[-required_len:], pred_row], ignore_index=True)
        df_for_calc['date'] = pd.to_datetime(df_for_calc['date'])
        
        df_with_features = self.fit_transform(df_for_calc)
        
        return df_with_features.iloc[[-1]][self.feature_names]

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
            if len(le.classes_) < 2: return None, None, None, None
            X_train = X_full[feature_subset] if feature_subset else X_full
            
            if TRAINING_PENALTY_CONFIG["ENABLED"]:
                class_counts = y_full.value_counts()
                total_samples = len(y_full)
                class_weights = total_samples / (len(le.classes_) * class_counts.reindex(le.classes_, fill_value=0.001))
                dynamic_weights = pd.Series([class_weights.get(c, 1.0) for c in y_full], index=y_full.index)
                sample_weights = sample_weights * dynamic_weights if sample_weights is not None else dynamic_weights

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, y_encoded, eval_set=[(X_train, y_encoded)], verbose=False, xgb_model=existing_model, sample_weight=sample_weights)
            model.feature_names_ = list(X_train.columns)
            importance_scores = model.get_booster().get_score(importance_type='weight')
            return model, le, importance_scores, y_encoded
        except Exception as e:
            logger.error(f"Error training model untuk digit {digit}: {e}", exc_info=True)
            return None, None, None, None
    
    @error_handler(logger)
    def train_cb_digit_model(self, X_full, y_full, digit, sample_weights=None):
        try:
            model = xgb.XGBClassifier(**self.training_params) 
            if 'scale_pos_weight' not in self.training_params:
                scale_pos_weight = (y_full == 0).sum() / max(1, (y_full == 1).sum())
                model.set_params(scale_pos_weight=scale_pos_weight)
            model.fit(X_full, y_full, eval_set=[(X_full, y_full)], verbose=False, sample_weight=sample_weights)
            model.feature_names_ = list(X_full.columns)
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
        try:
            self.data_manager.get_data()
        except DataFetchingError as e:
            logger.warning(f"Gagal memuat data awal untuk {self.pasaran}: {e}")
        self.models_ready = self.load_models()

    def _get_config(self) -> Dict[str, Any]:
        market_config = MARKET_CONFIGS.get(self.pasaran)
        if not market_config: raise KeyError(f"Konfigurasi untuk pasaran '{self.pasaran}' tidak ditemukan.")
        config = market_config.copy()
        config["training_params"] = TRAINING_CONFIG_OPTIONS.get('OPTIMIZED')
        return config

    @error_handler(logger)
    def load_models(self) -> bool:
        if not os.path.exists(self.model_dir_base): return False
        xgb_loaded, cb_loaded, rf_loaded, lgbm_loaded = 0, 0, 0, 0
        for d in self.digits:
            model_path = os.path.join(self.model_dir_base, f"model_{d}.pkl")
            if os.path.exists(model_path):
                self.models[d] = joblib.load(model_path)
                if hasattr(self.models[d], 'feature_names_'): xgb_loaded += 1
                else: self.models[d] = None
            encoder_path = os.path.join(self.model_dir_base, f"encoder_{d}.pkl")
            if os.path.exists(encoder_path): self.label_encoders[d] = joblib.load(encoder_path)
            rf_model_path = os.path.join(self.model_dir_base, f"rf_model_{d}.pkl")
            if os.path.exists(rf_model_path): self.rf_models[d] = joblib.load(rf_model_path); rf_loaded += 1
            lgbm_model_path = os.path.join(self.model_dir_base, f"lgbm_model_{d}.pkl")
            if os.path.exists(lgbm_model_path): self.lgbm_models[d] = joblib.load(lgbm_model_path); lgbm_loaded += 1
        for i in range(10):
            cb_model_path = os.path.join(self.model_dir_base, f"model_cb_{i}.pkl")
            if os.path.exists(cb_model_path):
                self.cb_models[str(i)] = joblib.load(cb_model_path)
                if hasattr(self.cb_models[str(i)], 'feature_names_'): cb_loaded += 1
                else: self.cb_models[str(i)] = None
        return xgb_loaded == len(self.digits) and cb_loaded == 10

    @error_handler(logger)
    def train_model(self, training_mode: str = 'OPTIMIZED', use_recency_bias: bool = True, custom_data: Optional[pd.DataFrame] = None) -> Any:
        is_evaluation_run = (custom_data is not None)
        df_full = custom_data if is_evaluation_run else self.data_manager.get_data(force_refresh=True, force_github=True)
        feature_processor = FeatureProcessor(self.config["strategy"]["timesteps"], self.config["feature_engineering"])
        training_df = feature_processor.fit_transform(df_full)
        current_feature_names = feature_processor.feature_names
        if len(training_df) < self.config["strategy"]["min_training_samples"]:
            raise TrainingError(f"Data tidak cukup. Perlu {self.config['strategy']['min_training_samples']}, tersedia {len(training_df)}.")
        if not is_evaluation_run: os.makedirs(self.model_dir_base, exist_ok=True)
        sample_weights = None
        if use_recency_bias and ADAPTIVE_LEARNING_CONFIG["USE_RECENCY_WEIGHTING"]:
            days_since_latest = (training_df['date'].max() - training_df['date']).dt.days
            decay_rate = np.log(2) / ADAPTIVE_LEARNING_CONFIG["RECENCY_HALF_LIFE_DAYS"]
            sample_weights = pd.Series(np.exp(-decay_rate * days_since_latest), index=training_df.index)
        X_full = training_df[current_feature_names]
        temp_models_4d, temp_cb_models, temp_encoders, temp_rf_models, temp_lgbm_models = {}, {}, {}, {}, {}
        for digit in self.digits:
            config_options = TRAINING_CONFIG_OPTIONS.get(training_mode, TRAINING_CONFIG_OPTIONS['OPTIMIZED'])
            training_params = config_options.get("xgb_params", {}).get(digit, {})
            model_trainer = ModelTrainer(self.digits, training_params, self.pasaran)
            y_full = training_df[digit]
            if y_full.nunique() < 2: continue
            model, le, imp, y_enc = model_trainer.train_digit_model(X_full, y_full, digit, sample_weights=sample_weights)
            if model and le:
                temp_models_4d[digit], temp_encoders[digit] = model, le
                if not is_evaluation_run:
                    joblib.dump(model, os.path.join(self.model_dir_base, f"model_{digit}.pkl"))
                    joblib.dump(le, os.path.join(self.model_dir_base, f"encoder_{digit}.pkl"))
                    if ENSEMBLE_CONFIG.get("USE_ENSEMBLE", False): train_ensemble_models(X_full, y_enc, self.model_dir_base, digit)
                elif is_evaluation_run and training_mode == 'deep':
                    temp_rf_models[digit], temp_lgbm_models[digit] = train_temporary_ensemble_models(X_full, y_enc, digit)
        for i in range(10):
            digit, cb_params = str(i), XGB_PARAMS_CB
            cb_trainer = ModelTrainer(self.digits, cb_params, self.pasaran)
            y_cb = training_df[f'cb_target_{digit}']
            if y_cb.nunique() < 2: continue
            cb_model = cb_trainer.train_cb_digit_model(X_full, y_cb, digit, sample_weights=sample_weights)
            if cb_model:
                temp_cb_models[digit] = cb_model
                if not is_evaluation_run: joblib.dump(cb_model, os.path.join(self.model_dir_base, f"model_cb_{digit}.pkl"))
        if is_evaluation_run: return temp_models_4d, temp_cb_models, temp_encoders, current_feature_names, temp_rf_models, temp_lgbm_models
        self.models_ready = self.load_models()
        return self.models_ready

    def _determine_colok_bebas_dedicated(self, latest_features: pd.DataFrame, cb_models: Dict) -> str:
        cb_probas = {}
        for i in range(10):
            model = cb_models.get(str(i))
            if model:
                validated_features = self._validate_and_reorder_features(latest_features, model)
                proba = model.predict_proba(validated_features)[0][1]
                cb_probas[str(i)] = proba
        return max(cb_probas, key=cb_probas.get) if cb_probas else ""
    
    def _determine_colok_bebas_aggregated(self, all_4d_probas: Dict[str, np.ndarray], encoders: Dict) -> str:
        total_probs = {str(i): 0.0 for i in range(10)}
        for digit_pos, probabilities in all_4d_probas.items():
            encoder = encoders.get(digit_pos)
            if not encoder: continue
            for i, class_label in enumerate(encoder.classes_): total_probs[str(class_label)] += probabilities[i]
        return max(total_probs, key=total_probs.get) if any(p > 0 for p in total_probs.values()) else ""

    def _determine_angka_main(self, predictions: Dict[str, List[str]]) -> List[str]:
        am_candidates = OrderedDict()
        for d in self.digits: am_candidates[predictions[d][0]] = None
        if len(am_candidates) < 4:
            for d in self.digits:
                if len(am_candidates) >= 4: break
                am_candidates[predictions[d][1]] = None
        return sorted(list(am_candidates.keys()))[:4]

    def _validate_and_reorder_features(self, features: pd.DataFrame, model: Any) -> pd.DataFrame:
        if not hasattr(model, 'feature_names_'):
            raise PredictionError(f"Model '{type(model).__name__}' tidak memiliki 'feature_names_'. Harap latih ulang.")
        model_features = model.feature_names_
        input_features_set = set(features.columns)
        model_features_set = set(model_features)
        if input_features_set != model_features_set:
            missing = model_features_set - input_features_set
            extra = input_features_set - model_features_set
            raise PredictionError(f"Ketidakcocokan fitur. Hilang: {missing if missing else 'None'}. Tambahan: {extra if extra else 'None'}. Harap latih ulang.")
        return features[model_features]

    @error_handler(logger)
    def predict_next_day(self, target_date_str: Optional[str] = None, evaluation_mode: str = 'deep') -> Dict[str, Any]:
        if not self.models_ready: raise PredictionError("Model tidak siap. Silakan jalankan training.")
        target_date = pd.to_datetime(target_date_str) if target_date_str else datetime.now() + timedelta(days=1)
        full_df = self.data_manager.get_data()
        historical_df = full_df[full_df['date'] < target_date].copy()
        if historical_df.empty: raise PredictionError(f"Tidak ada data historis sebelum {target_date.strftime('%Y-%m-%d')}.")
        latest_features_raw = self.feature_processor.transform_for_prediction(historical_df, target_date)
        predictions, all_4d_probas = {}, {}
        for d in self.digits:
            encoder = self.label_encoders[d]
            if not encoder: raise PredictionError(f"Encoder untuk '{d}' tidak tersedia.")
            if evaluation_mode == 'deep':
                active_models = {'xgb': self.models.get(d), 'rf': self.rf_models.get(d), 'lgbm': self.lgbm_models.get(d)}
                loaded_models = {name: model for name, model in active_models.items() if model is not None}
                if not loaded_models: raise PredictionError(f"Tidak ada model yang termuat untuk digit '{d}'.")
                probabilities = ensemble_predict_proba(loaded_models, latest_features_raw)[0]
            else: 
                xgb_model = self.models.get(d)
                if not xgb_model: raise PredictionError(f"Model XGBoost tidak ditemukan untuk digit '{d}'.")
                validated_features = self._validate_and_reorder_features(latest_features_raw, xgb_model)
                probabilities = xgb_model.predict_proba(validated_features)[0]
            all_4d_probas[d] = probabilities
            top_indices = np.argsort(probabilities)[::-1]
            top_three_digits = encoder.inverse_transform(top_indices[:3])
            predictions[d] = [str(digit) for digit in top_three_digits]
        strategy = CB_STRATEGY_CONFIG.get(self.pasaran, "dedicated")
        main_model_for_schema = self.models.get('as')
        if not main_model_for_schema: raise PredictionError("Model 'as' tidak ditemukan untuk skema fitur CB.")
        validated_features_for_cb = self._validate_and_reorder_features(latest_features_raw, main_model_for_schema)
        colok_bebas = self._determine_colok_bebas_aggregated(all_4d_probas, self.label_encoders) if strategy == "aggregated" else self._determine_colok_bebas_dedicated(validated_features_for_cb, self.cb_models)
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
    
    @error_handler(logger)
    def evaluate_performance(self, start_date: datetime, end_date: datetime, evaluation_mode: str = 'quick') -> Dict[str, Any]:
        RETRAIN_INTERVAL_DAYS = 7 
        full_df = self.data_manager.get_data()
        eval_range = full_df[(full_df['date'] >= start_date) & (full_df['date'] <= end_date)].copy()
        if eval_range.empty: return {"summary": {"error": "Tidak ada data pada periode yang diminta"}, "results": []}
        results_list, temp_models, last_retrain_date = [], None, None
        for index, row in eval_range.iterrows():
            eval_date, actual_result = row['date'], row['result']
            historical_df = full_df[full_df['date'] < eval_date].copy()
            if historical_df.empty: continue
            try:
                if temp_models is None or (eval_date - last_retrain_date).days >= RETRAIN_INTERVAL_DAYS:
                    temp_models = self.train_model(training_mode=evaluation_mode, custom_data=historical_df)
                    last_retrain_date = eval_date
                pred_result = self._predict_with_custom_models(historical_df, eval_date, evaluation_mode, *temp_models)
                if pred_result: results_list.append({ "date": eval_date.strftime('%Y-%m-%d'), "actual": actual_result, **pred_result })
            except (PredictionError, TrainingError) as e:
                logger.error(f"Error saat backtest untuk {eval_date}: {e}", exc_info=True)
                continue
        if not results_list: return {"summary": {"error": "Gagal menghasilkan prediksi."}, "results": []}
        eval_summary_df = pd.DataFrame(results_list)
        def check_hit(p, a): return a in p.replace(' ','').split(',') if isinstance(p, str) else False
        summary = {
            "total_days_evaluated": len(eval_summary_df),
            "as_accuracy": eval_summary_df.apply(lambda r: check_hit(r['kandidat_as'], r['actual'][0]), axis=1).mean(),
            "kop_accuracy": eval_summary_df.apply(lambda r: check_hit(r['kandidat_kop'], r['actual'][1]), axis=1).mean(),
            "kepala_accuracy": eval_summary_df.apply(lambda r: check_hit(r['kandidat_kepala'], r['actual'][2]), axis=1).mean(),
            "ekor_accuracy": eval_summary_df.apply(lambda r: check_hit(r['kandidat_ekor'], r['actual'][3]), axis=1).mean(),
            "am_accuracy": eval_summary_df.apply(lambda r: any(d in r['actual'] for d in r['angka_main'].replace(' ','').split(',')), axis=1).mean(),
            "cb_accuracy": eval_summary_df.apply(lambda r: r['colok_bebas'] in r['actual'], axis=1).mean(),
        }
        # PERBAIKAN: Konversi eksplisit ke bool standar Python untuk mencegah TypeError
        summary["retraining_recommended"] = bool(summary["kepala_accuracy"] < ACCURACY_THRESHOLD_FOR_RETRAIN or summary["ekor_accuracy"] < CRITICAL_ACCURACY_THRESHOLD)
        summary["retraining_reason"] = f"Akurasi Kepala/Ekor ({summary['kepala_accuracy']:.1%}/{summary['ekor_accuracy']:.1%}) di bawah ambang batas." if summary["retraining_recommended"] else "N/A"
        return {"summary": summary, "results": results_list}
        
    def _predict_with_custom_models(self, historical_df, target_date, evaluation_mode, custom_models_4d, custom_cb_models, custom_encoders, feature_names, custom_rf_models, custom_lgbm_models) -> Dict[str, Any]:
        eval_feature_processor = FeatureProcessor(self.config["strategy"]["timesteps"], self.config["feature_engineering"])
        latest_features = eval_feature_processor.transform_for_prediction(historical_df, target_date)
        latest_features = latest_features.reindex(columns=feature_names, fill_value=0)
        predictions, all_4d_probas = {}, {}
        for d in self.digits:
            encoder = custom_encoders.get(d)
            if not encoder: raise PredictionError(f"Encoder untuk '{d}' tidak tersedia.")
            if evaluation_mode == 'deep':
                active_models = {'xgb': custom_models_4d.get(d), 'rf': custom_rf_models.get(d), 'lgbm': custom_lgbm_models.get(d)}
                loaded_models = {name: model for name, model in active_models.items() if model is not None}
                if not loaded_models: raise PredictionError(f"Tidak ada model yang termuat untuk digit '{d}'.")
                probabilities = ensemble_predict_proba(loaded_models, latest_features)[0]
            else:
                model = custom_models_4d.get(d)
                if not model: raise PredictionError(f"Model XGBoost tidak ditemukan untuk digit '{d}'.")
                probabilities = model.predict_proba(latest_features)[0]
            all_4d_probas[d] = probabilities
            top_indices = np.argsort(probabilities)[::-1]
            top_three_digits = encoder.inverse_transform(top_indices[:3])
            predictions[d] = [str(digit) for digit in top_three_digits]
        strategy = CB_STRATEGY_CONFIG.get(self.pasaran, "dedicated")
        colok_bebas = self._determine_colok_bebas_aggregated(all_4d_probas, custom_encoders) if strategy == "aggregated" else self._determine_colok_bebas_dedicated(latest_features, custom_cb_models)
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
    def _check_for_drift(self, digit: str) -> bool:
        new_importance_path = os.path.join(self.model_dir_base, f"feature_importance_{digit}.csv")
        baseline_path = os.path.join(self.model_dir_base, f"feature_importance_{digit}_baseline.csv")
        if not os.path.exists(new_importance_path): return False
        if not os.path.exists(baseline_path):
            shutil.copy(new_importance_path, baseline_path)
            return False
        try:
            new_df, baseline_df = pd.read_csv(new_importance_path), pd.read_csv(baseline_path)
            top_features = set(baseline_df.head(50)['feature']).union(set(new_df.head(50)['feature']))
            new_df, baseline_df = new_df.set_index('feature').reindex(top_features, fill_value=0), baseline_df.set_index('feature').reindex(top_features, fill_value=0)
            baseline_dist, new_dist = baseline_df['weight'] / baseline_df['weight'].sum(), new_df['weight'] / new_df['weight'].sum()
            jaccard_sim = len(set(baseline_df.head(20).index) & set(new_df.head(20).index)) / len(set(baseline_df.head(20).index) | set(new_df.head(20).index))
            baseline_dist[baseline_dist == 0] = 1e-6
            new_dist[new_dist == 0] = 1e-6
            psi = np.sum((new_dist - baseline_dist) * np.log(new_dist / baseline_dist))
            if jaccard_sim < DRIFT_THRESHOLD or psi > 0.25:
                drift_logger.warning(f"DRIFT DETECTED for {self.pasaran}-{digit}! Jaccard ({jaccard_sim:.3f}) < {DRIFT_THRESHOLD} or PSI ({psi:.4f}) > 0.25. Updating baseline.")
                shutil.copy(new_importance_path, baseline_path)
                return True
            return False
        except Exception as e:
            drift_logger.error(f"Gagal saat memeriksa drift untuk {self.pasaran}-{digit}: {e}")
            return False