# predictor.py (Final - Dengan Paksaan Refresh Data)

# -*- coding: utf-8 -*-
import os
import shutil
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from constants import (MODELS_DIR, MARKET_CONFIGS, DRIFT_THRESHOLD, 
                     ACCURACY_THRESHOLD_FOR_RETRAIN, ADAPTIVE_LEARNING_CONFIG, ENSEMBLE_CONFIG, CONTINUAL_LEARNING_CONFIG)
from utils import logger, error_handler, drift_logger
from model_config import TRAINING_CONFIG_OPTIONS
from exceptions import TrainingError, PredictionError, DataFetchingError
from data_fetcher import DataFetcher
from evaluation import calculate_brier_score, calculate_ece 
from ensemble_helper import train_ensemble_models, ensemble_predict_proba
from continual_learner import ContinualLearner
from monitoring import ModelPerformanceMonitor

class DataManager:
    # TIDAK ADA PERUBAHAN
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
            # force_refresh akan melewati cache self.df dan memanggil fetch_data
            if self.df is None or force_refresh:
                df_raw = self.fetcher.fetch_data(force_github=force_github)
                df_validated = self._validate_data(df_raw)
                df_sorted = df_validated.sort_values("date").reset_index(drop=True)
                self.df = df_sorted
            return self.df.copy()

class FeatureProcessor:
    # KELAS INI TELAH DIPERBAIKI SECARA FUNDAMENTAL UNTUK MENGHILANGKAN DATA LEAKAGE
    def __init__(self, timesteps, feature_config):
        self.timesteps = timesteps
        self.feature_config = feature_config
        self.digits = ["as", "kop", "kepala", "ekor"]
    
    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        if self.feature_config.get("add_cyclical_features", True):
            df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 6.0)
            df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 6.0)
        return df

    def _add_derived_past_features(self, df: pd.DataFrame) -> pd.DataFrame:
        s_as, s_kop, s_kepala, s_ekor = df['as'].shift(1), df['kop'].shift(1), df['kepala'].shift(1), df['ekor'].shift(1)
        df['prev_as_kop_sum'] = s_as + s_kop
        df['prev_kepala_ekor_diff'] = s_kepala - s_ekor
        df['prev_sum_all'] = s_as + s_kop + s_kepala + s_ekor
        df['prev_range_all'] = pd.concat([s_as, s_kop, s_kepala, s_ekor], axis=1).max(axis=1) - pd.concat([s_as, s_kop, s_kepala, s_ekor], axis=1).min(axis=1)
        for d_str, s in zip(self.digits, [s_as, s_kop, s_kepala, s_ekor]):
             df[f'prev_{d_str}_is_even'] = (s % 2 == 0).fillna(False).astype(int)
        return df

    def _add_rolling_past_features(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.feature_config.get("volatility_window", 10)
        for d in self.digits:
            series = df[d].shift(1)
            df[f'prev_{d}_rolling_mean_{window}'] = series.rolling(window=window).mean()
            df[f'prev_{d}_rolling_std_{window}'] = series.rolling(window=window).std()
        return df

    @error_handler(logger)
    def process_data(self, df_input: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = df_input.copy()
        df["result_cleaned"] = df["result"]
        df = df.dropna(subset=['date']) 
        for i, digit in enumerate(self.digits):
            df[digit] = pd.to_numeric(df["result_cleaned"].str[i], errors='coerce').astype('Int8')
        df = self._add_date_features(df)
        df = self._add_derived_past_features(df)
        if self.feature_config.get("add_interaction_features", True):
             df = self._add_rolling_past_features(df)
        lags = list(range(1, self.timesteps + 1))
        lag_cols_obj = [df[d].shift(i).rename(f"{d}_lag_{i}") for i in lags for d in self.digits]
        df = pd.concat([df] + lag_cols_obj, axis=1)
        lag_column_names = [col.name for col in lag_cols_obj]
        prev_cols = [c for c in df.columns if 'prev_' in c]
        df.dropna(subset=lag_column_names + prev_cols, inplace=True)
        df_final = df.reset_index(drop=True)
        final_features = [col for col in df_final.columns if col not in ['date', 'result', 'result_cleaned'] + self.digits]
        for col in final_features:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0).astype('float32')
        return df_final, final_features

class ModelTrainer:
    # TIDAK ADA PERUBAHAN
    def __init__(self, digits, training_params):
        self.digits = digits
        self.training_params = training_params
    
    @error_handler(logger)
    def train_digit_model(self, X_full, y_full, digit, existing_model=None, feature_subset=None, sample_weights=None):
        try:
            xgb_params = self.training_params.get("xgb_params", {}).get(digit, {})
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_full)
            if len(le.classes_) < 2: 
                logger.warning(f"Hanya ada satu kelas unik untuk digit {digit}. Skipping training.")
                return None, None, None, None
            X_train = X_full[feature_subset] if feature_subset else X_full
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, y_encoded, eval_set=[(X_train, y_encoded)], verbose=False, xgb_model=existing_model, sample_weight=sample_weights)
            importance_scores = model.get_booster().get_score(importance_type='weight')
            feature_importance = {feature: importance_scores.get(feature, 0) for feature in X_train.columns}
            return model, le, feature_importance, y_encoded
        except Exception as e:
            logger.error(f"Error training model untuk digit {digit}: {e}", exc_info=True)
            return None, None, None, None

class ModelPredictor:
    # TIDAK ADA PERUBAHAN
    def __init__(self, pasaran: str):
        self.pasaran = pasaran.lower()
        self.config = self._get_config()
        self.model_dir_base = os.path.join(MODELS_DIR, self.pasaran)
        self.digits = ["as", "kop", "kepala", "ekor"]
        self.data_manager = DataManager(pasaran)
        self.feature_processor = FeatureProcessor(self.config["strategy"]["timesteps"], self.config["feature_engineering"])
        self.models = {d: None for d in self.digits}
        self.rf_models = {d: None for d in self.digits}
        self.lgbm_models = {d: None for d in self.digits}
        self.label_encoders = {d: None for d in self.digits}
        self.feature_names = None
        self.continual_learner = ContinualLearner(self) if CONTINUAL_LEARNING_CONFIG.get("ENABLED") else None
        try:
            self.data_manager.get_data()
        except DataFetchingError as e:
            logger.warning(f"Gagal memuat data awal untuk {self.pasaran}, mungkin perlu training: {e}")
        self.models_ready = self.load_models()

    def _get_config(self) -> Dict[str, Any]:
        default_config = MARKET_CONFIGS["default"].copy()
        market_specific_config = MARKET_CONFIGS.get(self.pasaran, {})
        default_config.update(market_specific_config)
        default_config["training_params"] = TRAINING_CONFIG_OPTIONS.get('OPTIMIZED')
        return default_config

    @error_handler(logger)
    def load_models(self) -> bool:
        if not os.path.exists(self.model_dir_base):
            return False
        features_path = os.path.join(self.model_dir_base, "features.pkl")
        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
        else:
            return False
        xgb_loaded, rf_loaded, lgbm_loaded = 0, 0, 0
        for d in self.digits:
            model_path = os.path.join(self.model_dir_base, f"model_{d}.pkl")
            if os.path.exists(model_path):
                self.models[d] = joblib.load(model_path)
                xgb_loaded += 1
            encoder_path = os.path.join(self.model_dir_base, f"encoder_{d}.pkl")
            if os.path.exists(encoder_path):
                self.label_encoders[d] = joblib.load(encoder_path)
            if ENSEMBLE_CONFIG.get("USE_ENSEMBLE"):
                rf_path, lgbm_path = os.path.join(self.model_dir_base, f"rf_model_{d}.pkl"), os.path.join(self.model_dir_base, f"lgbm_model_{d}.pkl")
                if os.path.exists(rf_path): self.rf_models[d], rf_loaded = joblib.load(rf_path), rf_loaded + 1
                if os.path.exists(lgbm_path): self.lgbm_models[d], lgbm_loaded = joblib.load(lgbm_path), lgbm_loaded + 1
        logger.info(f"Model loaded: XGB({xgb_loaded}/{len(self.digits)}), RF({rf_loaded}/{len(self.digits)}), LGBM({lgbm_loaded}/{len(self.digits)})")
        return xgb_loaded == len(self.digits)

    @error_handler(logger)
    def train_model(self, training_mode: str = 'OPTIMIZED', use_recency_bias: bool = True, custom_data: Optional[pd.DataFrame] = None) -> bool:
        logger.info(f"Memulai training untuk {self.pasaran} mode {training_mode}.")
        self.config["training_params"] = TRAINING_CONFIG_OPTIONS.get(training_mode, TRAINING_CONFIG_OPTIONS['OPTIMIZED'])
        model_trainer = ModelTrainer(self.digits, self.config["training_params"])
        os.makedirs(self.model_dir_base, exist_ok=True)
        df_full = custom_data if custom_data is not None else self.data_manager.get_data(force_refresh=True, force_github=True)
        processed_df, features = self.feature_processor.process_data(df_full)
        if len(processed_df) < self.config["strategy"]["min_training_samples"]:
            raise TrainingError(f"Data tidak cukup: Perlu {self.config['strategy']['min_training_samples']}, tersedia {len(processed_df)}.")
        self.feature_names = features
        joblib.dump(self.feature_names, os.path.join(self.model_dir_base, "features.pkl"))
        sample_weights = None
        if use_recency_bias and ADAPTIVE_LEARNING_CONFIG["USE_RECENCY_WEIGHTING"]:
            half_life = ADAPTIVE_LEARNING_CONFIG["RECENCY_HALF_LIFE_DAYS"]
            decay_rate = np.log(2) / half_life
            days_since_latest = (processed_df['date'].max() - processed_df['date']).dt.days
            weights = np.exp(-decay_rate * days_since_latest)
            sample_weights = pd.Series(weights, index=processed_df.index)
        for digit in self.digits:
            if processed_df[digit].nunique() < 2: continue
            model, le, _, y_encoded = model_trainer.train_digit_model(processed_df[self.feature_names], processed_df[digit], digit, existing_model=self.models.get(digit), sample_weights=sample_weights)
            if model and le:
                joblib.dump(model, os.path.join(self.model_dir_base, f"model_{digit}.pkl"))
                joblib.dump(le, os.path.join(self.model_dir_base, f"encoder_{digit}.pkl"))
                if ENSEMBLE_CONFIG.get("USE_ENSEMBLE"):
                    train_ensemble_models(processed_df[self.feature_names], y_encoded, self.model_dir_base, digit)
        self.models_ready = self.load_models()
        return self.models_ready

    @error_handler(logger)
    def predict_next_day(self, target_date_str: Optional[str] = None, for_evaluation: bool = False) -> Dict[str, Any]:
        if not self.models_ready:
            raise PredictionError("Model tidak siap atau tidak konsisten. Silakan jalankan training.")
        
        target_date = pd.to_datetime(target_date_str) if target_date_str else datetime.now() + timedelta(days=1)
        
        # // PERBAIKAN: Memaksa refresh data setiap kali prediksi untuk memastikan kesegaran
        base_df = self.data_manager.get_data(force_refresh=True)
        historical_data = base_df[base_df['date'] < target_date].copy()

        if len(historical_data) < self.config["strategy"]["timesteps"]:
            raise PredictionError(f"Data historis tidak cukup untuk membuat prediksi.")

        future_row = pd.DataFrame([{'date': target_date, 'result': np.nan}])
        data_with_future = pd.concat([historical_data, future_row], ignore_index=True)

        processed_df, _ = self.feature_processor.process_data(data_with_future)

        if processed_df.empty:
            raise PredictionError("Tidak ada data yang tersisa setelah rekayasa fitur.")

        latest_features_unaligned = processed_df.iloc[-1:]
        latest_features = latest_features_unaligned.reindex(columns=self.feature_names, fill_value=0)
        
        predictions = {}
        all_probas_for_eval = {}
        all_candidates_with_probas = []
        for d in self.digits:
            encoder = self.label_encoders[d]
            if not encoder: raise PredictionError(f"Encoder untuk digit '{d}' tidak tersedia.")
            current_models = {'xgb': self.models.get(d)}
            if ENSEMBLE_CONFIG.get("USE_ENSEMBLE"):
                current_models['rf'] = self.rf_models.get(d)
                current_models['lgbm'] = self.lgbm_models.get(d)
            probabilities = ensemble_predict_proba(current_models, latest_features)[0]
            all_probas_for_eval[d] = probabilities
            top_two_indices = np.argsort(probabilities)[-2:][::-1]
            top_two_digits = encoder.inverse_transform(top_two_indices)
            predictions[d] = [str(int(digit)) for digit in top_two_digits]
            top_two_probas = probabilities[top_two_indices]
            for digit, proba in zip(top_two_digits, top_two_probas):
                all_candidates_with_probas.append((str(int(digit)), proba))
        
        kandidat_as, kandidat_kop, kandidat_kepala, kandidat_ekor = predictions['as'], predictions['kop'], predictions['kepala'], predictions['ekor']
        combined_candidates = kandidat_as + kandidat_kop + kandidat_kepala + kandidat_ekor
        angka_main_set = sorted(list(set(combined_candidates)))
        best_cb = ""
        if all_candidates_with_probas: best_cb = max(all_candidates_with_probas, key=lambda item: item[1])[0]

        result = {
            "prediction_date": target_date.strftime("%Y-%m-%d"),
            "final_4d_prediction": f"{kandidat_as[0]}{kandidat_kop[0]}{kandidat_kepala[0]}{kandidat_ekor[0]}",
            "kandidat_as": ", ".join(kandidat_as), "kandidat_kop": ", ".join(kandidat_kop),
            "kandidat_kepala": ", ".join(kandidat_kepala), "kandidat_ekor": ", ".join(kandidat_ekor),
            "angka_main": ", ".join(angka_main_set[:4]), "colok_bebas": best_cb
        }
        if for_evaluation:
            result["probabilities"], result["label_encoders"] = all_probas_for_eval, self.label_encoders
        return result

    @error_handler(logger)
    def evaluate_performance(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        df = self.data_manager.get_data(force_github=True)
        eval_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        if eval_df.empty: return {"summary": {"error": "Tidak ada data pada periode yang diminta."}, "results": []}
        results_list, prob_metrics = [], {d: {'y_true': [], 'y_prob': [], 'y_true_label': [], 'y_prob_max': []} for d in self.digits}
        monitor = ModelPerformanceMonitor()
        for _, row in eval_df.iterrows():
            try:
                pred_result = self.predict_next_day(row['date'].strftime('%Y-%m-%d'), for_evaluation=True)
                actual_result = row['result']
                if pred_result and actual_result:
                    for i, d in enumerate(self.digits):
                        actual_digit, le = int(actual_result[i]), pred_result['label_encoders'][d]
                        if actual_digit in le.classes_:
                            y_true_one_hot = np.zeros(len(le.classes_))
                            class_index = np.where(le.classes_ == actual_digit)[0][0]
                            y_true_one_hot[class_index] = 1
                            prob_metrics[d]['y_true'].append(y_true_one_hot)
                            prob_metrics[d]['y_prob'].append(pred_result['probabilities'][d])
                            prob_metrics[d]['y_true_label'].append(class_index)
                            prob_metrics[d]['y_prob_max'].append(pred_result['probabilities'][d].max())
                    del pred_result['probabilities'], pred_result['label_encoders']
                    summary_metrics = {'cb_accuracy': 100.0 if any(d in pred_result['colok_bebas'] for d in actual_result) else 0.0}
                    monitor.track_performance(self.pasaran, row['date'], summary_metrics)
                    results_list.append({ "date": row['date'].strftime('%Y-%m-%d'), "actual": actual_result, **pred_result })
            except PredictionError as e:
                logger.warning(f"Skipping evaluasi untuk {row['date'].strftime('%Y-%m-%d')}: {e}")
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
            "cb_accuracy": eval_summary_df.apply(lambda r: any(d in r['colok_bebas'] for d in r['actual']), axis=1).mean(),
            "avg_brier_score": -1, "avg_calibration_error": -1,
            "retraining_recommended": False, "retraining_reason": "N/A"
        }
        brier_scores, calib_errors = [], []
        for d in self.digits:
            if prob_metrics[d]['y_true']:
                brier_scores.append(calculate_brier_score(np.array(prob_metrics[d]['y_true']), np.array(prob_metrics[d]['y_prob'])))
                calib_errors.append(calculate_ece(np.array(prob_metrics[d]['y_true_label']), np.array(prob_metrics[d]['y_prob_max']), np.array(prob_metrics[d]['y_prob'])))
        summary.update({"avg_brier_score": np.mean(brier_scores) if brier_scores else -1, "avg_calibration_error": np.mean(calib_errors) if calib_errors else -1})
        drift_detected = any(self._check_for_drift(d) for d in self.digits)
        if (summary["kepala_accuracy"] < ACCURACY_THRESHOLD_FOR_RETRAIN or summary["ekor_accuracy"] < ACCURACY_THRESHOLD_FOR_RETRAIN):
            summary.update({"retraining_recommended": True, "retraining_reason": f"Akurasi Kepala/Ekor ({summary['kepala_accuracy']:.1%}/{summary['ekor_accuracy']:.1%}) di bawah ambang batas."})
        if self.continual_learner and CONTINUAL_LEARNING_CONFIG.get("AUTO_TRIGGER_ENABLED"):
            self.continual_learner.check_and_retrain(summary, drift_detected)
        return {"summary": summary, "results": results_list}

    @error_handler(drift_logger)
    def _check_for_drift(self, digit: str) -> bool:
        # TIDAK ADA PERUBAHAN
        new_importance_path, baseline_path = os.path.join(self.model_dir_base, f"feature_importance_{digit}.csv"), os.path.join(self.model_dir_base, f"feature_importance_{digit}_baseline.csv")
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
            baseline_dist[baseline_dist == 0], new_dist[new_dist == 0] = 1e-6, 1e-6
            psi = np.sum((new_dist - baseline_dist) * np.log(new_dist / baseline_dist))
            if jaccard_sim < DRIFT_THRESHOLD or psi > 0.25:
                logger.warning(f"DRIFT DETECTED for {self.pasaran}-{digit}! Jaccard ({jaccard_sim:.3f}) < {DRIFT_THRESHOLD} or PSI ({psi:.4f}) > 0.25. Updating baseline.")
                shutil.copy(new_importance_path, baseline_path)
                return True
            return False
        except Exception as e:
            drift_logger.error(f"Gagal saat memeriksa drift untuk {self.pasaran}-{digit}: {e}")
            return False