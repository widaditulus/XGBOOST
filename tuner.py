# tuner.py (Final - Anti Data Leakage Absolut)
# BEJO
# -*- coding: utf-8 -*-
import os
import json
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
# PERBAIKAN: Impor gc untuk garbage collection dan time untuk jeda
import gc
import time

# Mengimpor FeatureProcessor dan DataManager yang benar dari predictor
from predictor import FeatureProcessor, DataManager
from utils import logger, error_handler
from constants import MODELS_DIR, MARKET_CONFIGS

TUNING_CONFIG = {
    "N_TRIALS_4D": 75,
    "N_TRIALS_CB": 25,
    "N_SPLITS": 5,
    "TIMEOUT": 4000
}

class HyperparameterTuner:
    def __init__(self, pasaran: str, digit: str, mode: str = '4D'):
        self.pasaran = pasaran
        self.digit = digit
        self.mode = mode
        self.study = None
        self.best_params_ = None
        logger.info(f"TUNER: Mode '{self.mode}' untuk pasaran '{self.pasaran}' digit '{self.digit}'.")
        data_manager = DataManager(self.pasaran)
        # Mengambil data mentah. Ini akan digunakan untuk split di _objective
        self.df_raw = data_manager.get_data(force_refresh=True, force_github=True)
        self.config = MARKET_CONFIGS[self.pasaran]
        
    def _objective(self, trial: optuna.Trial) -> float:
        if self.mode == '4D':
            params = {
                'objective': 'multi:softprob', 'eval_metric': 'mlogloss',
                'n_estimators': trial.suggest_int('n_estimators', 400, 2000, step=100),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0),
            }
        else:
            params = {
                'objective': 'binary:logistic', 'eval_metric': 'logloss',
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 4.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)
            }
        params.update({
            'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'early_stopping_rounds': 30 
        })
        cv = TimeSeriesSplit(n_splits=TUNING_CONFIG["N_SPLITS"])
        logloss_scores = []
        
        # PERBAIKAN: Logika rekayasa fitur dipindahkan ke dalam loop TimeSeriesSplit
        # untuk mencegah data leakage sekecil apa pun. Fitur (misal: rolling mean)
        # untuk setiap fold kini hanya dihitung berdasarkan data yang tersedia hingga saat itu.
        for train_idx, val_idx in cv.split(self.df_raw):
            # 1. Ambil data mentah HANYA untuk fold saat ini (train + validasi)
            # Ini memastikan tidak ada data dari masa depan (fold berikutnya) yang ikut terproses.
            current_fold_raw_df = self.df_raw.iloc[:val_idx[-1] + 1].copy()

            # 2. Lakukan rekayasa fitur pada data fold saat ini.
            # Instance FeatureProcessor dibuat baru setiap iterasi (stateless).
            fp = FeatureProcessor(self.config["strategy"]["timesteps"], self.config["feature_engineering"])
            processed_fold_df = fp.fit_transform(current_fold_raw_df)
            
            # 3. Dapatkan kembali index yang sesuai dari dataframe yang sudah diproses.
            # Hal ini diperlukan karena `fit_transform` dapat menghapus baris awal (misal: karena data tidak cukup untuk rolling window).
            train_rows = processed_fold_df.index.intersection(train_idx)
            val_rows = processed_fold_df.index.intersection(val_idx)

            if len(train_rows) == 0 or len(val_rows) == 0:
                continue

            feature_names = fp.feature_names
            if self.mode == '4D':
                target_col = self.digit
                le = LabelEncoder().fit(processed_fold_df[target_col])
                all_labels = le.classes_
            else:
                target_col = f'cb_target_{self.digit}'
                all_labels = [0, 1]

            X_train = processed_fold_df.loc[train_rows, feature_names]
            y_train = processed_fold_df.loc[train_rows, target_col]
            X_val = processed_fold_df.loc[val_rows, feature_names]
            y_val = processed_fold_df.loc[val_rows, target_col]
            
            if self.mode == '4D':
                y_train_encoded = le.transform(y_train)
                y_val_encoded = le.transform(y_val)
            else:
                y_train_encoded = y_train
                y_val_encoded = y_val

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train_encoded, eval_set=[(X_val, y_val_encoded)], verbose=False)
            
            preds = model.predict_proba(X_val)
            loss = log_loss(y_val_encoded, preds, labels=range(len(all_labels)))
            logloss_scores.append(loss)
            
        if not logloss_scores:
            logger.warning(f"TUNER: Tidak ada data validasi yang cukup untuk {self.mode}-{self.digit}. Kembali ke infinity.")
            return float('inf')
        return np.mean(logloss_scores)

    @error_handler(logger)
    def run_tuning(self) -> dict:
        db_file_name = f"{self.pasaran}_{self.digit}_{self.mode}_tuning.db"
        storage_name = f"sqlite:///{db_file_name}"
        
        study = optuna.create_study(direction='minimize', study_name=f"{self.pasaran}_{self.digit}_{self.mode}", storage=storage_name, load_if_exists=True)
        n_trials = TUNING_CONFIG['N_TRIALS_4D'] if self.mode == '4D' else TUNING_CONFIG['N_TRIALS_CB']
        
        logger.info(f"TUNER: Memulai optimasi {self.mode} untuk {self.pasaran}-{self.digit}. Trials: {n_trials}")
        study.optimize(self._objective, n_trials=n_trials, timeout=TUNING_CONFIG["TIMEOUT"], callbacks=[self.log_progress])
        
        self.best_params_ = study.best_params
        logger.info(f"TUNER: Optimasi selesai. Best score (logloss): {study.best_value:.4f}")
        self.save_best_params()
        
        try:
            # PERBAIKAN FINAL: Hancurkan objek studi dan panggil garbage collector
            # untuk memaksa pelepasan handle file database.
            del study
            gc.collect()
            
            # Beri jeda 1 detik untuk memastikan OS Windows sempat melepaskan file lock.
            time.sleep(1)

            if os.path.exists(db_file_name):
                 os.remove(db_file_name)
                 logger.info(f"TUNER: File DB sementara '{db_file_name}' berhasil dihapus.")
        except OSError as e:
            logger.warning(f"TUNER: Gagal menghapus file DB sementara '{db_file_name}': {e}")
            
        return self.best_params_

    def save_best_params(self):
        if not self.best_params_: return
        final_params = self.best_params_.copy()
        if self.mode == '4D':
            final_params.update({'objective': 'multi:softprob', 'eval_metric': 'mlogloss'})
            file_name = f"best_params_{self.digit}.json"
        else:
            final_params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss'})
            file_name = f"best_params_cb_{self.digit}.json"
        if 'early_stopping_rounds' not in final_params: final_params['early_stopping_rounds'] = 50
        final_params['random_state'] = 42
        model_dir = os.path.join(MODELS_DIR, self.pasaran)
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump(final_params, f, indent=4)
        logger.info(f"TUNER: Parameter terbaik untuk {self.mode}-{self.digit} disimpan di {file_path}")
        
    def log_progress(self, study, trial):
        n_trials = TUNING_CONFIG['N_TRIALS_4D'] if self.mode == '4D' else TUNING_CONFIG['N_TRIALS_CB']
        best_value = study.best_value if study.best_trial else float('inf')
        logger.info(f"TUNER [{self.pasaran}-{self.digit}-{self.mode}]: Trial {trial.number}/{n_trials} | Logloss: {trial.value:.4f} | Best: {best_value:.4f}")