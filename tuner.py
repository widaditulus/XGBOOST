# tuner.py (Final - KeyError Diperbaiki)

# -*- coding: utf-8 -*-
import os
import json
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from predictor import FeatureProcessor, DataManager
from utils import logger, error_handler
from constants import MODELS_DIR, MARKET_CONFIGS

TUNING_CONFIG = {
    "N_TRIALS_4D": 75,
    "N_TRIALS_CB": 25,
    "N_SPLITS": 3,
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
        df_full = data_manager.get_data(force_refresh=True, force_github=True)
        
        config = MARKET_CONFIGS[self.pasaran]
        fp = FeatureProcessor(config["strategy"]["timesteps"], config["feature_engineering"])
        training_df = fp.fit_transform(df_full)
        
        # --- UPDATED: Reset index untuk mencegah KeyError saat cross-validation ---
        training_df.reset_index(drop=True, inplace=True)
        
        self.X = training_df[fp.feature_names]

        if self.mode == '4D':
            self.y = training_df[self.digit]
            self.le = LabelEncoder()
            self.y_encoded = self.le.fit_transform(self.y)
        elif self.mode == 'CB':
            self.y_encoded = training_df[f'cb_target_{self.digit}']
        else:
            raise ValueError("Mode tuning tidak valid. Pilih '4D' atau 'CB'.")

    def _objective(self, trial: optuna.Trial) -> float:
        if self.mode == '4D':
            params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'n_estimators': trial.suggest_int('n_estimators', 400, 2000, step=100),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0),
            }
            cv = StratifiedKFold(n_splits=TUNING_CONFIG["N_SPLITS"], shuffle=True, random_state=42)
        else:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 4.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, (self.y_encoded == 0).sum() / (self.y_encoded == 1).sum() if (self.y_encoded == 1).sum() > 0 else 1)
            }
            cv = KFold(n_splits=TUNING_CONFIG["N_SPLITS"], shuffle=True, random_state=42)

        params.update({
            'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'early_stopping_rounds': 30 
        })

        logloss_scores = []
        # Gunakan .values untuk mengubah ke NumPy array dan menghindari masalah index Pandas
        y_values = self.y_encoded if isinstance(self.y_encoded, np.ndarray) else self.y_encoded.values

        for train_idx, val_idx in cv.split(self.X, y_values):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = y_values[train_idx], y_values[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
            
            preds = model.predict_proba(X_val)
            loss = log_loss(y_val, preds)
            logloss_scores.append(loss)

        return np.mean(logloss_scores)

    @error_handler(logger)
    def run_tuning(self) -> dict:
        storage_name = f"sqlite:///{self.pasaran}_{self.digit}_{self.mode}_tuning.db"
        self.study = optuna.create_study(
            direction='minimize',
            study_name=f"{self.pasaran}_{self.digit}_{self.mode}",
            storage=storage_name,
            load_if_exists=True
        )
        
        n_trials = TUNING_CONFIG['N_TRIALS_4D'] if self.mode == '4D' else TUNING_CONFIG['N_TRIALS_CB']
        logger.info(f"TUNER: Memulai optimasi {self.mode} untuk {self.pasaran}-{self.digit}. Trials: {n_trials}")
        
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=TUNING_CONFIG["TIMEOUT"],
            callbacks=[self.log_progress]
        )

        self.best_params_ = self.study.best_params
        logger.info(f"TUNER: Optimasi selesai. Best score (logloss): {self.study.best_value:.4f}")
        self.save_best_params()

        try:
            db_file = f"{self.pasaran}_{self.digit}_{self.mode}_tuning.db"
            if os.path.exists(db_file):
                os.remove(db_file)
                logger.info(f"TUNER: File database sementara '{db_file}' telah dihapus.")
        except OSError as e:
            logger.warning(f"TUNER: Gagal menghapus file DB: {e}")

        return self.best_params_

    def save_best_params(self):
        if not self.best_params_: return

        final_params = self.best_params_.copy()
        if self.mode == '4D':
            final_params['objective'] = 'multi:softprob'
            final_params['eval_metric'] = 'mlogloss'
            file_name = f"best_params_{self.digit}.json"
        else:
            final_params['objective'] = 'binary:logistic'
            final_params['eval_metric'] = 'logloss'
            file_name = f"best_params_cb_{self.digit}.json"
        
        if 'early_stopping_rounds' not in final_params:
             final_params['early_stopping_rounds'] = 50
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