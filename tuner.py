# tuner.py

# -*- coding: utf-8 -*-
import os
import json
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from predictor import FeatureProcessor, DataManager
from utils import logger, error_handler
from constants import MODELS_DIR

# Konfigurasi untuk proses tuning
TUNING_CONFIG = {
    "N_TRIALS": 50,  # Jumlah kombinasi parameter yang akan dicoba
    "N_SPLITS": 3,   # Jumlah lipatan untuk cross-validation
    "TIMEOUT": 3600  # Batas waktu maksimal dalam detik (1 jam)
}

class HyperparameterTuner:
    """
    Kelas untuk mencari hiperparameter terbaik untuk model XGBoost
    menggunakan Optuna dan Cross-Validation.
    """
    def __init__(self, pasaran: str, digit: str):
        self.pasaran = pasaran
        self.digit = digit
        self.study = None
        self.best_params_ = None

        # Siapkan data sekali saja
        logger.info(f"TUNER: Mempersiapkan data untuk {pasaran}...")
        data_manager = DataManager(self.pasaran)
        df_full = data_manager.get_data(force_refresh=True, force_github=True)
        
        # Mengambil config timesteps dari Market Config (meskipun default)
        from constants import MARKET_CONFIGS
        config = MARKET_CONFIGS.get(self.pasaran, MARKET_CONFIGS["default"])
        timesteps = config["strategy"]["timesteps"]
        feature_config = config["feature_engineering"]

        fp = FeatureProcessor(timesteps, feature_config)
        
        training_df = fp.fit_transform(df_full)
        
        self.X = training_df[fp.feature_names]
        self.y = training_df[self.digit]
        
        # Encode target variable
        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(self.y)

    def _objective(self, trial: optuna.Trial) -> float:
        """Fungsi objektif yang akan diminimalkan oleh Optuna."""
        # Definisikan ruang pencarian parameter
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 400, 2000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0),
            # UPDATED: 'early_stopping_rounds' dipindahkan ke sini
            'early_stopping_rounds': 30
        }

        # Gunakan Stratified K-Fold untuk menjaga distribusi kelas
        skf = StratifiedKFold(n_splits=TUNING_CONFIG["N_SPLITS"], shuffle=True, random_state=42)
        logloss_scores = []

        for train_idx, val_idx in skf.split(self.X, self.y_encoded):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y_encoded[train_idx], self.y_encoded[val_idx]

            model = xgb.XGBClassifier(**params)
            
            # UPDATED: 'early_stopping_rounds' dihapus dari .fit()
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
            
            preds = model.predict_proba(X_val)
            loss = log_loss(y_val, preds)
            logloss_scores.append(loss)

        avg_logloss = np.mean(logloss_scores)
        return avg_logloss

    @error_handler(logger)
    def run_tuning(self) -> dict:
        """Memulai proses optimasi."""
        storage_name = f"sqlite:///{self.pasaran}_{self.digit}_tuning.db"
        self.study = optuna.create_study(
            direction='minimize',
            study_name=f"{self.pasaran}_{self.digit}",
            storage=storage_name,
            load_if_exists=True
        )
        
        logger.info(f"TUNER: Memulai optimasi untuk {self.pasaran}-{self.digit}. Total trials: {TUNING_CONFIG['N_TRIALS']}")
        
        self.study.optimize(
            self._objective,
            n_trials=TUNING_CONFIG["N_TRIALS"],
            timeout=TUNING_CONFIG["TIMEOUT"],
            callbacks=[self.log_progress]
        )

        self.best_params_ = self.study.best_params
        logger.info(f"TUNER: Optimasi selesai untuk {self.pasaran}-{self.digit}. Best score (logloss): {self.study.best_value}")
        logger.info(f"TUNER: Best params: {self.best_params_}")
        
        self.save_best_params()

        try:
            db_file = f"{self.pasaran}_{self.digit}_tuning.db"
            if os.path.exists(db_file):
                os.remove(db_file)
                logger.info(f"TUNER: File database sementara '{db_file}' telah dihapus.")
        except OSError as e:
            logger.warning(f"TUNER: Gagal menghapus file DB: {e}")

        return self.best_params_

    def save_best_params(self):
        """Menyimpan parameter terbaik ke file JSON."""
        if not self.best_params_:
            logger.warning("TUNER: Tidak ada parameter terbaik untuk disimpan.")
            return

        final_params = self.best_params_.copy()
        final_params['objective'] = 'multi:softprob'
        final_params['eval_metric'] = 'mlogloss'
        final_params['random_state'] = 42
        final_params['early_stopping_rounds'] = 50 

        model_dir = os.path.join(MODELS_DIR, self.pasaran)
        os.makedirs(model_dir, exist_ok=True)
        
        file_path = os.path.join(model_dir, f"best_params_{self.digit}.json")
        with open(file_path, 'w') as f:
            json.dump(final_params, f, indent=4)
        logger.info(f"TUNER: Parameter terbaik untuk {self.digit} disimpan di {file_path}")
        
    def log_progress(self, study, trial):
        """Callback untuk mencatat progress setiap trial."""
        best_value = study.best_value if study.best_trial else float('inf')
        logger.info(f"TUNER [{self.pasaran}-{self.digit}]: Trial {trial.number}/{TUNING_CONFIG['N_TRIALS']} | Logloss: {trial.value:.4f} | Best: {best_value:.4f}")