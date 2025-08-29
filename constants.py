# constants.py

# -*- coding: utf-8 -*-
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "4d_data.db")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

PASARAN_LIST = ["sgp", "hk", "sydney", "taiwan", "china", "magnum"]

PASARAN_DISPLAY_MAPPING = {
    "sgp": "Singapore (SGP)",
    "hk": "Hongkong (HK)",
    "sydney": "Sydney",
    "taiwan": "Taiwan",
    "china": "China",
    "magnum": "Magnum",
}

MARKET_CONFIGS = {
    "default": {
        "strategy": {
            "timesteps": 20,
            "min_training_samples": 100,
        },
        "feature_engineering": {
            "add_cyclical_features": True, "add_frequency_features": True,
            "add_advanced_pattern_features": True, "add_interaction_features": True,
            "volatility_window": 10, "frequency_window": 30
        }
    },
    # --- BLOK KHUSUS UNTUK SGP (DIPERBAIKI) ---
    "sgp": {
        "strategy": {
            # SGP adalah pasaran yang stabil, kita buat model melihat lebih jauh ke belakang
            "timesteps": 30,
            # Tetap sertakan kunci ini agar tidak hilang saat digabungkan
            "min_training_samples": 150
        },
        "feature_engineering": {
            # Karena SGP tidak buka setiap hari, jendela analisis yang lebih lebar mungkin lebih baik
            "volatility_window": 15,
            "frequency_window": 120
        }
    }
}

# --- PARAMETER SISTEM ADAPTIF ---
DRIFT_THRESHOLD = 0.4
ACCURACY_THRESHOLD_FOR_RETRAIN = 0.10

ADAPTIVE_LEARNING_CONFIG = {
    "USE_RECENCY_WEIGHTING": True,
    "RECENCY_HALF_LIFE_DAYS": 45,
}

# --- KONFIGURASI ENSEMBLE ---
ENSEMBLE_CONFIG = {
    "USE_ENSEMBLE": True,
    "rf_params": {
        "n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5,
        "random_state": 42, "n_jobs": -1
    },
    "lgbm_params": {
        "objective": "multiclass", "metric": "multi_logloss", "n_estimators": 200,
        "learning_rate": 0.05, "num_leaves": 31, "random_state": 42,
        "n_jobs": -1, "colsample_bytree": 0.8, "subsample": 0.8, "verbosity": -1
    }
}

# --- KONFIGURASI CONTINUAL LEARNING ---
CONTINUAL_LEARNING_CONFIG = {
    "ENABLED": True,
    # Aktifkan pemicu retrain otomatis jika akurasi turun atau drift terdeteksi
    "AUTO_TRIGGER_ENABLED": True,
    # Retrain hanya pada data 6 bulan (180 hari) terakhir
    "WINDOW_DAYS": 180
}