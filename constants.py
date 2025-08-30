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
    "sgp": {
        "strategy": {
            "timesteps": 30,
            "min_training_samples": 150
        },
        "feature_engineering": {
            "volatility_window": 15,
            "frequency_window": 120
        }
    }
}

DRIFT_THRESHOLD = 0.4
ACCURACY_THRESHOLD_FOR_RETRAIN = 0.08

ADAPTIVE_LEARNING_CONFIG = {
    "USE_RECENCY_WEIGHTING": True,
    "RECENCY_HALF_LIFE_DAYS": 45,
}

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

CONTINUAL_LEARNING_CONFIG = {
    "ENABLED": True,
    "AUTO_TRIGGER_ENABLED": True,
    "WINDOW_DAYS": 180
}

# --- UPDATED: MENGGANTI DYNAMIC_WEIGHTING_CONFIG DENGAN HYBRID_SCORING_CONFIG ---
# Konfigurasi baru untuk sistem skoring hibrida yang lebih kuat dan terkontrol.
HYBRID_SCORING_CONFIG = {
    "ENABLED": True,
    # Bobot untuk skor dari prediksi AI.
    # Nilai antara 0.0 dan 1.0.
    "AI_SCORE_WEIGHT": 0.8, # Artinya 80% pengaruh dari AI

    # Bobot untuk skor dari frekuensi historis.
    # Nilai antara 0.0 dan 1.0.
    "HISTORICAL_SCORE_WEIGHT": 0.2, # Artinya 20% pengaruh dari data historis

    # Pastikan AI_SCORE_WEIGHT + HISTORICAL_SCORE_WEIGHT = 1.0
}


TRAINING_PENALTY_CONFIG = {
    "ENABLED": True,
    "LOW_HIT_RATE_PENALTY_FACTOR": 0.7,
    "HIT_RATE_THRESHOLD": 0.08
}