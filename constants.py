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

# UPDATED: Menghapus 'default' dan memberikan konfigurasi unik untuk setiap pasaran
MARKET_CONFIGS = {
    "sgp": {
        "strategy": {
            "timesteps": 25,
            "min_training_samples": 180
        },
        "feature_engineering": {
            "volatility_window": 30,
            "frequency_window": 90
        }
    },
    "hk": {
        "strategy": {
            "timesteps": 15, # HK lebih dinamis, pola jangka menengah lebih relevan
            "min_training_samples": 180
        },
        "feature_engineering": {
            "volatility_window": 30,
            "frequency_window": 90
        }
    },
    "sydney": {
        "strategy": {
            "timesteps": 15, # Sydney cenderung memiliki tren jangka panjang yang lebih stabil
            "min_training_samples": 180
        },
        "feature_engineering": {
            "volatility_window": 30,
            "frequency_window": 90
        }
    },
    "taiwan": {
        "strategy": {
            "timesteps": 15, # Data Taiwan bisa cukup acak, fokus pada pola jangka pendek-menengah
            "min_training_samples": 150
        },
        "feature_engineering": {
            "volatility_window": 30,
            "frequency_window": 60
        }
    },
    "china": {
        "strategy": {
            "timesteps": 15, # Pola di pasaran China seringkali lebih pendek
            "min_training_samples": 180
        },
        "feature_engineering": {
            "volatility_window": 30,
            "frequency_window": 90
        }
    },
    "magnum": {
        "strategy": {
            "timesteps": 15, # Fokus pada data yang sangat baru karena karakteristiknya
            "min_training_samples": 190 # Mungkin memiliki data historis lebih sedikit
        },
        "feature_engineering": {
            "volatility_window": 30,
            "frequency_window": 90
        }
    }
} # <--- PERBAIKAN: Kurung kurawal penutup yang hilang telah ditambahkan di sini

DRIFT_THRESHOLD = 0.4
ACCURACY_THRESHOLD_FOR_RETRAIN = 0.08
CRITICAL_ACCURACY_THRESHOLD = 0.05 

ADAPTIVE_LEARNING_CONFIG = {
    "USE_RECENCY_WEIGHTING": True,
    "RECENCY_HALF_LIFE_DAYS": 30,
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
    "WINDOW_DAYS": 30
}

HYBRID_SCORING_CONFIG = {
    "ENABLED": True,
    "AI_SCORE_WEIGHT": 0.6, 
    "HISTORICAL_SCORE_WEIGHT": 0.4,
}

TRAINING_PENALTY_CONFIG = {
    "ENABLED": True,
    "LOW_HIT_RATE_PENALTY_FACTOR": 0.7,
    "HIT_RATE_THRESHOLD": 0.08
}