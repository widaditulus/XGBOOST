# model_config.py
# -*- coding: utf-8 -*-

# --- PARAMETER UNTUK CONSTRUCTOR XGBOOST ---
# Setiap set parameter sekarang berisi 'early_stopping_rounds' secara langsung.
XGB_PARAMS_OPTIMIZED = {
    "as":     {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "gamma": 0.1, "reg_alpha": 0.1, "random_state": 42, "early_stopping_rounds": 30},
    "kop":    {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "gamma": 0.1, "reg_alpha": 0.1, "random_state": 42, "early_stopping_rounds": 30},
    "kepala": {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "gamma": 0.1, "reg_alpha": 0.1, "random_state": 42, "early_stopping_rounds": 30},
    "ekor":   {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "gamma": 0.1, "reg_alpha": 0.1, "random_state": 42, "early_stopping_rounds": 30}
}

XGB_PARAMS_AKURAT = {
    "as":     {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 1500, "learning_rate": 0.06, "max_depth": 7, "subsample": 0.6, "colsample_bytree": 0.7, "gamma": 0.1, "reg_alpha": 0.6, "reg_lambda": 1.8, "random_state": 42, "early_stopping_rounds": 50},
    "kop":    {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 1200, "learning_rate": 0.02, "max_depth": 7, "subsample": 0.6, "colsample_bytree": 0.5, "gamma": 0.4, "reg_alpha": 0.7, "reg_lambda": 1.9, "random_state": 42, "early_stopping_rounds": 50},
    "kepala": {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 1000, "learning_rate": 0.01, "max_depth": 6, "subsample": 0.6, "colsample_bytree": 0.6, "gamma": 0.1, "reg_alpha": 0.2, "reg_lambda": 1.8, "random_state": 42, "early_stopping_rounds": 50},
    "ekor":   {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 1000, "learning_rate": 0.02, "max_depth": 7, "subsample": 0.6, "colsample_bytree": 0.5, "gamma": 0.4, "reg_alpha": 0.8, "reg_lambda": 1.8, "random_state": 42, "early_stopping_rounds": 50}
}

XGB_PARAMS_CEPAT = {
    "as":     {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "early_stopping_rounds": 20},
    "kop":    {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "early_stopping_rounds": 20},
    "kepala": {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "early_stopping_rounds": 20},
    "ekor":   {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "early_stopping_rounds": 20}
}

# UPDATED: Menambahkan set parameter baru untuk analisis yang lebih mendalam
# PERBAIKAN: Parameter untuk kepala & ekor diubah untuk menangani kasus sulit
XGB_PARAMS_DEEP_ANALYSIS = {
    "as":     {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 7, "subsample": 0.7, "colsample_bytree": 0.7, "gamma": 0.2, "reg_alpha": 0.1, "random_state": 42, "early_stopping_rounds": 50},
    "kop":    {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 7, "subsample": 0.7, "colsample_bytree": 0.7, "gamma": 0.2, "reg_alpha": 0.1, "random_state": 42, "early_stopping_rounds": 50},
    "kepala": {"objective": "multi:softprob", "eval_metric": "mlogloss",
               "n_estimators": 2500,      # Beri lebih banyak kesempatan belajar
               "learning_rate": 0.008,   # Langkah belajar lebih kecil/hati-hati
               "max_depth": 8,           # Izinkan model melihat interaksi fitur yang lebih kompleks
               "subsample": 0.65, "colsample_bytree": 0.65, "gamma": 0.25,
               "reg_alpha": 0.15, "random_state": 42,
               "early_stopping_rounds": 60}, # Perpanjang kesabaran
    "ekor":   {"objective": "multi:softprob", "eval_metric": "mlogloss",
               "n_estimators": 2500,      # Beri lebih banyak kesempatan belajar
               "learning_rate": 0.008,   # Langkah belajar lebih kecil/hati-hati
               "max_depth": 8,           # Izinkan model melihat interaksi fitur yang lebih kompleks
               "subsample": 0.65, "colsample_bytree": 0.65, "gamma": 0.25,
               "reg_alpha": 0.15, "random_state": 42,
               "early_stopping_rounds": 60} # Perpanjang kesabaran
}


# --- PENGGABUNGAN KONFIGURASI ---
TRAINING_CONFIG_OPTIONS = {
    'QUICK': {"xgb_params": XGB_PARAMS_CEPAT},
    'COMPREHENSIVE': {"xgb_params": XGB_PARAMS_AKURAT},
    'OPTIMIZED': {"xgb_params": XGB_PARAMS_OPTIMIZED},
    # UPDATED: Menambahkan mode training baru
    'DEEP_ANALYSIS': {"xgb_params": XGB_PARAMS_DEEP_ANALYSIS}
}