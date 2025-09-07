# model_config.py (Final - Lengkap dan Fungsional)
# -*- coding: utf-8 -*-

# --- PERBAIKAN: Menambahkan parameter akselerasi ---
# 'tree_method': 'gpu_hist' -> Menggunakan GPU untuk training (jika tersedia, jika tidak XGBoost akan fallback ke CPU).
#                             Memberikan percepatan paling signifikan.
# 'n_jobs': -1 -> Menggunakan semua core CPU yang tersedia untuk pre-processing. Berguna saat GPU tidak tersedia.
ACCELERATION_PARAMS = {"tree_method": "gpu_hist", "n_jobs": -1}

# --- UPDATED: Menambahkan parameter khusus untuk Model Colok Bebas ---
# Parameter ini dioptimalkan untuk klasifikasi biner (muncul/tidak muncul)
# dan dibuat lebih 'ringan' karena kita akan melatih 10 model sekaligus.
XGB_PARAMS_CB = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 250,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "early_stopping_rounds": 20,
    **ACCELERATION_PARAMS 
}

# --- PARAMETER UNTUK CONSTRUCTOR XGBOOST ---
# Setiap set parameter sekarang berisi 'early_stopping_rounds' secara langsung.
XGB_PARAMS_OPTIMIZED = {
    "as":     {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "gamma": 0.1, "reg_alpha": 0.1, "random_state": 42, "early_stopping_rounds": 30, **ACCELERATION_PARAMS},
    "kop":    {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "gamma": 0.1, "reg_alpha": 0.1, "random_state": 42, "early_stopping_rounds": 30, **ACCELERATION_PARAMS},
    "kepala": {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "gamma": 0.1, "reg_alpha": 0.1, "random_state": 42, "early_stopping_rounds": 30, **ACCELERATION_PARAMS},
    "ekor":   {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "gamma": 0.1, "reg_alpha": 0.1, "random_state": 42, "early_stopping_rounds": 30, **ACCELERATION_PARAMS}
}

XGB_PARAMS_AKURAT = {
    "as":     {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 1500, "learning_rate": 0.06, "max_depth": 7, "subsample": 0.6, "colsample_bytree": 0.7, "gamma": 0.1, "reg_alpha": 0.6, "reg_lambda": 1.8, "random_state": 42, "early_stopping_rounds": 50, **ACCELERATION_PARAMS},
    "kop":    {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 1200, "learning_rate": 0.02, "max_depth": 7, "subsample": 0.6, "colsample_bytree": 0.5, "gamma": 0.4, "reg_alpha": 0.7, "reg_lambda": 1.9, "random_state": 42, "early_stopping_rounds": 50, **ACCELERATION_PARAMS},
    "kepala": {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 1000, "learning_rate": 0.01, "max_depth": 6, "subsample": 0.6, "colsample_bytree": 0.6, "gamma": 0.1, "reg_alpha": 0.2, "reg_lambda": 1.8, "random_state": 42, "early_stopping_rounds": 50, **ACCELERATION_PARAMS},
    "ekor":   {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 1000, "learning_rate": 0.02, "max_depth": 7, "subsample": 0.6, "colsample_bytree": 0.5, "gamma": 0.4, "reg_alpha": 0.8, "reg_lambda": 1.8, "random_state": 42, "early_stopping_rounds": 50, **ACCELERATION_PARAMS}
}

XGB_PARAMS_CEPAT = {
    "as":     {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "early_stopping_rounds": 20, **ACCELERATION_PARAMS},
    "kop":    {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "early_stopping_rounds": 20, **ACCELERATION_PARAMS},
    "kepala": {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "early_stopping_rounds": 20, **ACCELERATION_PARAMS},
    "ekor":   {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "early_stopping_rounds": 20, **ACCELERATION_PARAMS}
}

XGB_PARAMS_DEEP_ANALYSIS = {
    "as":     {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 2000, "learning_rate": 0.03, "max_depth": 7, "subsample": 0.6, "colsample_bytree": 0.6, "gamma": 0.2, "reg_alpha": 0.2, "random_state": 42, "early_stopping_rounds": 50, **ACCELERATION_PARAMS},
    "kop":    {"objective": "multi:softprob", "eval_metric": "mlogloss", "n_estimators": 2000, "learning_rate": 0.03, "max_depth": 7, "subsample": 0.6, "colsample_bytree": 0.6, "gamma": 0.2, "reg_alpha": 0.2, "random_state": 42, "early_stopping_rounds": 50, **ACCELERATION_PARAMS},
    "kepala": {"objective": "multi:softprob", "eval_metric": "mlogloss",
               "n_estimators": 2500, "learning_rate": 0.03, "max_depth": 7,
               "subsample": 0.6, "colsample_bytree": 0.6, "gamma": 0.2,
               "reg_alpha": 0.2, "random_state": 42, "early_stopping_rounds": 60, **ACCELERATION_PARAMS},
    "ekor":   {"objective": "multi:softprob", "eval_metric": "mlogloss",
               "n_estimators": 2500, "learning_rate": 0.03, "max_depth": 7,
               "subsample": 0.6, "colsample_bytree": 0.6, "gamma": 0.2,
               "reg_alpha": 0.2, "random_state": 42, "early_stopping_rounds": 60, **ACCELERATION_PARAMS}
}


# --- PENGGABUNGAN KONFIGURASI ---
TRAINING_CONFIG_OPTIONS = {
    'QUICK': {"xgb_params": XGB_PARAMS_CEPAT, "cb_params": XGB_PARAMS_CB},
    'COMPREHENSIVE': {"xgb_params": XGB_PARAMS_AKURAT, "cb_params": XGB_PARAMS_CB},
    'OPTIMIZED': {"xgb_params": XGB_PARAMS_OPTIMIZED, "cb_params": XGB_PARAMS_CB},
    'DEEP_ANALYSIS': {"xgb_params": XGB_PARAMS_DEEP_ANALYSIS, "cb_params": XGB_PARAMS_CB}
}