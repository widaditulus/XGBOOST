# ensemble_helper.py

# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import pandas as pd

from utils import logger
from constants import ENSEMBLE_CONFIG
from exceptions import PredictionError

def _train_single_model(model_type, X, y_encoded, params, model_path):
    try:
        if model_type == 'rf': model = RandomForestClassifier(**params)
        elif model_type == 'lgbm': model = lgb.LGBMClassifier(**params)
        else: return
        model.fit(X, y_encoded)
        model.feature_names_ = list(X.columns)
        joblib.dump(model, model_path)
    except Exception as e:
        logger.error(f"Gagal melatih model {model_type.upper()}: {e}", exc_info=True)

def train_ensemble_models(X, y_encoded, model_dir_base, digit):
    rf_params = ENSEMBLE_CONFIG.get("rf_params", {})
    lgbm_params = ENSEMBLE_CONFIG.get("lgbm_params", {})
    tasks = [
        ('rf', X, y_encoded, rf_params, os.path.join(model_dir_base, f"rf_model_{digit}.pkl")),
        ('lgbm', X, y_encoded, lgbm_params, os.path.join(model_dir_base, f"lgbm_model_{digit}.pkl"))
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in executor.map(lambda p: _train_single_model(*p), tasks):
            pass

def train_temporary_ensemble_models(X, y_encoded, digit):
    rf_params = ENSEMBLE_CONFIG.get("rf_params", {})
    lgbm_params = ENSEMBLE_CONFIG.get("lgbm_params", {})
    temp_rf_model, temp_lgbm_model = None, None
    try:
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X, y_encoded)
        rf.feature_names_ = list(X.columns)
        temp_rf_model = rf
    except Exception as e: logger.error(f"EVAL: Gagal melatih model RF sementara: {e}")
    try:
        lgbm = lgb.LGBMClassifier(**lgbm_params)
        lgbm.fit(X, y_encoded)
        lgbm.feature_names_ = list(X.columns)
        temp_lgbm_model = lgbm
    except Exception as e: logger.error(f"EVAL: Gagal melatih model LGBM sementara: {e}")
    return temp_rf_model, temp_lgbm_model

def _validate_and_reorder_features(features: pd.DataFrame, model: any) -> pd.DataFrame:
    if not hasattr(model, 'feature_names_'):
        raise PredictionError(f"Model '{type(model).__name__}' tidak memiliki 'feature_names_'. Harap latih ulang.")
    model_features = model.feature_names_
    input_features_set = set(features.columns)
    model_features_set = set(model_features)
    if input_features_set != model_features_set:
        missing = model_features_set - input_features_set
        extra = input_features_set - model_features_set
        raise PredictionError(f"Ketidakcocokan fitur: Hilang: {missing if missing else 'None'}. Tambahan: {extra if extra else 'None'}. Harap latih ulang.")
    return features[model_features]

def ensemble_predict_proba(models, X):
    all_probas = []
    for model_name, model in models.items():
        if model:
            try:
                validated_X = _validate_and_reorder_features(X, model)
                probas = model.predict_proba(validated_X)
                all_probas.append(probas)
            except Exception as e:
                logger.warning(f"Gagal prediksi dari model {model_name}: {e}")
                raise 
    if not all_probas: raise ValueError("Tidak ada model yang berhasil memberikan prediksi.")
    return np.mean(all_probas, axis=0)