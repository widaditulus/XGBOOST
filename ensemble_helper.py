# ensemble_helper.py

# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from utils import logger
from constants import ENSEMBLE_CONFIG

def _train_single_model(model_type, X, y_encoded, params, model_path):
    """Fungsi internal untuk melatih satu model (RF atau LGBM)."""
    try:
        if model_type == 'rf':
            model = RandomForestClassifier(**params)
        elif model_type == 'lgbm':
            model = lgb.LGBMClassifier(**params)
        else:
            logger.warning(f"Tipe model tidak dikenal: {model_type}")
            return

        model.fit(X, y_encoded)
        joblib.dump(model, model_path)
        logger.info(f"Model {model_type.upper()} berhasil dilatih dan disimpan di {model_path}")
    except Exception as e:
        logger.error(f"Gagal melatih model {model_type.upper()}: {e}", exc_info=True)

def train_ensemble_models(X, y_encoded, model_dir_base, digit):
    """Melatih dan menyimpan model komplementer (RF & LGBM) secara paralel."""
    
    rf_params = ENSEMBLE_CONFIG.get("rf_params", {})
    lgbm_params = ENSEMBLE_CONFIG.get("lgbm_params", {})

    tasks = [
        ('rf', X, y_encoded, rf_params, os.path.join(model_dir_base, f"rf_model_{digit}.pkl")),
        ('lgbm', X, y_encoded, lgbm_params, os.path.join(model_dir_base, f"lgbm_model_{digit}.pkl"))
    ]

    # Melatih model secara paralel untuk efisiensi
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_train_single_model, *task) for task in tasks]
        for future in futures:
            try:
                future.result() # Menunggu semua task selesai dan menangkap exception jika ada
            except Exception as e:
                logger.error(f"Terjadi error pada thread training ensemble: {e}")

def ensemble_predict_proba(models, X):
    """
    Mengambil prediksi probabilitas dari beberapa model dan merata-ratakannya.
    Models adalah dict: {'xgb': model1, 'rf': model2, ...}
    """
    all_probas = []
    
    for model_name, model in models.items():
        if model:
            try:
                # UPDATED: Tambahkan validasi jumlah fitur sebelum prediksi
                # Ini untuk mencegah crash jika model lama dimuat dengan data (fitur) baru
                if hasattr(model, 'n_features_in_') and model.n_features_in_ != X.shape[1]:
                    logger.warning(
                        f"Skipping model '{model_name}' due to feature mismatch. "
                        f"Model expects {model.n_features_in_} features, but input has {X.shape[1]}."
                    )
                    continue # Lewati model ini dan lanjut ke model berikutnya

                probas = model.predict_proba(X)
                all_probas.append(probas)
            except Exception as e:
                logger.warning(f"Gagal mendapatkan prediksi dari model {model_name}: {e}")

    if not all_probas:
        raise ValueError("Tidak ada model yang berhasil memberikan prediksi.")

    if len(all_probas) == 1:
        return all_probas[0]

    avg_probas = np.mean(all_probas, axis=0)
    return avg_probas