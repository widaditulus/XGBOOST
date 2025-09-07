# utils.py
# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from constants import LOGS_DIR
from exceptions import DataFetchingError, TrainingError, PredictionError

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    return logger

logger = setup_logger('AppLogger', os.path.join(LOGS_DIR, 'app.log'))
drift_logger = setup_logger('DriftLogger', os.path.join(LOGS_DIR, 'drift_detection.log'))

def error_handler(logger_instance):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (DataFetchingError, TrainingError, PredictionError) as e:
                logger_instance.error(f"Error di {func.__name__}: {e}")
                raise
            except Exception as e:
                logger_instance.error(f"Unexpected error di {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

def log_system_info():
    """
    PERBAIKAN: Fungsi baru untuk mendeteksi dan menampilkan status GPU.
    Ini akan menjadi indikator utama Anda saat aplikasi dimulai.
    """
    logger.info("--- Pengecekan Perangkat Keras untuk Akselerasi ---")
    try:
        # Coba jalankan perintah nvidia-smi, cara paling andal untuk cek GPU NVIDIA
        import subprocess
        subprocess.check_output('nvidia-smi')
        logger.info("✅ INDIKATOR: GPU NVIDIA terdeteksi. XGBoost akan mencoba menggunakan akselerasi GPU ('gpu_hist').")
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.warning("⚠️ INDIKATOR: GPU NVIDIA tidak terdeteksi atau nvidia-smi tidak ditemukan.")
        logger.info("XGBoost akan menggunakan akselerasi CPU (menggunakan semua core).")
    logger.info("----------------------------------------------------")


def check_dependencies():
    logger.info("--- Pengecekan Versi Library ---")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Pandas version: {pd.__version__}")
    logger.info(f"Scikit-learn version: {sklearn.__version__}")
    logger.info(f"XGBoost version: {xgb.__version__}")
    logger.info("---------------------------------")