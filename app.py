# app.py (FINAL - PENANGANAN ERROR LEBIH BAIK)
# BEJO
# -*- coding: utf-8 -*-
import os
import traceback
import json
from flask import Flask, request, jsonify, render_template, abort
from datetime import datetime
import threading
from functools import wraps
import pandas as pd
import signal
import sys
from waitress import serve
import dotenv

from predictor import ModelPredictor
from utils import check_dependencies, logger
from constants import PASARAN_LIST, PASARAN_DISPLAY_MAPPING, LOGS_DIR, MODELS_DIR
from exceptions import TrainingError, PredictionError, DataFetchingError
from model_config import TRAINING_CONFIG_OPTIONS
from tuner import HyperparameterTuner

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# UPDATED: Mengganti lru_cache dengan mekanisme caching berbasis dictionary sederhana untuk mencegah kebocoran memori
predictor_cache = {}
predictor_lock = threading.Lock()

active_threads = []

training_status = {}
training_lock = threading.Lock()
evaluation_status = {}
evaluation_lock = threading.Lock()
update_status = {}
update_lock = threading.Lock()
tuning_status = {}
tuning_lock = threading.Lock()
cb_tuning_status = {}
cb_tuning_lock = threading.Lock()

@app.errorhandler(Exception)
def handle_unexpected_error(e):
    if hasattr(e, 'code') and 400 <= e.code < 600:
        return jsonify(error=str(e.description)), e.code
    tb_str = traceback.format_exc()
    logger.error(f"Unhandled Exception: {e}\n{tb_str}")
    return jsonify(error="Terjadi kesalahan internal pada server.", details=str(e)), 500

def validate_pasaran(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        pasaran = kwargs.get('pasaran') or request.args.get('pasaran') or request.form.get('pasaran')
        if not pasaran or pasaran not in PASARAN_LIST:
            abort(400, description=f"Parameter 'pasaran' tidak valid atau tidak ditemukan.")
        kwargs['pasaran'] = pasaran
        return f(*args, **kwargs)
    return decorated_function

# UPDATED: Menghapus lru_cache dan mengimplementasikan cache manual yang aman untuk thread
def get_predictor(pasaran: str) -> ModelPredictor:
    """
    Mengambil atau membuat instance ModelPredictor untuk pasaran tertentu.
    Menggunakan cache sederhana yang aman untuk thread.
    """
    with predictor_lock:
        if pasaran not in predictor_cache:
            logger.info(f"CACHE MISS: Membuat instance predictor baru untuk: {pasaran.upper()}")
            predictor_cache[pasaran] = ModelPredictor(pasaran)
        else:
            logger.info(f"CACHE HIT: Menggunakan instance predictor dari cache untuk: {pasaran.upper()}")
        return predictor_cache[pasaran]

@app.route('/')
def index():
    return render_template('index.html', pasaran_list=PASARAN_LIST, display_mapping=PASARAN_DISPLAY_MAPPING)

@app.route('/data-status/<pasaran>')
@validate_pasaran
def data_status(pasaran):
    try:
        predictor = get_predictor(pasaran)
        status = predictor.data_manager.check_data_freshness()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Gagal memeriksa status data untuk {pasaran}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

def manage_thread(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        active_threads[:] = [t for t in active_threads if t.is_alive()]
        
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.daemon = True
        active_threads.append(thread)
        thread.start()
    return wrapper

@app.route('/start-training', methods=['POST'])
@validate_pasaran
def start_training(pasaran):
    training_mode = request.form.get('training_mode', 'OPTIMIZED')
    use_recency_bias = request.form.get('use_recency_bias') == 'true'

    if training_mode not in TRAINING_CONFIG_OPTIONS and training_mode != 'AUTO':
        abort(400, description=f"Mode training '{training_mode}' tidak valid.")
        
    # UPDATED: Lock diperoleh SEBELUM pengecekan status untuk mencegah race condition
    with training_lock:
        if training_status.get(pasaran) == 'running':
            return jsonify({"status": "error", "message": f"Training untuk {pasaran.upper()} sudah berjalan."}), 409
        training_status[pasaran] = 'running'
        training_status[f"{pasaran}_message"] = "Proses training dimulai..."

    run_training_in_background_threaded(pasaran, training_mode, use_recency_bias)
    return jsonify({"status": "success", "message": f"Proses training untuk {pasaran.upper()} telah dimulai."})

@manage_thread
def run_training_in_background_threaded(pasaran: str, training_mode: str, use_recency_bias: bool):
    """Fungsi ini sekarang menangani TrainingError secara spesifik."""
    try:
        logger.info(f"Memulai thread training untuk pasaran: {pasaran} dengan mode: {training_mode}")
        predictor_to_train = get_predictor(pasaran)
        success = predictor_to_train.train_model(training_mode=training_mode, use_recency_bias=use_recency_bias)
        with training_lock:
            training_status[pasaran] = "completed"
            training_status[f"{pasaran}_message"] = f"Model untuk {pasaran.upper()} berhasil diperbarui."
    except TrainingError as e:
        with training_lock:
            training_status[pasaran] = "failed"
            training_status[f"{pasaran}_message"] = f"Error Training: {str(e)}"
        logger.error(f"TrainingError di thread training untuk {pasaran}: {e}", exc_info=True)
    except Exception as e:
        with training_lock:
            training_status[pasaran] = "failed"
            training_status[f"{pasaran}_message"] = f"Error Tak Terduga: {str(e)}"
        logger.error(f"Exception di thread training untuk {pasaran}: {e}", exc_info=True)

@app.route('/training-status', methods=['GET'])
@validate_pasaran
def get_training_status(pasaran):
    with training_lock:
        status = training_status.get(pasaran, 'idle')
        message = training_status.get(f"{pasaran}_message", "")
    return jsonify({"status": status, "message": message})
    
@app.route('/start-tuning', methods=['POST'])
@validate_pasaran
def start_tuning(pasaran):
    # UPDATED: Lock diperoleh SEBELUM pengecekan status untuk mencegah race condition
    with tuning_lock:
        if tuning_status.get(pasaran) == 'running':
            return jsonify({"status": "error", "message": f"Optimasi 4D untuk {pasaran.upper()} sudah berjalan."}), 409
        tuning_status[pasaran] = 'running'
        tuning_status[f"{pasaran}_message"] = "Memulai proses optimasi parameter 4D..."

    run_tuning_in_background_threaded(pasaran)
    return jsonify({"status": "success", "message": f"Optimasi parameter 4D untuk {pasaran.upper()} telah dimulai."})

@manage_thread
def run_tuning_in_background_threaded(pasaran: str):
    digits = ["as", "kop", "kepala", "ekor"]
    try:
        for digit in digits:
            with tuning_lock:
                tuning_status[f"{pasaran}_message"] = f"Mengoptimasi parameter 4D untuk digit: {digit.upper()}..."
            tuner = HyperparameterTuner(pasaran, digit, mode='4D')
            tuner.run_tuning()
        with tuning_lock:
            tuning_status[pasaran] = "completed"
            tuning_status[f"{pasaran}_message"] = f"Optimasi 4D untuk {pasaran.upper()} berhasil."
    except Exception as e:
        with tuning_lock:
            tuning_status[pasaran] = "failed"
            tuning_status[f"{pasaran}_message"] = f"Error saat optimasi 4D: {str(e)}"
        logger.error(f"Exception di thread tuning 4D untuk {pasaran}: {e}", exc_info=True)

@app.route('/tuning-status', methods=['GET'])
@validate_pasaran
def get_tuning_status(pasaran):
    with tuning_lock:
        status = tuning_status.get(pasaran, 'idle')
        message = tuning_status.get(f"{pasaran}_message", "")
    return jsonify({"status": status, "message": message})

@app.route('/start-tuning-cb', methods=['POST'])
@validate_pasaran
def start_cb_tuning(pasaran):
    # UPDATED: Lock diperoleh SEBELUM pengecekan status untuk mencegah race condition
    with cb_tuning_lock:
        if cb_tuning_status.get(pasaran) == 'running':
            return jsonify({"status": "error", "message": f"Optimasi CB untuk {pasaran.upper()} sudah berjalan."}), 409
        cb_tuning_status[pasaran] = 'running'
        cb_tuning_status[f"{pasaran}_message"] = "Memulai proses optimasi parameter CB..."

    run_cb_tuning_in_background_threaded(pasaran)
    return jsonify({"status": "success", "message": f"Optimasi parameter CB untuk {pasaran.upper()} telah dimulai."})

@manage_thread
def run_cb_tuning_in_background_threaded(pasaran: str):
    try:
        for i in range(10):
            digit = str(i)
            with cb_tuning_lock:
                cb_tuning_status[f"{pasaran}_message"] = f"Mengoptimasi parameter CB untuk digit: {digit}..."
            tuner = HyperparameterTuner(pasaran, digit, mode='CB')
            tuner.run_tuning()
        with cb_tuning_lock:
            cb_tuning_status[pasaran] = "completed"
            cb_tuning_status[f"{pasaran}_message"] = f"Optimasi CB untuk {pasaran.upper()} berhasil."
    except Exception as e:
        with cb_tuning_lock:
            cb_tuning_status[pasaran] = "failed"
            cb_tuning_status[f"{pasaran}_message"] = f"Error saat optimasi CB: {str(e)}"
        logger.error(f"Exception di thread tuning CB untuk {pasaran}: {e}", exc_info=True)

@app.route('/cb-tuning-status', methods=['GET'])
@validate_pasaran
def get_cb_tuning_status(pasaran):
    with cb_tuning_lock:
        status = cb_tuning_status.get(pasaran, 'idle')
        message = cb_tuning_status.get(f"{pasaran}_message", "")
    return jsonify({"status": status, "message": message})

@app.route('/start-evaluation', methods=['POST'])
@validate_pasaran
def start_evaluation(pasaran):
    start_date_str = request.form.get('start_date')
    end_date_str = request.form.get('end_date')
    evaluation_mode = request.form.get('evaluation_mode', 'quick') 

    try:
        start_date = datetime.strptime(start_date_str, '%Y-m-d')
        end_date = datetime.strptime(end_date_str, '%Y-m-d')
        if start_date > end_date:
            abort(400, description="Tanggal mulai tidak boleh setelah tanggal akhir.")
    except (ValueError, TypeError):
        abort(400, description="Format tanggal tidak valid. Gunakan YYYY-MM-DD.")

    # UPDATED: Lock diperoleh SEBELUM pengecekan status untuk mencegah race condition
    with evaluation_lock:
        if evaluation_status.get(pasaran) == 'running':
            return jsonify({"status": "error", "message": f"Evaluasi untuk {pasaran.upper()} sudah berjalan."}), 409
        evaluation_status[pasaran] = 'running'
        evaluation_status[f"{pasaran}_data"] = None

    logger.info(f"Memulai thread evaluasi untuk {pasaran}. Peringatan: Proses ini akan melatih model untuk setiap hari dalam rentang waktu, yang dapat memakan waktu lama.")
    
    run_evaluation_in_background_threaded(pasaran, start_date, end_date, evaluation_mode)
    return jsonify({"status": "success", "message": f"Proses evaluasi untuk {pasaran.upper()} dimulai. Proses ini akan memakan waktu."})

@manage_thread
def run_evaluation_in_background_threaded(pasaran: str, start_date: datetime, end_date: datetime, evaluation_mode: str):
    try:
        logger.info(f"Memulai thread evaluasi untuk {pasaran} dari {start_date} hingga {end_date} dengan mode '{evaluation_mode}'.")
        predictor = get_predictor(pasaran)
        result = predictor.evaluate_performance(start_date, end_date, evaluation_mode=evaluation_mode)
        with evaluation_lock:
            evaluation_status[pasaran] = 'completed'
            evaluation_status[f"{pasaran}_data"] = result
            evaluation_status[f"{pasaran}_message"] = "Evaluasi selesai."
    except Exception as e:
        logger.error(f"Exception di thread evaluasi untuk {pasaran}: {e}", exc_info=True)
        with evaluation_lock:
            evaluation_status[pasaran] = 'failed'
            evaluation_status[f"{pasaran}_data"] = {"summary": {"error": f"Terjadi kesalahan: {str(e)}"}, "results": []}
            evaluation_status[f"{pasaran}_message"] = f"Evaluasi gagal: {str(e)}"

@app.route('/evaluation-status', methods=['GET'])
@validate_pasaran
def get_evaluation_status(pasaran):
    with evaluation_lock:
        status = evaluation_status.get(pasaran, 'idle')
        data = evaluation_status.get(f"{pasaran}_data", {})
        message = evaluation_status.get(f"{pasaran}_message", "Status evaluasi.")
    return jsonify({"status": status, "data": data, "message": message})

@app.route('/update-data', methods=['POST'])
@validate_pasaran
def update_data(pasaran):
    # UPDATED: Lock diperoleh SEBELUM pengecekan status untuk mencegah race condition
    with update_lock:
        if update_status.get(pasaran) == 'running':
            return jsonify({"status": "error", "message": f"Proses update data untuk {pasaran.upper()} sudah berjalan."}), 409
        update_status[pasaran] = 'running'
        update_status[f"{pasaran}_message"] = "Memulai sinkronisasi data..."

    run_data_update_in_background_threaded(pasaran)
    return jsonify({"status": "success", "message": f"Sinkronisasi data untuk {pasaran.upper()} telah dimulai."})

@manage_thread
def run_data_update_in_background_threaded(pasaran: str):
    try:
        logger.info(f"Memulai thread update data untuk pasaran: {pasaran}")
        predictor = get_predictor(pasaran)
        predictor.data_manager.get_data(force_refresh=True, force_github=True)
        with update_lock:
            update_status[pasaran] = "completed"
            update_status[f"{pasaran}_message"] = f"Data untuk {pasaran.upper()} berhasil disinkronkan."
    except Exception as e:
        with update_lock:
            update_status[pasaran] = "failed"
            update_status[f"{pasaran}_message"] = f"Gagal sinkronisasi data: {str(e)}"
        logger.error(f"Exception di thread update data untuk {pasaran}: {e}", exc_info=True)

@app.route('/update-status', methods=['GET'])
@validate_pasaran
def get_update_status(pasaran):
    with update_lock:
        status = update_status.get(pasaran, 'idle')
        message = update_status.get(f"{pasaran}_message", "")
    return jsonify({"status": status, "message": message})

@app.route('/predict', methods=['POST'])
@validate_pasaran
def predict(pasaran):
    prediction_date_str = request.form.get('prediction_date')
    evaluation_mode = request.form.get('evaluation_mode', 'deep')
    
    if not prediction_date_str:
        abort(400, description="Parameter 'prediction_date' tidak boleh kosong.")
    
    try:
        predictor = get_predictor(pasaran)
        prediction_result = predictor.predict_next_day(
            target_date_str=prediction_date_str, 
            evaluation_mode=evaluation_mode
        )
        return jsonify(prediction_result)
    except (PredictionError, DataFetchingError) as e:
        return jsonify({"error": "Gagal membuat prediksi", "details": str(e)}), 400

@app.route('/check-optimized-params/<pasaran>')
@validate_pasaran
def check_optimized_params(pasaran):
    model_dir = os.path.join(MODELS_DIR, pasaran)
    all_files_exist = True
    for digit in ["as", "kop", "kepala", "ekor"]:
        if not os.path.exists(os.path.join(model_dir, f"best_params_{digit}.json")):
            all_files_exist = False
            break
    return jsonify({"available": all_files_exist})

@app.route('/feature-importance/<pasaran>')
@validate_pasaran
def get_feature_importance(pasaran):
    importance_data = {}
    for digit in ["as", "kop", "kepala", "ekor"]:
        file_path = os.path.join(MODELS_DIR, pasaran, f"feature_importance_{digit}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            importance_data[digit] = df.head(10).to_dict('records')
        else:
            importance_data[digit] = []
    return jsonify(importance_data)

@app.route('/drift-log')
def get_drift_log():
    log_path = os.path.join(LOGS_DIR, 'drift_detection.log')
    if not os.path.exists(log_path):
        return jsonify(["Log deteksi drift belum dibuat."])
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return jsonify(lines[-50:])

@app.route('/debug/model-status/<pasaran>')
@validate_pasaran
def debug_model_status(pasaran):
    predictor = get_predictor(pasaran)
    status = {
        "pasaran": pasaran,
        "models_ready": predictor.models_ready,
        "loaded_models": [d for d, m in predictor.models.items() if m is not None],
        "data_manager_df_shape": predictor.data_manager.df.shape if predictor.data_manager.df is not None else "Data belum dimuat",
    }
    return jsonify(status)