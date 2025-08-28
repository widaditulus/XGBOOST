# app.py (VERSI REFACTOR TOTAL - FUNGSI LENGKAP)
# -*- coding: utf-8 -*-
import os
import traceback
from flask import Flask, request, jsonify, render_template, abort
from datetime import datetime
import threading
from functools import wraps
import pandas as pd

from predictor import ModelPredictor
from utils import logger
from constants import PASARAN_LIST, PASARAN_DISPLAY_MAPPING, LOGS_DIR, MODELS_DIR

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# --- MANAJEMEN STATUS TERPUSAT & STABIL ---
predictors_cache = {}
predictor_lock = threading.RLock()
# Satu kamus untuk semua status tugas, mencegah konflik
task_status = {}
task_lock = threading.Lock()

@app.errorhandler(Exception)
def handle_unexpected_error(e):
    tb_str = traceback.format_exc()
    logger.error(f"UNHANDLED EXCEPTION: {e}\n{tb_str}")
    # Berikan detail error ke frontend untuk debug jika dalam mode debug
    details = str(e) if app.debug else "Silakan periksa log server."
    return jsonify(error="Terjadi kesalahan fatal pada server.", details=details), 500

def validate_pasaran(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        pasaran = request.args.get('pasaran') or request.form.get('pasaran')
        if not pasaran or pasaran not in PASARAN_LIST:
            abort(400, description="Parameter 'pasaran' tidak valid atau kosong.")
        kwargs['pasaran'] = pasaran
        return f(*args, **kwargs)
    return decorated_function

def get_predictor(pasaran: str) -> ModelPredictor:
    with predictor_lock:
        if pasaran not in predictors_cache:
            logger.info(f"Membuat instance predictor BARU untuk: {pasaran.upper()}")
            predictors_cache[pasaran] = ModelPredictor(pasaran)
        return predictors_cache[pasaran]

def clear_predictor_cache(pasaran: str):
    with predictor_lock:
        if pasaran in predictors_cache:
            del predictors_cache[pasaran]
            logger.info(f"CACHE DIHAPUS untuk {pasaran.upper()}.")

def run_task_in_background(pasaran: str, task_name: str, target_func, *args):
    """Fungsi generik dan aman untuk menjalankan semua tugas di background."""
    def wrapper():
        try:
            result = target_func(*args)
            with task_lock:
                task_status[pasaran] = {"status": "completed", "name": task_name, "message": f"Proses {task_name} berhasil.", "data": result or {}}
        except Exception as e:
            error_message = f"Error fatal di {task_name}: {str(e)}"
            logger.error(f"Exception di background thread {task_name} untuk {pasaran}: {e}", exc_info=True)
            with task_lock:
                task_status[pasaran] = {"status": "failed", "name": task_name, "message": error_message, "data": {}}

    with task_lock:
        current_task = task_status.get(pasaran, {})
        if current_task.get('status') == 'running':
            return False, f"Tugas '{current_task.get('name', 'lain')}' sedang berjalan untuk {pasaran.upper()}."
        task_status[pasaran] = {'status': 'running', 'name': task_name, 'message': f'Memulai proses {task_name}...', 'data': {}}

    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    return True, f"Proses {task_name} untuk {pasaran.upper()} telah dimulai."

@app.route('/')
def index():
    return render_template('index.html', pasaran_list=PASARAN_LIST, display_mapping=PASARAN_DISPLAY_MAPPING)

@app.route('/start-training', methods=['POST'])
@validate_pasaran
def start_training(pasaran):
    training_mode = request.form.get('training_mode', 'OPTIMIZED')
    use_recency_bias = request.form.get('use_recency_bias') == 'true'
    def target():
        clear_predictor_cache(pasaran)
        predictor = get_predictor(pasaran)
        predictor.train_model(training_mode=training_mode, use_recency_bias=use_recency_bias)
        clear_predictor_cache(pasaran)
    success, message = run_task_in_background(pasaran, "training", target)
    return jsonify({"status": "success" if success else "error", "message": message}), 200 if success else 409

@app.route('/update-data', methods=['POST'])
@validate_pasaran
def update_data(pasaran):
    def target():
        clear_predictor_cache(pasaran)
        predictor = get_predictor(pasaran)
        predictor.data_manager.get_data(force_refresh=True, force_github=True)
    success, message = run_task_in_background(pasaran, "update data", target)
    return jsonify({"status": "success" if success else "error", "message": message}), 200 if success else 409

@app.route('/start-evaluation', methods=['POST'])
@validate_pasaran
def start_evaluation(pasaran):
    start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d')
    end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d')
    def target():
        predictor = get_predictor(pasaran)
        return predictor.evaluate_performance(start_date, end_date)
    success, message = run_task_in_background(pasaran, "evaluation", target)
    return jsonify({"status": "success" if success else "error", "message": message}), 200 if success else 409

# Endpoint tunggal untuk semua status, agar sinkron dengan frontend
# Frontend (script.js) harus disesuaikan untuk menangani format ini.
@app.route('/task-status', methods=['GET'])
@validate_pasaran
def get_task_status(pasaran):
    with task_lock:
        status_data = task_status.get(pasaran, {"status": "idle", "message": "Tidak ada tugas aktif."})
        
        # Terjemahkan ke format yang diharapkan oleh script.js
        if status_data.get('name') == 'training':
            return jsonify({"status": status_data['status'], "message": status_data['message']})
        elif status_data.get('name') == 'evaluation':
            return jsonify({"status": status_data['status'], "data": status_data.get('data', {})})
        elif status_data.get('name') == 'update data':
             return jsonify({"status": status_data['status'], "message": status_data['message']})
        else: # Idle atau status lain
            return jsonify(status_data)

# Endpoint-endpoint lain yang memerlukan data langsung (bukan background task)
@app.route('/predict', methods=['POST'])
@validate_pasaran
def predict(pasaran):
    prediction_date_str = request.form.get('prediction_date')
    if not prediction_date_str: abort(400, "Tanggal prediksi wajib diisi.")
    try:
        predictor = get_predictor(pasaran)
        return jsonify(predictor.predict_next_day(target_date_str=prediction_date_str))
    except Exception as e:
        logger.error(f"PREDICT FAILED: {e}", exc_info=True)
        return jsonify({"error": "Gagal total membuat prediksi.", "details": str(e)}), 500

@app.route('/feature-importance/<pasaran>')
@validate_pasaran
def get_feature_importance(pasaran):
    # Fungsi ini cukup cepat, tidak perlu background task
    try:
        predictor = get_predictor(pasaran)
        return jsonify(predictor.get_feature_importance_data())
    except Exception as e:
        return jsonify({"error": f"Gagal mengambil feature importance: {e}"})

@app.route('/drift-log')
def get_drift_log():
    try:
        log_path = os.path.join(LOGS_DIR, 'drift_detection.log')
        if not os.path.exists(log_path): return jsonify(["Log drift belum ada."])
        with open(log_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        return jsonify(lines[-50:]) # Ambil 50 baris terakhir
    except Exception as e:
        return jsonify({"error": f"Gagal membaca log drift: {e}"})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)