# app.py (Final Definitif)

# -*- coding: utf-8 -*-
import os
import traceback
from flask import Flask, request, jsonify, render_template, abort
from datetime import datetime
import threading
from functools import wraps
import pandas as pd

from predictor import ModelPredictor
from utils import check_dependencies, logger
from constants import PASARAN_LIST, PASARAN_DISPLAY_MAPPING, LOGS_DIR, MODELS_DIR
from exceptions import TrainingError, PredictionError, DataFetchingError
from model_config import TRAINING_CONFIG_OPTIONS

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# --- State Global Aplikasi ---
predictors_cache = {}
predictor_lock = threading.RLock()
training_status = {}
training_lock = threading.Lock()
evaluation_status = {}
evaluation_lock = threading.Lock()
update_status = {}
update_lock = threading.Lock()


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

def get_predictor(pasaran: str) -> ModelPredictor:
    with predictor_lock:
        if pasaran not in predictors_cache:
            logger.info(f"CACHE MISS: Membuat instance predictor baru untuk: {pasaran.upper()}")
            predictors_cache[pasaran] = ModelPredictor(pasaran)
        else:
            logger.info(f"CACHE HIT: Menggunakan instance predictor dari cache untuk: {pasaran.upper()}")
        return predictors_cache[pasaran]

# // PERBAIKAN: Fungsi baru untuk membersihkan cache secara aman
def clear_predictor_cache(pasaran: str):
    with predictor_lock:
        if pasaran in predictors_cache:
            del predictors_cache[pasaran]
            logger.info(f"CACHE CLEARED: Instance predictor untuk {pasaran.upper()} dihapus dari cache.")

@app.route('/')
def index():
    return render_template('index.html', pasaran_list=PASARAN_LIST, display_mapping=PASARAN_DISPLAY_MAPPING)

# --- Rute API untuk Training ---
@app.route('/start-training', methods=['POST'])
@validate_pasaran
def start_training(pasaran):
    training_mode = request.form.get('training_mode', 'OPTIMIZED')
    use_recency_bias = request.form.get('use_recency_bias') == 'true'

    if training_mode not in TRAINING_CONFIG_OPTIONS:
        abort(400, description=f"Mode training '{training_mode}' tidak valid.")
    with training_lock:
        if training_status.get(pasaran) == 'running':
            return jsonify({"status": "error", "message": f"Training untuk {pasaran.upper()} sudah berjalan."}), 409
        training_status[pasaran] = 'running'
        training_status[f"{pasaran}_message"] = "Proses training dimulai..."
    
    thread = threading.Thread(target=run_training_in_background, args=(pasaran, training_mode, use_recency_bias))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "success", "message": f"Proses training untuk {pasaran.upper()} telah dimulai."})

def run_training_in_background(pasaran: str, training_mode: str, use_recency_bias: bool):
    """Fungsi worker yang menjalankan training di thread terpisah."""
    try:
        # // PERBAIKAN: Selalu bersihkan cache sebelum training untuk membuat instance baru
        clear_predictor_cache(pasaran)
        fresh_predictor = ModelPredictor(pasaran)
        success = fresh_predictor.train_model(training_mode=training_mode, use_recency_bias=use_recency_bias)
        
        with training_lock:
            if success:
                training_status[pasaran] = "completed"
                training_status[f"{pasaran}_message"] = f"Model untuk {pasaran.upper()} berhasil diperbarui."
                # // PERBAIKAN: Hapus lagi cache SETELAH training berhasil untuk memastikan
                # // permintaan berikutnya akan membuat instance baru dengan model baru.
                clear_predictor_cache(pasaran)
            else:
                training_status[pasaran] = "failed"
                training_status[f"{pasaran}_message"] = f"Gagal melatih model. Periksa log."
    except Exception as e:
        with training_lock:
            training_status[pasaran] = "failed"
            training_status[f"{pasaran}_message"] = f"Error: {str(e)}"
        logger.error(f"Exception di thread training untuk {pasaran}: {e}", exc_info=True)

@app.route('/training-status', methods=['GET'])
@validate_pasaran
def get_training_status(pasaran):
    with training_lock:
        status = training_status.get(pasaran, 'idle')
        message = training_status.get(f"{pasaran}_message", "")
    return jsonify({"status": status, "message": message})

@app.route('/trigger-continual-learning', methods=['POST'])
@validate_pasaran
def trigger_continual_learning(pasaran):
    logger.info(f"Menerima permintaan manual trigger untuk continual learning pasaran {pasaran}.")
    with training_lock:
        if training_status.get(pasaran) == 'running':
            return jsonify({"status": "error", "message": f"Proses lain sedang berjalan untuk {pasaran.upper()}."}), 409
        training_status[pasaran] = 'running'
        training_status[f"{pasaran}_message"] = "Proses incremental retraining manual dimulai..."

    thread = threading.Thread(target=run_incremental_retrain_in_background, args=(pasaran,))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "success", "message": "Proses incremental retraining telah dimulai."})

def run_incremental_retrain_in_background(pasaran: str):
    try:
        # // PERBAIKAN: Hapus cache sebelum retrain
        clear_predictor_cache(pasaran)
        fresh_predictor = ModelPredictor(pasaran)
        
        if not hasattr(fresh_predictor, 'continual_learner'):
             raise AttributeError("Instance predictor tidak memiliki atribut 'continual_learner'.")
        
        success, message = fresh_predictor.continual_learner.trigger_incremental_retrain()
        with training_lock:
            training_status[pasaran] = "completed" if success else "failed"
            training_status[f"{pasaran}_message"] = message
            if success:
                # // PERBAIKAN: Hapus cache setelah retrain berhasil
                clear_predictor_cache(pasaran)
    except Exception as e:
        logger.error(f"Exception di thread incremental retrain untuk {pasaran}: {e}", exc_info=True)
        with training_lock:
            training_status[pasaran] = "failed"
            training_status[f"{pasaran}_message"] = f"Error: {str(e)}"

@app.route('/start-evaluation', methods=['POST'])
@validate_pasaran
def start_evaluation(pasaran):
    start_date_str = request.form.get('start_date')
    end_date_str = request.form.get('end_date')
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        if start_date > end_date:
            abort(400, description="Tanggal mulai tidak boleh setelah tanggal akhir.")
    except (ValueError, TypeError):
        abort(400, description="Format tanggal tidak valid. Gunakan YYYY-MM-DD.")
    
    with evaluation_lock:
        if evaluation_status.get(pasaran) == 'running':
            return jsonify({"status": "error", "message": f"Evaluasi untuk {pasaran.upper()} sudah berjalan."}), 409
        evaluation_status[pasaran] = 'running'
        evaluation_status[f"{pasaran}_data"] = None

    thread = threading.Thread(target=run_evaluation_in_background, args=(pasaran, start_date, end_date))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "success", "message": f"Proses evaluasi untuk {pasaran.upper()} dimulai."})

def run_evaluation_in_background(pasaran: str, start_date: datetime, end_date: datetime):
    try:
        logger.info(f"Memulai thread evaluasi untuk {pasaran} dari {start_date} hingga {end_date}.")
        predictor = get_predictor(pasaran)
        result = predictor.evaluate_performance(start_date, end_date)
        with evaluation_lock:
            evaluation_status[pasaran] = 'completed'
            evaluation_status[f"{pasaran}_data"] = result
    except Exception as e:
        logger.error(f"Exception di thread evaluasi untuk {pasaran}: {e}", exc_info=True)
        with evaluation_lock:
            evaluation_status[pasaran] = 'failed'
            evaluation_status[f"{pasaran}_data"] = {"summary": {"error": f"Terjadi kesalahan: {str(e)}"}, "results": []}

@app.route('/evaluation-status', methods=['GET'])
@validate_pasaran
def get_evaluation_status(pasaran):
    with evaluation_lock:
        status = evaluation_status.get(pasaran, 'idle')
        data = evaluation_status.get(f"{pasaran}_data", {})
    return jsonify({"status": status, "data": data})

@app.route('/update-data', methods=['POST'])
@validate_pasaran
def update_data(pasaran):
    with update_lock:
        if update_status.get(pasaran) == 'running':
            return jsonify({"status": "error", "message": f"Proses update data untuk {pasaran.upper()} sudah berjalan."}), 409
        update_status[pasaran] = 'running'
        update_status[f"{pasaran}_message"] = "Memulai sinkronisasi data..."

    thread = threading.Thread(target=run_data_update_in_background, args=(pasaran,))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "success", "message": f"Sinkronisasi data untuk {pasaran.upper()} telah dimulai."})

def run_data_update_in_background(pasaran: str):
    try:
        logger.info(f"Memulai thread update data untuk pasaran: {pasaran}")
        # // PERBAIKAN: Hapus cache agar instance baru dibuat dengan data baru
        clear_predictor_cache(pasaran)
        predictor = get_predictor(pasaran)
        predictor.data_manager.get_data(force_refresh=True, force_github=True)
        with update_lock:
            update_status[pasaran] = "completed"
            update_status[f"{pasaran}_message"] = f"Data untuk {pasaran.upper()} berhasil disinkronkan."
            # // PERBAIKAN: Hapus cache setelah update berhasil
            clear_predictor_cache(pasaran)
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
    try:
        predictor = get_predictor(pasaran)
        prediction_result = predictor.predict_next_day(target_date_str=prediction_date_str)
        return jsonify(prediction_result)
    except (PredictionError, DataFetchingError) as e:
        return jsonify({"error": "Gagal membuat prediksi", "details": str(e)}), 400

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

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        check_dependencies()
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)