# continual_learner.py (File Baru)

# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import pandas as pd
# UPDATED: Menambahkan impor 'threading' yang hilang untuk memperbaiki NameError
import threading
from utils import logger
from constants import CONTINUAL_LEARNING_CONFIG

class ContinualLearner:
    def __init__(self, predictor_instance):
        """
        Menginisialisasi Continual Learner.
        Membutuhkan instance dari ModelPredictor untuk mengakses data dan fungsi training.
        """
        self.predictor = predictor_instance
        self.config = CONTINUAL_LEARNING_CONFIG
        self.pasaran = predictor_instance.pasaran

    def _get_windowed_data(self) -> pd.DataFrame:
        """Mengambil data dari N hari terakhir sesuai konfigurasi untuk retraining."""
        window_days = self.config.get("WINDOW_DAYS", 180)
        logger.info(f"Mengambil data untuk windowed retraining: {window_days} hari terakhir.")

        full_data = self.predictor.data_manager.get_data(force_refresh=True, force_github=True)

        # Pastikan kolom tanggal adalah datetime
        full_data['date'] = pd.to_datetime(full_data['date'])

        cutoff_date = full_data['date'].max() - timedelta(days=window_days)

        windowed_df = full_data[full_data['date'] >= cutoff_date].copy()

        # Fallback ke data penuh jika data di dalam window tidak mencukupi
        if len(windowed_df) < self.predictor.config["strategy"]["min_training_samples"]:
            logger.warning(f"Data dalam window ({len(windowed_df)}) tidak cukup. Fallback ke data penuh.")
            return full_data

        logger.info(f"Menggunakan {len(windowed_df)} baris data untuk incremental retraining.")
        return windowed_df

    def trigger_incremental_retrain(self) -> bool:
        """Memicu proses retraining parsial (incremental) menggunakan data windowed."""
        if not self.config.get("ENABLED", False):
            logger.info("Continual Learning dinonaktifkan dalam konfigurasi.")
            return False

        logger.info(f"Memulai incremental retraining untuk pasaran: {self.pasaran.upper()}")
        try:
            training_data = self._get_windowed_data()

            # Memanggil fungsi train_model di predictor dengan data yang sudah difilter
            success = self.predictor.train_model(
                use_recency_bias=True,
                custom_data=training_data # Mengirim data parsial
            )

            if success:
                logger.info(f"Incremental retraining untuk {self.pasaran.upper()} berhasil.")
                return True
            else:
                logger.error(f"Incremental retraining untuk {self.pasaran.upper()} gagal.")
                return False
        except Exception as e:
            logger.error(f"Error saat incremental retraining: {e}", exc_info=True)
            return False

    def check_and_retrain(self, evaluation_summary: dict, drift_detected: bool):
        """
        Memeriksa hasil evaluasi dan deteksi drift untuk memutuskan apakah retraining diperlukan.
        """
        if not self.config.get("AUTO_TRIGGER_ENABLED", False):
            return

        low_accuracy = evaluation_summary.get("retraining_recommended", False)
        reason = evaluation_summary.get("retraining_reason", "N/A")

        if low_accuracy or drift_detected:
            if drift_detected and not low_accuracy:
                reason = "Feature drift terdeteksi."

            logger.warning(f"Pemicu otomatis terdeteksi untuk {self.pasaran.upper()}. Alasan: {reason}")
            # Menjalankan retraining dalam thread baru agar tidak memblokir proses evaluasi
            thread = threading.Thread(target=self.trigger_incremental_retrain)
            thread.daemon = True
            thread.start()
        else:
            logger.info(f"Performa model {self.pasaran.upper()} masih baik. Tidak perlu retraining otomatis.")