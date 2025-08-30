# monitoring.py
# -*- coding: utf-8 -*-
import numpy as np
from utils import logger
from constants import CRITICAL_ACCURACY_THRESHOLD

class ModelPerformanceMonitor:
    def __init__(self):
        self.performance_history = {}

    def track_performance(self, pasaran, date, metrics):
        """
        Melacak performa model dari berbagai metrik.
        Metrics harus berisi: 'as_accuracy', 'kop_accuracy', 'kepala_accuracy', 'ekor_accuracy'
        """
        if pasaran not in self.performance_history:
            self.performance_history[pasaran] = []

        # Memastikan semua metrik utama ada
        if not all(m in metrics for m in ['as_accuracy', 'kop_accuracy', 'kepala_accuracy', 'ekor_accuracy']):
            logger.warning(f"MONITOR: Metrik akurasi tidak lengkap untuk {pasaran} pada {date}. Tracking dibatalkan.")
            return

        self.performance_history[pasaran].append({
            'date': date,
            'as_accuracy': metrics.get('as_accuracy', 0),
            'kop_accuracy': metrics.get('kop_accuracy', 0),
            'kepala_accuracy': metrics.get('kepala_accuracy', 0),
            'ekor_accuracy': metrics.get('ekor_accuracy', 0),
            'cb_accuracy': metrics.get('cb_accuracy', 0)
        })

        logger.info(f"MONITOR: Performa dilacak untuk {pasaran} pada {date}. Akurasi: AS({metrics['as_accuracy']:.2%}), KOP({metrics['kop_accuracy']:.2%}), KEPALA({metrics['kepala_accuracy']:.2%}), EKOR({metrics['ekor_accuracy']:.2%})")

        self._check_degradation(pasaran)

    def _check_degradation(self, pasaran):
        """
        Memeriksa degradasi performa pada semua metrik digit.
        """
        history = self.performance_history.get(pasaran, [])
        if len(history) < 10:
            return False

        # Ambil rata-rata performa dari 5 data terakhir
        recent_metrics = history[-5:]
        recent_acc_as = np.mean([h['as_accuracy'] for h in recent_metrics])
        recent_acc_kop = np.mean([h['kop_accuracy'] for h in recent_metrics])
        recent_acc_kepala = np.mean([h['kepala_accuracy'] for h in recent_metrics])
        recent_acc_ekor = np.mean([h['ekor_accuracy'] for h in recent_metrics])

        # Ambil rata-rata performa dari 5 data sebelumnya
        older_metrics = history[-10:-5]
        older_acc_as = np.mean([h['as_accuracy'] for h in older_metrics])
        older_acc_kop = np.mean([h['kop_accuracy'] for h in older_metrics])
        older_acc_kepala = np.mean([h['kepala_accuracy'] for h in older_metrics])
        older_acc_ekor = np.mean([h['ekor_accuracy'] for h in older_metrics])

        degradation_detected = False
        degradation_reasons = []

        # Periksa setiap digit untuk degradasi signifikan
        if older_acc_as > 0 and recent_acc_as < older_acc_as * 0.8:
            degradation_detected = True
            degradation_reasons.append(f"AS: Turun dari {older_acc_as:.2%} ke {recent_acc_as:.2%}")
        if older_acc_kop > 0 and recent_acc_kop < older_acc_kop * 0.8:
            degradation_detected = True
            degradation_reasons.append(f"KOP: Turun dari {older_acc_kop:.2%} ke {recent_acc_kop:.2%}")
        if older_acc_kepala > 0 and recent_acc_kepala < older_acc_kepala * 0.8:
            degradation_detected = True
            degradation_reasons.append(f"KEPALA: Turun dari {older_acc_kepala:.2%} ke {recent_acc_kepala:.2%}")
        if older_acc_ekor > 0 and recent_acc_ekor < older_acc_ekor * 0.8:
            degradation_detected = True
            degradation_reasons.append(f"EKOR: Turun dari {older_acc_ekor:.2%} ke {recent_acc_ekor:.2%}")

        if degradation_detected:
            logger.warning(f"DEGRADATION DETECTED for {pasaran}. Alasan: {', '.join(degradation_reasons)}")
            self._send_alert(pasaran)

        # Periksa juga jika salah satu akurasi jatuh di bawah ambang batas kritis
        if recent_acc_kepala < CRITICAL_ACCURACY_THRESHOLD or recent_acc_ekor < CRITICAL_ACCURACY_THRESHOLD:
            logger.critical(f"CRITICAL: Akurasi Kepala/Ekor ({recent_acc_kepala:.2%}/{recent_acc_ekor:.2%}) di bawah ambang batas kritis {CRITICAL_ACCURACY_THRESHOLD:.2%}!")
            self._send_alert(pasaran, critical=True)

        return degradation_detected

    def _send_alert(self, pasaran, critical=False):
        alert_type = "CRITICAL ALERT" if critical else "ALERT"
        logger.critical(f"{alert_type}: Significant performance degradation detected for model '{pasaran.upper()}'!")