# monitoring.py
# -*- coding: utf-8 -*-
import numpy as np
from utils import logger

class ModelPerformanceMonitor:
    def __init__(self):
        self.performance_history = {}
        
    def track_performance(self, pasaran, date, metrics):
        if pasaran not in self.performance_history:
            self.performance_history[pasaran] = []
        
        cb_accuracy = metrics.get('cb_accuracy', 0)
        
        self.performance_history[pasaran].append({
            'date': date,
            'accuracy': cb_accuracy
        })
        
        logger.info(f"MONITOR: Tracked performance for {pasaran} on {date}. CB Accuracy: {cb_accuracy}%")
        
        if self._check_degradation(pasaran):
            self._send_alert(pasaran)
    
    def _check_degradation(self, pasaran):
        history = self.performance_history.get(pasaran, [])
        if len(history) < 10:
            return False
            
        recent_history = history[-10:]
        recent_acc = np.mean([h['accuracy'] for h in recent_history[-5:]])
        older_acc = np.mean([h['accuracy'] for h in recent_history[:5]])
        
        if older_acc == 0: # Hindari pembagian dengan nol
            return False

        degraded = recent_acc < older_acc * 0.8
        if degraded:
            logger.warning(f"DEGRADATION DETECTED for {pasaran}: Recent Avg Accuracy ({recent_acc:.2f}%) is 20% lower than Older Avg Accuracy ({older_acc:.2f}%)")
        return degraded
    
    def _send_alert(self, pasaran):
        logger.critical(f"ALERT: Significant performance degradation detected for model '{pasaran.upper()}'!")