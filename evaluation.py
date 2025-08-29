# evaluation.py

# -*- coding: utf-8 -*-
import numpy as np

def calculate_brier_score(y_true_one_hot, y_prob):
    """
    Menghitung Brier Score Loss.
    Skor: [0, 2]. Semakin rendah semakin baik (0 adalah skor sempurna).
    Mengukur akurasi prediksi probabilistik.
    """
    if y_true_one_hot.shape != y_prob.shape:
        raise ValueError("Shape dari y_true dan y_prob harus sama.")
    return np.mean(np.sum((y_prob - y_true_one_hot)**2, axis=1))

def calculate_ece(y_true_labels, y_pred_probs_max, y_pred_probs_full, n_bins=10):
    """
    Menghitung Expected Calibration Error (ECE).
    Skor: [0, 1]. Semakin rendah semakin baik.
    Mengukur seberapa baik 'confidence' model sesuai dengan akurasi aktualnya.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_probs_max > bin_lower) & (y_pred_probs_max <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Periksa apakah ada prediksi yang benar di dalam bin ini
            # y_true_labels[in_bin] akan berisi label sebenarnya untuk sampel di bin ini
            # y_pred_probs_full[in_bin].argmax(axis=1) akan berisi label yang diprediksi
            correct_predictions_in_bin = (y_true_labels[in_bin] == y_pred_probs_full[in_bin].argmax(axis=1))
            accuracy_in_bin = np.mean(correct_predictions_in_bin)
            
            avg_confidence_in_bin = np.mean(y_pred_probs_max[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece
