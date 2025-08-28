# exceptions.py
# -*- coding: utf-8 -*-

class DataFetchingError(Exception):
    """Exception kustom untuk error saat pengambilan data."""
    pass

class TrainingError(Exception):
    """Exception kustom untuk error saat training model."""
    pass

class PredictionError(Exception):
    """Exception kustom untuk error saat prediksi."""
    pass
