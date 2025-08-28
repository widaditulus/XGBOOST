# test_app.py
# -*- coding: utf-8 -*-
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from io import StringIO

# Tambahkan path proyek agar bisa mengimpor modul aplikasi
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app
from data_fetcher import DataFetcher
from predictor import DataManager

# PEMBARUAN (Testing): Suite pengujian komprehensif baru.
# File ini berisi unit test untuk memvalidasi komponen-komponen kritis.
# Untuk menjalankan: `pytest test_app.py`

@pytest.fixture
def client():
    """Fixture untuk membuat test client Flask."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Test 1: Memastikan halaman utama dapat diakses (HTTP 200)."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Sistem Prediksi 4D" in response.data

def test_validate_pasaran_invalid(client):
    """Test 2: Memastikan endpoint mengembalikan 400 jika pasaran tidak valid."""
    response = client.post('/predict', data={'pasaran': 'tidakvalid', 'prediction_date': '2025-01-01'})
    assert response.status_code == 400
    assert b"tidak valid" in response.data

def test_predict_missing_date(client):
    """Test 3: Memastikan endpoint prediksi mengembalikan 400 jika tanggal tidak ada."""
    response = client.post('/predict', data={'pasaran': 'sgp'})
    assert response.status_code == 400
    assert b"tidak boleh kosong" in response.data

@patch('data_fetcher.requests.get')
def test_data_fetcher_success(mock_get):
    """Test 4: Memvalidasi DataFetcher dapat mem-parsing data CSV dengan benar."""
    # Siapkan mock response dari requests.get
    csv_data = "tanggal,nomor\n2025-01-01,1234\n2025-01-02,5678"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = csv_data
    mock_get.return_value = mock_response

    fetcher = DataFetcher('sgp')
    df = fetcher.fetch_data()

    assert not df.empty
    assert list(df.columns) == ['date', 'result']
    assert len(df) == 2
    assert df['result'].iloc[0] == '1234'

def test_data_manager_validation():
    """Test 5: Memvalidasi logika pembersihan dan validasi di DataManager."""
    # Buat data mentah dengan berbagai masalah
    raw_data = {
        'date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-01', 'invalid-date'],
        'result': ['1234', '567', '8910', '9999', '1111']
    }
    df_raw = pd.DataFrame(raw_data)
    
    dm = DataManager('test')
    df_validated = dm._validate_data(df_raw)

    # Harusnya:
    # - Menghapus 'invalid-date'
    # - Menghapus '567' (bukan 4 digit)
    # - Menghapus duplikat '2025-01-01', mempertahankan yang pertama ('1234')
    assert len(df_validated) == 2
    assert '1234' in df_validated['result'].values
    assert '8910' in df_validated['result'].values
    assert '567' not in df_validated['result'].values
    assert df_validated['date'].is_unique

@patch('predictor.ModelPredictor.predict_next_day')
def test_predict_endpoint_success(mock_predict, client):
    """Test 6: Memastikan endpoint /predict berfungsi dengan mock predictor."""
    # Atur mock untuk mengembalikan hasil prediksi yang diharapkan
    mock_predict.return_value = {
        "final_4d_prediction": "1234",
        "prediction_date": "2025-01-02"
    }

    # Patch get_predictor agar tidak membuat instance sungguhan
    with patch('app.get_predictor') as mock_get_predictor:
        mock_predictor_instance = MagicMock()
        mock_predictor_instance.predict_next_day.return_value = mock_predict.return_value
        mock_get_predictor.return_value = mock_predictor_instance

        response = client.post('/predict', data={'pasaran': 'hk', 'prediction_date': '2025-01-02'})
        
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['final_4d_prediction'] == "1234"
        
