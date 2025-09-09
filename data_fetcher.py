# data_fetcher.py (FINAL - Disesuaikan dengan DB Anti-Duplikasi)
# BEJO
# -*- coding: utf-8 -*-
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional
from io import StringIO

from utils import logger, error_handler
from exceptions import DataFetchingError
from database import get_latest_data, save_data_to_db

class DataFetcher:
    def __init__(self, pasaran: str):
        self.pasaran = pasaran.lower()
        self.github_urls = {
            "sgp": "https://raw.githubusercontent.com/widaditulus/4D/main/sgp_data.csv",
            "hk": "https://raw.githubusercontent.com/widaditulus/4D/main/hk_data.csv",
            "sydney": "https://raw.githubusercontent.com/widaditulus/4D/main/sydney_data.csv",
            "taiwan": "https://raw.githubusercontent.com/widaditulus/4D/main/taiwan_data.csv",
            "china": "https://raw.githubusercontent.com/widaditulus/4D/main/china_data.csv",
            "magnum": "https://raw.githubusercontent.com/widaditulus/4D/main/magnum_data.csv",
        }
        self.github_url = self.github_urls.get(self.pasaran)

    def _parse_date(self, date_str):
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d'):
            try: return datetime.strptime(str(date_str).strip(), fmt)
            except (ValueError, TypeError): continue
        return None

    @error_handler(logger)
    def fetch_data(self, force_github: bool = False) -> Optional[pd.DataFrame]:
        # Coba ambil dari DB terlebih dahulu jika tidak dipaksa dari GitHub
        if not force_github:
            df_from_db = get_latest_data(self.pasaran)
            # Jika DB tidak kosong, gunakan data dari DB
            if df_from_db is not None and not df_from_db.empty:
                logger.info(f"Data untuk {self.pasaran} dimuat dari database lokal. Total: {len(df_from_db)} baris.")
                return df_from_db

        # Jika dipaksa, atau jika DB kosong, ambil dari GitHub
        logger.info(f"Mengambil data dari GitHub untuk {self.pasaran}.")
        if not self.github_url:
            raise DataFetchingError(f"URL tidak ditemukan untuk pasaran {self.pasaran}")

        try:
            response = requests.get(self.github_url, timeout=15)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text), dtype={'result': str, 'nomor': str})
            if df.empty: raise DataFetchingError(f"Data kosong dari URL untuk {self.pasaran}")

            df.columns = df.columns.str.lower().str.strip()
            date_col = next((c for c in df.columns if c in ['date', 'tanggal']), None)
            result_col = next((c for c in df.columns if c in ['result', 'nomor']), None)

            if not date_col or not result_col:
                raise DataFetchingError(f"Kolom tanggal atau hasil tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")

            df.rename(columns={date_col: 'date', result_col: 'result'}, inplace=True)
            df['date'] = df['date'].apply(self._parse_date)
            df.dropna(subset=['date', 'result'], inplace=True)
            
            # Gabungkan dengan data dari DB yang mungkin sudah ada (untuk kelengkapan)
            df_from_db = get_latest_data(self.pasaran)
            if df_from_db is not None and not df_from_db.empty:
                 combined_df = pd.concat([df_from_db, df]).drop_duplicates(subset=['date'], keep='last')
            else:
                 combined_df = df

            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            
            # Simpan data gabungan (fungsi save sudah handle duplikasi)
            save_data_to_db(self.pasaran, combined_df)

            return combined_df

        except requests.exceptions.RequestException as e:
            raise DataFetchingError(f"Gagal mengambil data untuk {self.pasaran}: {e}")