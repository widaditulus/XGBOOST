# data_fetcher.py

# -*- coding: utf-8 -*-
import os
import pandas as pd
import requests
from datetime import datetime, timedelta # // PERBAIKAN: Menambahkan impor timedelta
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
        if not force_github:
            df_from_db = get_latest_data(self.pasaran)
            if df_from_db is not None and not df_from_db.empty:
                logger.info(f"Data untuk {self.pasaran} dimuat dari database lokal. Total: {len(df_from_db)} baris.")
                df_from_db['result'] = df_from_db['result'].astype(str)

                latest_date_in_db = df_from_db['date'].max()
                # Cek jika data lebih tua dari kemarin (lebih fleksibel untuk pasaran yang tidak buka tiap hari)
                if latest_date_in_db.date() < (datetime.now() - timedelta(days=1)).date():
                    logger.warning(f"Data di database lokal ketinggalan (terakhir {latest_date_in_db.date()}). Memaksa ambil dari GitHub.")
                    return self.fetch_data(force_github=True)

                return df_from_db

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

            df = df.rename(columns={date_col: 'date', result_col: 'result'})
            df['date'] = df['date'].apply(self._parse_date)
            df.dropna(subset=['date', 'result'], inplace=True)
            df = df[['date', 'result']].sort_values('date').reset_index(drop=True)

            save_data_to_db(self.pasaran, df)

            return df

        except requests.exceptions.RequestException as e:
            raise DataFetchingError(f"Gagal mengambil data untuk {self.pasaran}: {e}")