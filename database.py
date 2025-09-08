# database.py
# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
import threading

from constants import DB_PATH
from utils import logger, error_handler
from exceptions import DataFetchingError

# UPDATED: Impor DataFetcher di sini
from data_fetcher import DataFetcher

@error_handler(logger)
def create_connection():
    """Membuat koneksi ke database SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error saat terhubung ke database: {e}")
        raise DataFetchingError("Gagal terhubung ke database lokal.")

@error_handler(logger)
def create_pasaran_table(pasaran: str):
    """Membuat tabel untuk pasaran tertentu jika belum ada."""
    with create_connection() as conn:
        cursor = conn.cursor()
        table_name = f"data_{pasaran}"
        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            date TEXT,
            result TEXT NOT NULL
        )
        """
        cursor.execute(sql)
        conn.commit()

@error_handler(logger)
def get_latest_data(pasaran: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Mengambil data historis terbaru dari database untuk pasaran tertentu."""
    create_pasaran_table(pasaran)
    with create_connection() as conn:
        table_name = f"data_{pasaran}"
        try:
            query = f"SELECT date, result FROM {table_name} ORDER BY date ASC"
            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql_query(query, conn)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except pd.io.sql.DatabaseError as e:
            logger.warning(f"Tabel {table_name} mungkin kosong atau tidak ada. Error: {e}")
            return None

@error_handler(logger)
def save_data_to_db(pasaran: str, df: pd.DataFrame):
    """Menyimpan atau memperbarui data dari DataFrame ke database."""
    create_pasaran_table(pasaran)
    if df.empty:
        logger.warning(f"Tidak ada data untuk disimpan ke DB untuk pasaran: {pasaran}")
        return

    with create_connection() as conn:
        table_name = f"data_{pasaran}"
        df_to_save = df.copy()
        df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
        df_to_save.to_sql(table_name, conn, if_exists='replace', index=False)
    logger.info(f"Data untuk pasaran {pasaran} berhasil disimpan ke database.")

# UPDATED: Menambahkan kelas DataManager yang diperbaiki
class DataManager:
    def __init__(self, pasaran):
        self.pasaran = pasaran
        self.df = None
        self.lock = threading.RLock()
        self.fetcher = DataFetcher(pasaran)

    def _clean_and_pad_result(self, series: pd.Series) -> pd.Series:
        cleaned_series = series.astype(str).str.strip()
        cleaned_series = cleaned_series.str.split('.').str[0]
        cleaned_series = cleaned_series.str.zfill(4)
        return cleaned_series

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: raise DataFetchingError("DataFrame sumber kosong atau tidak ada.")
        required = ['date', 'result']
        if not all(col in df.columns for col in required): raise DataFetchingError(f"Kolom wajib '{', '.join(required)}' tidak ditemukan di data.")
        df = df.copy()
        if df['date'].duplicated().any():
            df.drop_duplicates(subset=['date'], keep='first', inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df['result'] = self._clean_and_pad_result(df['result'])
        valid_results = df['result'].str.match(r'^\d{4}$')
        if not valid_results.all():
            invalid_count = (~valid_results).sum()
            logger.warning(f"Ditemukan {invalid_count} format 'result' yang tidak valid (bukan 4-digit) SETELAH pembersihan. Baris ini akan dihapus.")
            df = df[valid_results]
        if df.empty: raise DataFetchingError("Tidak ada data valid yang tersisa setelah proses pembersihan.")
        return df

    @error_handler(logger)
    def get_data(self, force_refresh: bool = False, force_github: bool = False) -> Optional[pd.DataFrame]:
        with self.lock:
            if self.df is None or force_refresh:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        df_raw = self.fetcher.fetch_data(force_github=force_github or (attempt > 0))
                        df_validated = self._validate_data(df_raw)
                        df_sorted = df_validated.sort_values("date").reset_index(drop=True)
                        self.df = df_sorted
                        
                        if self.df is not None and not self.df.empty:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Percobaan {attempt + 1}/{max_retries} gagal: {e}")
                        if attempt == max_retries - 1:
                            raise DataFetchingError(f"Gagal mendapatkan data setelah {max_retries} percobaan: {e}")
            
            return self.df.copy() if self.df is not None else None

    @error_handler(logger)
    def check_data_freshness(self) -> Dict[str, Any]:
        local_df = self.get_data(force_refresh=False, force_github=False)
        if local_df is None or local_df.empty:
            return {"status": "stale", "message": "Data lokal tidak ditemukan."}
        local_latest_date = local_df['date'].max()
        try:
            remote_df = self.fetcher.fetch_data(force_github=True, use_lock=False)
            if remote_df is None or remote_df.empty:
                return {"status": "error", "message": "Gagal mengambil data remote."}
            remote_latest_date = self._validate_data(remote_df)['date'].max()
            status = "stale" if local_latest_date < remote_latest_date else "latest"
            return {
                "status": status,
                "local_date": local_latest_date.strftime('%Y-%m-%d'),
                "remote_date": remote_latest_date.strftime('%Y-%m-%d')
            }
        except Exception as e:
            logger.error(f"Error saat memeriksa kesegaran data: {e}", exc_info=True)
            return {"status": "error", "message": "Terjadi kesalahan saat perbandingan data."}