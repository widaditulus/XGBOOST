# database.py (FINAL - Anti Duplikasi Data)
# BEJO
# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

from constants import DB_PATH
from utils import logger, error_handler
from exceptions import DataFetchingError

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
def get_latest_data(pasaran: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Mengambil data historis terbaru dari database untuk pasaran tertentu."""
    with create_connection() as conn:
        table_name = f"data_{pasaran}"
        # UPDATED: Periksa keberadaan tabel sebelum query
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        if cursor.fetchone() is None:
            logger.warning(f"Tabel {table_name} tidak ada di database.")
            return pd.DataFrame(columns=['date', 'result']) # Kembalikan DataFrame kosong

        try:
            query = f"SELECT date, result FROM {table_name} ORDER BY date ASC"
            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql_query(query, conn, parse_dates=['date'])
            return df
        except Exception as e:
            logger.error(f"Error saat membaca dari tabel {table_name}: {e}")
            return pd.DataFrame(columns=['date', 'result'])

@error_handler(logger)
def save_data_to_db(pasaran: str, df: pd.DataFrame):
    """
    Menyimpan atau memperbarui data dari DataFrame ke database.
    UPDATED: Mencegah duplikasi data dengan memeriksa tanggal yang sudah ada.
    """
    if df.empty:
        logger.warning(f"Tidak ada data untuk disimpan ke DB untuk pasaran: {pasaran}")
        return

    table_name = f"data_{pasaran}"
    with create_connection() as conn:
        # 1. Dapatkan tanggal yang sudah ada di database
        try:
            existing_dates_df = pd.read_sql_query(f"SELECT date FROM {table_name}", conn, parse_dates=['date'])
            existing_dates = set(existing_dates_df['date'])
        except (pd.io.sql.DatabaseError, sqlite3.OperationalError):
            # Tabel belum ada, jadi tidak ada tanggal yang sudah ada
            existing_dates = set()

        # 2. Filter DataFrame untuk mendapatkan hanya data baru
        df_to_save = df[~df['date'].isin(existing_dates)].copy()

        if df_to_save.empty:
            logger.info(f"Tidak ada data baru untuk disimpan ke DB untuk pasaran: {pasaran}")
            return

        # 3. Simpan hanya data baru
        logger.info(f"Menyimpan {len(df_to_save)} baris data baru ke tabel {table_name}.")
        # Menggunakan 'append' karena kita sudah memastikan datanya unik
        df_to_save.to_sql(table_name, conn, if_exists='append', index=False)