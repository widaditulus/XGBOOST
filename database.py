# database.py
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
def create_pasaran_table(pasaran: str):
    """Membuat tabel untuk pasaran tertentu jika belum ada."""
    with create_connection() as conn:
        cursor = conn.cursor()
        table_name = f"data_{pasaran}"
        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            date TEXT PRIMARY KEY,
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
