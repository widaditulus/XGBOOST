# start.py (Final - Diperbaiki)
# Skrip ini digunakan untuk menjalankan aplikasi Flask dalam server produksi Waitress.
# Gunakan ini untuk deployment, bukan 'python app.py'

from waitress import serve
from app import app, active_threads # UPDATED: Import active_threads untuk graceful shutdown
import os
import dotenv
from utils import logger, check_dependencies
import signal
import sys

# PERBAIKAN: Fungsi signal handler untuk graceful shutdown
# UPDATED: Memastikan graceful shutdown menangani semua thread aktif
def graceful_shutdown(signum, frame):
    logger.info("Shutdown signal received. Waiting for active threads to complete...")
    # Beri waktu beberapa detik bagi thread untuk selesai
    for thread in active_threads:
        thread.join(timeout=10.0)
    logger.info("All active threads have been handled. Exiting.")
    sys.exit(0)

if __name__ == "__main__":
    dotenv.load_dotenv()

    # PERHATIAN: Pengecekan dependensi dan pendaftaran signal handler
    # dilakukan di sini, bukan di app.py. Ini adalah praktik terbaik
    # untuk memisahkan logika aplikasi dari logika deployment.
    check_dependencies()
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    logger.info("Starting production server with Waitress...")
    # Jalankan server Waitress secara langsung
    serve(app, host='0.0.0.0', port=5000)