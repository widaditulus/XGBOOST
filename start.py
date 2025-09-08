# start.py
# Skrip ini digunakan untuk menjalankan aplikasi Flask dalam server produksi Waitress.
# Gunakan ini untuk deployment, bukan 'python app.py'

from waitress import serve
from app import app
import os
import dotenv
from utils import logger, check_dependencies
import signal
import sys

# PERBAIKAN: Fungsi signal handler untuk graceful shutdown
def graceful_shutdown(signum, frame):
    logger.info("Shutdown signal received. Exiting.")
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