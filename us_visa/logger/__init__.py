import logging
import os
import sys
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logs_path = os.path.join(log_dir, LOG_FILE)

logging.basicConfig(
    format="[ %(asctime)s ] %(filename)s:%(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(logs_path),
        logging.StreamHandler(sys.stdout),
    ],
)