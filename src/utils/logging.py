import logging
import os


def setup_logger(name: str = "app", log_path: str = "logs/app.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Use plain FileHandler with truncate to avoid Windows rotate file locks.
        h = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger
