import os
import logging
from datetime import datetime
import structlog

class CustomLogger:
    _configured = False
    _log_file_path = None

    def __init__(self, log_dir="logs"):
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Create ONE log file path per process/run
        if CustomLogger._log_file_path is None:
            log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
            CustomLogger._log_file_path = os.path.join(self.logs_dir, log_file)

        self.log_file_path = CustomLogger._log_file_path

    def get_logger(self, name=__file__):
        logger_name = os.path.basename(name)

        # Configure handlers ONLY once
        if not CustomLogger._configured:
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(message)s"))

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            # Important: basicConfig is "first call wins"
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
                handlers=[console_handler, file_handler],
            )

            structlog.configure(
                processors=[
                    structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                    structlog.processors.add_log_level,
                    structlog.processors.EventRenamer(to="event"),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ],
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )

            CustomLogger._configured = True

        return structlog.get_logger(logger_name)
