import os
from utils.file_utils import ensure_folder_exists

ANSI_COLORS = {
    'black': '\033[30m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'gray': '\033[90m',  # Added gray color
    'reset': '\033[0m',
}

def init_logging():
    import logging
    import sys

    # Create a custom formatter with color
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            "ERROR": ANSI_COLORS["red"],
            "WARNING": ANSI_COLORS["yellow"],
            "INFO": ANSI_COLORS["cyan"],
            "DEBUG": ANSI_COLORS["gray"],
            "RESET": ANSI_COLORS["reset"],
        }

        def format(self, record):
            log_message = super(ColoredFormatter, self).format(record)
            log_level = record.levelname

            # Add color to log messages based on log level
            return (
                f"{self.COLORS.get(log_level, '')}{log_message}{self.COLORS['RESET']}"
            )


    # Create a console handler and set the formatter
    ensure_folder_exists("logs")
    #logging.basicConfig(filename=os.path.join("logs", "console.log"), level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    logging.root.handlers.clear()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    logging.root.addHandler(console_handler)
