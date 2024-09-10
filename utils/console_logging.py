import datetime
import os
import logging
import sys
from utils.file_utils import ensure_folder_exists

logging.getLogger("websockets.server").setLevel(logging.INFO)

class Emojis:
    high_power = chr(0x26A1)
    recycle = chr(0x267B)
    disk = chr(0x1F4BD)
    plugin = chr(0x1F50C)
    rocket = chr(0x1F680)

class Colors:

    reset = "\033[0m"

    # Regular colors
    black = "\033[30m"
    gray = "\033[90m"
    blue = "\033[34m"
    green = "\033[32m"
    red = "\033[31m"
    yellow = "\033[33m"
    orange = "\033[38;5;208m"
    purple = "\033[38;5;129m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    darkgreen = "\033[38;5;28m"
    bluegreen = "\033[38;5;30m"
    darkgray = "\033[38;5;59m"
    darkestgray = "\033[38;5;235m"  

    # Bright colors
    bright_black = "\033[90m"
    bright_red = "\033[91m"
    bright_green = "\033[92m"
    bright_yellow = "\033[93m"
    bright_blue = "\033[94m"
    bright_magenta = "\033[95m"
    bright_cyan = "\033[96m"
    bright_white = "\033[97m"

    bold = "\033[1m"


def log_loading(name, path):
    log_disk(f"Loading {name}: {path}")

def log_disk(message):
    logging.info(f"{Emojis.disk} {Colors.purple}{message}{Colors.reset}")

def log_highpower(message):
    logging.info(f"{Emojis.high_power} {Colors.purple}{message}{Colors.reset}")

def log_generate(message):
    logging.info(f"{Emojis.rocket} {Colors.purple}{message}{Colors.reset}")

def log_recycle(message):
    logging.info(f"{Emojis.recycle} {Colors.bluegreen}{message}{Colors.reset}")

def log_plugin(plugin_name):
    logging.info(f"{Emojis.plugin} {Colors.cyan}Using plugin: {plugin_name}{Colors.reset}")


def init_logging():

    # Create a custom formatter with color
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            "ERROR": Colors.red,
            "WARNING": Colors.bold + Colors.cyan,
            "INFO": Colors.cyan,
            "DEBUG": Colors.gray,
            "RESET": Colors.reset,
        }

        def format(self, record):
            log_message = super(ColoredFormatter, self).format(record)
            log_level = record.levelname

            current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_link = f"\033]8;;file://{os.path.abspath(record.pathname)}#{record.lineno}\033\\{record.filename}:{record.lineno}\033]8;;\033\\"

            term_width = os.get_terminal_size().columns
            right_justify_spaces = " " * (term_width - (len(log_message) + len(current_timestamp) + 40))

            with open(os.path.join("logs", "console.log"), "a", encoding="utf-8") as log_file:
                log_file.write(f"[{current_timestamp}] {log_message}\n")

            return f"{Colors.gray}[{current_timestamp}]{self.COLORS['RESET']} {self.COLORS.get(log_level, '')}{log_message} {right_justify_spaces}{Colors.darkestgray}{file_link}{self.COLORS['RESET']}"

    # Create a console handler and set the formatter
    ensure_folder_exists("logs")
    logging.basicConfig(
        filename=os.path.join("logs", "console.log"), level=logging.INFO
    )    
    logging.root.handlers.clear()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    logging.root.addHandler(console_handler)


def show_banner():
    print(
        "                                 ___                                __ "
        + "\n.--------..-----..-----..-----..'  _|.--.--.     ______     .---.-.|__|"
        + "\n|        ||  _  ||     ||  _  ||   _||  |  |    |______|    |  _  ||  |"
        + "\n|__|__|__||_____||__|__||_____||__|  |___  |                |___._||__|"
        + "\n                                     |_____|                           "
    )
