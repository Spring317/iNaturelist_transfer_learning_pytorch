import os
from datetime import datetime
from typing import Optional


LOG_FILE_PATH: Optional[str] = None


def initialize_logger(log_dir: str = "./logs", filename: Optional[str] = None):
    """
    Initializes the logger by setting up the log directory and log file.

    If no filename is provided, a default filename with a timestamp will be used.

    Args:
        log_dir (str, optional): The directory where log files will be saved. Defaults to "./logs".
        filename (str, optional): The name of the log file. If not provided, a timestamped filename will be used.
    """
    global LOG_FILE_PATH
    os.makedirs(log_dir, exist_ok=True)
    if not filename:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"log_{timestamp}.txt"
    LOG_FILE_PATH = os.path.join(log_dir, filename)


def log(message: str, verbose: bool = True, level: str = "INFO"):
    """
    Logs a message to the console and to the log file.

    The message is printed to the console if `verbose` is True and appended to 
    the log file specified by `initialize_logger`.

    Args:
        message (str): The message to log.
        verbose (bool, optional): Whether to print the message to the console. Defaults to True.
        level (str, optional): The log level (e.g., "INFO", "ERROR"). Defaults to "INFO".
    """
    formatted = f"[{level}] {message}"
    if verbose:
        print(formatted)
    if LOG_FILE_PATH:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(f"{formatted}\n")
