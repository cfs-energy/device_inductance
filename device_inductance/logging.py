import logging
from pathlib import Path
from rich.logging import RichHandler


def log() -> logging.Logger:
    """Get the root logger instance shared across the top level module"""
    return logging.getLogger("device_inductance")


def logger_is_set_up() -> bool:
    """Check if a rich logging handler has been set up yet"""
    return any([isinstance(x, RichHandler) for x in log().handlers])


def logger_setup(level: int, fp: Path | str | None = None):
    """Set up a logger, optionally writing to a file in addition to the terminal."""
    # Write to a file, clearing it first, if requested
    handlers = []
    if fp is not None:
        file_handler = logging.FileHandler(fp, mode="w")
        handlers.append(file_handler)

    # Always write to the terminal
    terminal_handler = RichHandler(level=level)
    handlers.append(terminal_handler)

    # Include timestamp and the origin of the message
    logging.basicConfig(
        format="%(name)s | %(asctime)s | %(message)s",
        level=level,
        handlers=handlers,
    )


def logger_setup_default():
    """Set up the default logger with WARNING level, if there is not already one set up."""
    if logger_is_set_up():
        return

    logger_setup(logging.WARNING)
