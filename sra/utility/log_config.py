import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a logger with a preconfigured handler and formatter."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Allow all messages from DEBUG and above

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)  # Let the handler handle all levels too

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",  # No milliseconds
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False  # Prevent double logging

    return logger
