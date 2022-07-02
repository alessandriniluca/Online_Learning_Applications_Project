import logging

# global constants
LOG_FORMAT = "[%(levelname)s] [%(name)s] [%(asctime)s] %(message)s"

logging.basicConfig(
    format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


def get_logger(file_name: str) -> logging.Logger:
    """
    Return logger with name and level given in input. Call should be:
    my_logger = get_logger(__name__)
    Parameters
    ----------
        - file_name: name of the logger. It should be the name of the
        current file.
    """

    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG)  # the level is fixed to DEBUG
    return logger
