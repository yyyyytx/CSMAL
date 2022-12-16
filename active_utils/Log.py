import os
import logging


def get_logger(logger_name):
    logging.basicConfig(level=logging.INFO,
                        filename=logger_name)
    logger = logging.getLogger()
    return logger


if __name__ == "__main__":
    logger = get_logger("my.log")
    logger.debug("debug test")
    logger.info("info test")
    logger.warn("warn test")
    logger.error("error test")