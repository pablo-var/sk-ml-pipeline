"""Logging configuration module"""

import logging as lg


def setup_logging():
    """Configuration of the logging settings"""
    logger = lg.getLogger()
    logger.setLevel(lg.INFO)
    logger.info('Logging set up')
    logger = lg.getLogger()
    logger.handlers = []
    handler = lg.StreamHandler()
    formatter = lg.Formatter('[%(asctime)s] [%(levelname)-5s] [%(name)-12s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(lg.INFO)
    logger.info('Logging set up')