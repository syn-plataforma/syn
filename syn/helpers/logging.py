import log4p


def set_logger():
    logger = log4p.GetLogger(__name__)
    return logger.logger
