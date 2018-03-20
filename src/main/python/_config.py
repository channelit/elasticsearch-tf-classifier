import configparser
import logging
import logging.config

def ConfigMap(section):
    config = configparser.ConfigParser()
    config.read("/conf/_config.ini")
    props = {}
    options = config.options(section)
    for option in options:
        props[option] = config.get(section, option)
    return props

def Logging(app):
    logging.config.fileConfig('/conf/_logging.ini')
    logger = logging.getLogger(app)
    logger.setLevel(logging.INFO)
    return logger