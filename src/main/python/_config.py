import configparser
import logging

def ConfigMap(section):
    config = configparser.ConfigParser()
    config.read("/conf/_config.ini")
    props = {}
    options = config.options(section)
    for option in options:
        props[option] = config.get(section, option)
    return props

def Logging(app):
    logger = logging.getLogger(app)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger