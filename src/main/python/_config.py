import configparser


def ConfigMap(section):
    config = configparser.ConfigParser()
    config.read("_config.ini")
    props = {}
    options = config.options(section)
    for option in options:
        props[option] = config.get(section, option)
    return props
