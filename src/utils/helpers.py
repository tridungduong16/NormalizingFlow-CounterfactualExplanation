import yaml


def load_config(config_path):
    """

    :param config_path:
    :type config_path:
    :return:
    :rtype:
    """
    with open(config_path, 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return conf
