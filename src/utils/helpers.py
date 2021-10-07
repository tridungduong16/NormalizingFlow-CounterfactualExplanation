from typing import Dict

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


def load_setup(path, method, data_name) -> Dict:
    with open(path, "r") as f:
        setup_catalog = yaml.safe_load(f)
    hyperparams = setup_catalog['recourse_methods'][method]["hyperparams"]
    hyperparams["data_name"] = data_name
    return hyperparams
