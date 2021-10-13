from typing import Dict
import yaml
import os

def load_configuration_from_yaml(config_path):
    with open(config_path, 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return conf

def load_hyperparameter_for_method(path, method, data_name) -> Dict:
    setup_catalog = load_configuration_from_yaml(path)
    hyperparameter = setup_catalog['recourse_methods'][method]["hyperparams"]
    hyperparameter["data_name"] = data_name
    return hyperparameter

def load_target_features_name(filename, dataset, keys):
    with open(os.path.join(filename), "r") as file_handle:
        catalog = yaml.safe_load(file_handle)

    for key in keys:
        if catalog[dataset][key] is None:
            catalog[dataset][key] = []

    return catalog[dataset]

