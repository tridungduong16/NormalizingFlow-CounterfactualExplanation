import pandas as pd
from utils.helpers import load_configuration_from_yaml, load_hyperparameter_for_method
from utils.data_catalog import DataCatalog

# def convert_from_data_to_tensor():
#     configuration_for_dataset = load_configuration_from_yaml("/home/trduong/Data/fairCE/configuration/data_catalog.yaml")
#     configuration_for_project = load_configuration_from_yaml("/home/trduong/Data/fairCE/configuration/project_configuration.yaml")
#     df = pd.read_csv(configuration_for_dataset['adult'])
#

# if __name__ == '__main__':
#     datalog_configuration_path = "/home/trduong/Data/fairCE/configuration/data_catalog.yaml"
#     configuration_for_project = load_configuration_from_yaml("/home/trduong/Data/fairCE/configuration/project_configurations.yaml")
#     data_path = configuration_for_project['adult_dataset']
#     data_name = 'adult'
#     dataset = DataCatalog(data_name, data_path, datalog_configuration_path)
#
#
#     print(1)