
from typing import Any, Dict

import dice_ml
import pandas as pd
import torch
from carla.models.api import MLModel
from recourse_method import RecourseMethod
from counterfactual_explanation.utils.data_catalog import DataCatalog
from counterfactual_explanation.utils.helpers import load_config, load_setup
from counterfactual_explanation.utils.process import merge_default_parameters


if __name__ == "__main__":
    """Setup PATH"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path1 = "/home/trduong/Data/fairCE/experimental_setup.yaml"
    path2 = "/home/trduong/Data/fairCE/project_configurations.yaml"

    """load model"""
    data_name = 'adult'
    data_path = '/home/trduong/Data/fairCE/data/processed_adult.csv'
    config_path = "/home/trduong/Data/fairCE/src/carla/data_catalog.yaml"
    dataset = DataCatalog(data_name, data_path, config_path)

    """Load configuration"""
    ML_modelpath = '/home/trduong/Data/anaconda_environment/research/lib/python3.8/site-packages/dice_ml/utils/sample_trained_models/adult.pth'
    configuration = load_config(path2)
    method = 'dice'
    hyperparams = load_setup(path1, method, data_name)
    mlmodel = MLModelCatalog(dataset, "ann", backend="pytorch")
    mlmodel.raw_model.to(device)
    dc = Dice(mlmodel, hyperparams, ML_modelpath)