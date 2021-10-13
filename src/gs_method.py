import pandas as pd
import torch

from carla import MLModelCatalog
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import (check_counterfactuals,
                                               encode_feature_names)
from gs_counterfactuals import growing_spheres_search
from utils.data_catalog import DataCatalog
from utils.helpers import load_configuration_from_yaml


class GrowingSpheres(RecourseMethod):

    def __init__(self, mlmodel: MLModel) -> None:
        super().__init__(mlmodel)

        self._immutables = encode_feature_names(
            self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        )
        self._mutables = [
            feature
            for feature in self._mlmodel.feature_input_order
            if feature not in self._immutables
        ]
        self._continuous = self._mlmodel.data.continous
        self._categoricals_enc = encode_feature_names(
            self._mlmodel.data.categoricals, self._mlmodel.feature_input_order
        )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Normalize and encode data
        df_enc_norm_fact = self.encode_normalize_order_factuals(factuals)

        list_cfs = []
        for _, row in df_enc_norm_fact.iterrows():
            counterfactual = growing_spheres_search(
                row,
                self._mutables,
                self._immutables,
                self._continuous,
                self._categoricals_enc,
                self._mlmodel.feature_input_order,
                self._mlmodel,
            )
            list_cfs.append(counterfactual)

        df_cfs = check_counterfactuals(self._mlmodel, list_cfs)

        return df_cfs


if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIG_PATH = "/home/trduong/Data/fairCE/src/carla/data_catalog.yaml"
    CONFIG_FOR_PROJECT = "/home/trduong/Data/fairCE/configuration/project_configurations.yaml"
    EXPR_PATH = "/home/trduong/Data/fairCE/experimental_setup.yaml"
    MODEL_PATH = "/home/trduong/Data/anaconda_environment/research/lib/python3.8/site-packages/dice_ml/utils/sample_trained_models/adult.pth"
    DATA_PATH = '/home/trduong/Data/fairCE/data/processed_adult.csv'
    DATA_NAME = 'adult'

    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)

    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    model = MLModelCatalog(data_catalog, MODEL_PATH)
    model.raw_model.to(DEVICE)
    gs = GrowingSpheres(model)
