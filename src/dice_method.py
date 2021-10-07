from typing import Any, Dict

import dice_ml
import pandas as pd
import torch
from carla.models.api import MLModel
from recourse_method import RecourseMethod
from utils.data_catalog import DataCatalog
from utils.helpers import load_config, load_setup
from utils.process import merge_default_parameters


class Dice(RecourseMethod):
    """
    Implementation of Dice from Mothilal et.al. [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "num": int, default: 1
            Number of counterfactuals per factual to generate
        * "desired_class": int, default: 1
            Given a binary class label, the desired class a counterfactual should have (e.g., 0 or 1)
        * "posthoc_sparsity_param": float, default: 0.1
            Fraction of post-hoc preprocessing steps.
    - Restrictions:
        *   Only the model agnostic approach (backend: sklearn) is used in our implementation.
        *   ML model needs to have a transformation pipeline for normalization, encoding and feature order.
            See pipelining at carla/models/catalog/catalog.py for an example ML model class implementation

    .. [1] R. K. Mothilal, Amit Sharma, and Chenhao Tan. 2020. Explaining machine learning classifiers
            through diverse counterfactual explanations
    """

    _DEFAULT_HYPERPARAMS = {"num": 1, "desired_class": 1, "posthoc_sparsity_param": 0.1}

    def __init__(self, mlmodel: MLModel,
                 hyperparams: Dict[str, Any],
                 ML_modelpath: str
                 ) -> None:
        super().__init__(mlmodel)
        self._continous = mlmodel.data.continous
        self._categoricals = mlmodel.data.categoricals
        self._target = mlmodel.data.target

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        # Prepare data for dice data structure
        self._dice_data = dice_ml.Data(
            dataframe=mlmodel.data.raw,
            continuous_features=self._continous,
            outcome_name=self._target,
        )

        m = dice_ml.Model(model_path=ML_modelpath, backend='PYT')
        self._dice = dice_ml.Dice(self._dice_data, m)
        self._num = checked_hyperparams["num"]
        self._desired_class = checked_hyperparams["desired_class"]
        self._post_hoc_sparsity_param = checked_hyperparams["posthoc_sparsity_param"]

        # Need scaler and encoder for get_counterfactual output
        self._scaler = mlmodel.scaler
        self._encoder = mlmodel.encoder
        self._feature_order = mlmodel.feature_input_order

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Prepare factuals
        querry_instances = factuals.copy()

        # check if querry_instances are not empty
        if not querry_instances.shape[0] > 0:
            raise ValueError("Factuals should not be empty")

        # Generate counterfactuals
        dice_exp = self._dice.generate_counterfactuals(
            querry_instances,
            total_CFs=self._num,
            desired_class=self._desired_class,
            posthoc_sparsity_param=self._post_hoc_sparsity_param,
        )

        list_cfs = dice_exp.cf_examples_list
        df_cfs = pd.concat([cf.final_cfs_df for cf in list_cfs], ignore_index=True)
        df_cfs[self._continous] = self._scaler.transform(df_cfs[self._continous])
        encoded_features = self._encoder.get_feature_names(self._categoricals)
        df_cfs[encoded_features] = self._encoder.transform(df_cfs[self._categoricals])
        df_cfs = df_cfs[self._feature_order + [self._target]]

        return df_cfs


if __name__ == "__main__":
    """Setup PATH"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path1 = "/home/trduong/Data/fairCE/experimental_setup.yaml"
    path2 = "/home/trduong/Data/fairCE/configurations.yaml"

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
