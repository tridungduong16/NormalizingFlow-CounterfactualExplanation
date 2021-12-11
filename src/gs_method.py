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
from utils.helpers import load_configuration_from_yaml, load_all_configuration_with_data_name
import numpy as np 
from train_predictive_model import Net
from utils.data_catalog import (DataCatalog, EncoderNormalizeDataCatalog,
                                TensorDatasetTraning)
from utils.helpers import load_configuration_from_yaml, load_all_configuration_with_data_name
from utils.mlcatalog import (load_pytorch_prediction_model_from_model_path,
                             negative_prediction_index,
                             negative_prediction_instances,
                             save_pytorch_model_to_model_path, model_prediction)
from utils.mlcatalog import original_space_value_from_latent_representation, get_latent_representation_from_flow
from tqdm import tqdm 
import timeit
from evaluation.benchmark import Benchmark

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
        for index_, row in df_enc_norm_fact.iterrows():
            print("Factual index {}".format(index_))
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

            # print("-----------------------------")
            # print(counterfactual)

        # df_cfs = check_counterfactuals(self._mlmodel, list_cfs)

        df_cfs = pd.DataFrame(np.array(list_cfs), columns=self._mlmodel.feature_input_order)

        return df_cfs


if __name__ == "__main__":

    DATA_NAME = 'simple_bn'
    predictive_model, flow_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(DATA_NAME)

    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    features = data_frame.drop(columns = [target], axis = 1).values.astype(np.float32)
    features = torch.Tensor(features)
    features = features.cuda()

    predictions = model_prediction(predictive_model, features)
    negative_index = negative_prediction_index(predictions)
    negative_instance_features = negative_prediction_instances(features, negative_index)

    position = negative_index.nonzero(as_tuple=False).cpu().detach().numpy().reshape(-1)



    factual_sample = data_frame.loc[position]
    factual_sample = factual_sample[:10]


    model = MLModelCatalog(encoder_normalize_data_catalog.data_catalog, predictive_model)
    model.raw_model.cuda()
    gs = GrowingSpheres(model)

    start = timeit.default_timer()
    counterfactuals_gs = gs.get_counterfactuals(factual_sample)
    stop = timeit.default_timer()
    run_time = stop - start

    cf_sample = torch.Tensor(counterfactuals_gs.values).cuda()
    counterfactuals_gs[target] = model_prediction(predictive_model, cf_sample).cpu().detach().numpy()

    factual_sample.to_csv(configuration_for_proj['result_simple_bn'].format("original_instance_gs.csv"), index=False)
    counterfactuals_gs.to_csv(configuration_for_proj['result_simple_bn'].format("cf_sample_gs.csv"), index=False)

    benchmark_instance = Benchmark(factual_sample, counterfactuals_gs, run_time)
    benchmark_distance = benchmark_instance.compute_distances()
    benchmark_time = benchmark_instance.compute_average_time()
    print(benchmark_distance)
    print(benchmark_time)

    # print(factual_sample)
    # print(counterfactuals_gs)



