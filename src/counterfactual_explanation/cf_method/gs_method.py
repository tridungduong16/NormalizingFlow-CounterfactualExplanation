import pandas as pd
import torch

from carla import MLModelCatalog
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import (check_counterfactuals,
                                               encode_feature_names)
# from gs_counterfactuals import growing_spheres_search


from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.growing_spheres.library import (
    growing_spheres_search,
)
from carla.recourse_methods.processing import (
    check_counterfactuals,
    encode_feature_names,
)

from utils.data_catalog import DataCatalog
from utils.helpers import load_configuration_from_yaml
from utils.helpers import load_configuration_from_yaml, load_all_configuration_with_data_name
import numpy as np 
from train_classifier import Net
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


import numpy as np
import pandas as pd
from numpy import linalg as LA
from tqdm import tqdm 

def hyper_sphere_coordindates(n_search_samples, instance, high, low, p_norm=2):
    delta_instance = np.random.randn(n_search_samples, instance.shape[1])
    dist = np.random.rand(n_search_samples) * (high - low) + low  # length range [l, h)
    norm_p = LA.norm(delta_instance, ord=p_norm, axis=1)
    d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
    delta_instance = np.multiply(delta_instance, d_norm)
    candidate_counterfactuals = instance + delta_instance

    return candidate_counterfactuals, dist


def growing_spheres_search(
    instance,
    keys_mutable,
    keys_immutable,
    continuous_cols,
    binary_cols,
    feature_order,
    model,
    n_search_samples=4000,
    p_norm=2,
    step=0.2,
    max_iter=4000,
):
    keys_correct = feature_order

    keys_mutable_continuous = list(set(keys_mutable) - set(binary_cols))
    keys_mutable_binary = list(set(keys_mutable) - set(continuous_cols))

    instance_immutable_replicated = np.repeat(
        instance[keys_immutable].values.reshape(1, -1), n_search_samples, axis=0
    )
    instance_replicated = np.repeat(
        instance.values.reshape(1, -1), n_search_samples, axis=0
    )
    instance_mutable_replicated_continuous = np.repeat(
        instance[keys_mutable_continuous].values.reshape(1, -1),
        n_search_samples,
        axis=0,
    )

    low = 0
    high = low + step

    count = 0
    counter_step = 1

    instance_label = np.argmax(model.predict_proba(instance.values.reshape(1, -1)))
    # print(instance_label)

    counterfactuals_found = False
    candidate_counterfactual_star = np.empty(
        instance_replicated.shape[1],
    )
    candidate_counterfactual_star[:] = np.nan

    for _ in tqdm(range(max_iter)):
        if counterfactuals_found:
            break 
    # while not counterfactuals_found or count < max_iter:

        count = count + counter_step

        # STEP 1 -- SAMPLE POINTS on hyper sphere around instance
        candidate_counterfactuals_continuous, _ = hyper_sphere_coordindates(
            n_search_samples, instance_mutable_replicated_continuous, high, low, p_norm
        )

        # sample random points from Bernoulli distribution
        candidate_counterfactuals_binary = np.random.binomial(
            n=1, p=0.5, size=n_search_samples * len(keys_mutable_binary)
        ).reshape(n_search_samples, -1)

        # make sure inputs are in correct order
        candidate_counterfactuals = pd.DataFrame(
            np.c_[
                instance_immutable_replicated,
                candidate_counterfactuals_continuous,
                candidate_counterfactuals_binary,
            ]
        )
        candidate_counterfactuals.columns = (
            keys_immutable + keys_mutable_continuous + keys_mutable_binary
        )
        # enforce correct order
        candidate_counterfactuals = candidate_counterfactuals[keys_correct]

        # STEP 2 -- COMPUTE l_1 DISTANCES
        if p_norm == 1:
            distances = np.abs(
                (candidate_counterfactuals.values - instance_replicated)
            ).sum(axis=1)
        elif p_norm == 2:
            distances = np.square(
                (candidate_counterfactuals.values - instance_replicated)
            ).sum(axis=1)
        else:
            raise ValueError("Distance not defined yet")

        # counterfactual labels
        y_candidate_logits = model.predict_proba(candidate_counterfactuals.values).reshape(-1)
        y_candidate = np.where(y_candidate_logits >= 0.5, 1, 0)
        indeces = np.where(y_candidate != instance_label)
        candidate_counterfactuals = candidate_counterfactuals.values[indeces]
        candidate_dist = distances[indeces]

        if len(candidate_dist) > 0:  # certain candidates generated
            min_index = np.argmin(candidate_dist)
            candidate_counterfactual_star = candidate_counterfactuals[min_index]
            counterfactuals_found = True

        # no candidate found & push search range outside
        low = high
        high = low + step

    return candidate_counterfactual_star



# class GrowingSpheres(RecourseMethod):

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


# if __name__ == "__main__":

#     DATA_NAME = 'simple_bn'
#     predictive_model, flow_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(DATA_NAME)

#     data_frame = encoder_normalize_data_catalog.data_frame
#     target = encoder_normalize_data_catalog.target
#     features = data_frame.drop(columns = [target], axis = 1).values.astype(np.float32)
#     features = torch.Tensor(features)
#     features = features.cuda()

#     predictions = model_prediction(predictive_model, features)
#     negative_index = negative_prediction_index(predictions)
#     negative_instance_features = negative_prediction_instances(features, negative_index)

#     position = negative_index.nonzero(as_tuple=False).cpu().detach().numpy().reshape(-1)



#     factual_sample = data_frame.loc[position]
#     factual_sample = factual_sample[:10]


#     model = MLModelCatalog(encoder_normalize_data_catalog.data_catalog, predictive_model)
#     model.raw_model.cuda()
#     gs = GrowingSpheres(model)

#     start = timeit.default_timer()
#     counterfactuals_gs = gs.get_counterfactuals(factual_sample)
#     stop = timeit.default_timer()
#     run_time = stop - start

#     cf_sample = torch.Tensor(counterfactuals_gs.values).cuda()
#     counterfactuals_gs[target] = model_prediction(predictive_model, cf_sample).cpu().detach().numpy()

#     factual_sample.to_csv(configuration_for_proj['result_simple_bn'].format("original_instance_gs.csv"), index=False)
#     counterfactuals_gs.to_csv(configuration_for_proj['result_simple_bn'].format("cf_sample_gs.csv"), index=False)

#     benchmark_instance = Benchmark(factual_sample, counterfactuals_gs, run_time)
#     benchmark_distance = benchmark_instance.compute_distances()
#     benchmark_time = benchmark_instance.compute_average_time()
#     print(benchmark_distance)
#     print(benchmark_time)




