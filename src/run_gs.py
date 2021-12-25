import timeit

import numpy as np
import pandas as pd
import torch
from numpy import linalg as LA
from tqdm import tqdm

from carla import MLModelCatalog
from carla.evaluation.benchmark import Benchmark
from carla.models.api import MLModel
from carla.recourse_methods import GrowingSpheres
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.growing_spheres.library import \
    growing_spheres_search
from carla.recourse_methods.processing import (check_counterfactuals,
                                               encode_feature_names)
from counterfactual_explanation.models.classifier import Net
from counterfactual_explanation.utils.data_catalog import (
    DataCatalog, EncoderNormalizeDataCatalog, TensorDatasetTraning)
from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import (
    get_latent_representation_from_flow,
    load_pytorch_prediction_model_from_model_path, model_prediction,
    negative_prediction_index, negative_prediction_instances,
    original_space_value_from_latent_representation,
    save_pytorch_model_to_model_path)

if __name__ == "__main__":
    DATA_NAME = 'simple_bn'
    predictive_model, flow_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
        DATA_NAME)

    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target

    features = data_frame.drop(
        columns=[target], axis=1).values.astype(np.float32)
    features = torch.Tensor(features)
    features = features.cuda()

    predictions = model_prediction(predictive_model, features)
    negative_index = negative_prediction_index(predictions)
    negative_instance_features = negative_prediction_instances(
        features, negative_index)

    position = negative_index.nonzero(
        as_tuple=False).cpu().detach().numpy().reshape(-1)

    factual_sample = data_frame.loc[position]
    factual_sample = factual_sample[:10]

    model = MLModelCatalog(
        encoder_normalize_data_catalog.data_catalog, predictive_model)
    model.raw_model.cuda()
    gs = GrowingSpheres(model)

    start = timeit.default_timer()
    counterfactuals_gs = gs.get_counterfactuals(factual_sample)
    stop = timeit.default_timer()
    run_time = stop - start

    cf_sample = torch.Tensor(counterfactuals_gs.values).cuda()
    counterfactuals_gs[target] = model_prediction(
        predictive_model, cf_sample).cpu().detach().numpy()

    factual_sample.to_csv(configuration_for_proj['result_simple_bn'].format(
        "original_instance_gs.csv"), index=False)
    counterfactuals_gs.to_csv(configuration_for_proj['result_simple_bn'].format(
        "cf_sample_gs.csv"), index=False)

    benchmark_instance = Benchmark(
        factual_sample, counterfactuals_gs, run_time, counterfactuals_gs[target].values)
    benchmark_distance = benchmark_instance.compute_distances()
    benchmark_time = benchmark_instance.compute_average_time()
    benchmark_rate = benchmark_instance.compute_success_rate()
    print(benchmark_distance)
    print(benchmark_time)
    print(benchmark_rate)
