import timeit
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from carla.evaluation.benchmark import Benchmark
from counterfactual_explanation.flow_ce.flow_method import (
    CounterfactualSimpleBn, FindCounterfactualSample)
from counterfactual_explanation.models.classifier import Net
from counterfactual_explanation.utils.helpers import \
    load_all_configuration_with_data_name
from counterfactual_explanation.utils.mlcatalog import (
    model_prediction, negative_prediction_index, prediction_instances,
    find_latent_mean_two_classes, positive_prediction_index)

if __name__ == '__main__':
    DATA_NAME = 'simple_bn'
    # DATA_NAME = 'moon'

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
    negative_instance_features = prediction_instances(
        features, negative_index)

    positive_index = positive_prediction_index(predictions)
    positive_instance_features = prediction_instances(
        features, positive_index)

    factual_sample = negative_instance_features[0:20, :]

    mean_z0, mean_z1 = find_latent_mean_two_classes(
        flow_model, negative_instance_features, positive_instance_features)

    counterfactual_instance = CounterfactualSimpleBn(
        predictive_model, flow_model, mean_z0, mean_z1)

    start = timeit.default_timer()
    cf_sample = counterfactual_instance.find_counterfactual_via_gradient_descent(
        factual_sample).cpu().detach().numpy()
    # cf_sample = counterfactual_instance.find_counterfactual_by_scaled_vector(
    #     factual_sample).cpu().detach().numpy()
    stop = timeit.default_timer()
    run_time = stop - start

    # Output
    factual_df = pd.DataFrame(factual_sample.cpu().detach(
    ).numpy(), columns=list(data_frame.columns)[:-1])
    counterfactual_df = pd.DataFrame(
        np.array(cf_sample), columns=list(data_frame.columns)[:-1])
    cf_sample = torch.Tensor(cf_sample).cuda()

    factual_df[target] = ''
    factual_df[target] = model_prediction(
        predictive_model, factual_sample).cpu().detach().numpy()

    counterfactual_df[target] = ''
    counterfactual_df[target] = model_prediction(
        predictive_model, cf_sample).cpu().detach().numpy()

    factual_df.to_csv(configuration_for_proj['result_' + DATA_NAME].format(
        "original_instance_flow.csv"), index=False)
    counterfactual_df.to_csv(configuration_for_proj['result_' + DATA_NAME].format(
        "cf_sample_flow.csv"), index=False)

    # Evaluation
    benchmark_instance = Benchmark(
        factual_df, counterfactual_df, run_time, counterfactual_df[target].values)
    benchmark_distance = benchmark_instance.compute_distances()
    benchmark_time = benchmark_instance.compute_average_time()
    benchmark_rate = benchmark_instance.compute_success_rate()
    print(benchmark_distance)
    print(benchmark_time)
    print(benchmark_rate)
    print(factual_df)
    print(counterfactual_df)
