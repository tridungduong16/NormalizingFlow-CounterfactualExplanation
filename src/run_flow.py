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
    model_prediction, negative_prediction_index, negative_prediction_instances)

if __name__ == '__main__':
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

    factual_sample = negative_instance_features[0:10, :]
    counterfactual_instance = CounterfactualSimpleBn(
        predictive_model, flow_model)

    start = timeit.default_timer()
    # cf_sample = counterfactual_instance.optimize_loss(factual_sample)
    # cf_sample = counterfactual_instance.simple_find_counterfactual_via_iterations(factual_sample)
    x = factual_sample[0:1,:]
    z = counterfactual_instance._get_latent_representation_from_flow(x)
    x_hat = counterfactual_instance._original_space_value_from_latent_representation(z)



    cf_sample = []
    for i in tqdm(factual_sample):
        # term = counterfactual_instance.find_counterfactual_via_iterations(
        #     i.reshape(1, -1))
        term = counterfactual_instance.find_counterfactual_via_gradient_descent(
            i.reshape(1, -1))
        cf_sample.append(term.cpu().detach().numpy())

    print(cf_sample)

    stop = timeit.default_timer()
    run_time = stop - start

    factual_df = pd.DataFrame(factual_sample.cpu().detach(
    ).numpy(), columns=list(data_frame.columns)[:-1])
    # counterfactual_df = pd.DataFrame(cf_sample.cpu().detach().numpy(), columns=list(data_frame.columns)[:-1])
    counterfactual_df = pd.DataFrame(
        np.array(cf_sample), columns=list(data_frame.columns)[:-1])

    print(counterfactual_df)
    cf_sample = torch.Tensor(counterfactual_df.values).cuda()
    factual_df[target] = ''
    factual_df[target] = model_prediction(
        predictive_model, factual_sample).cpu().detach().numpy()

    counterfactual_df[target] = ''
    counterfactual_df[target] = model_prediction(
        predictive_model, cf_sample).cpu().detach().numpy()

    factual_df.to_csv(configuration_for_proj['result_simple_bn'].format(
        "original_instance_flow.csv"), index=False)
    counterfactual_df.to_csv(configuration_for_proj['result_simple_bn'].format(
        "cf_sample_flow.csv"), index=False)

    benchmark_instance = Benchmark(
        factual_df, counterfactual_df, run_time, counterfactual_df[target].values)
    benchmark_distance = benchmark_instance.compute_distances()
    benchmark_time = benchmark_instance.compute_average_time()
    benchmark_rate = benchmark_instance.compute_success_rate()
    print(benchmark_distance)
    print(benchmark_time)
    print(benchmark_rate)
    print(counterfactual_df)
