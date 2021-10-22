from abc import ABC, abstractmethod

import numpy as np
import pandas as pd 
import torch
import torch.nn as nn 
import torch.optim as optim 

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

class FindCounterfactualSample(ABC):
    @abstractmethod
    def initialize_latent_representation(self):
        pass

    @abstractmethod
    def distance_loss(self):
        pass

    @abstractmethod
    def prediction_loss(self):
        pass

    @abstractmethod
    def fair_loss(self):
        pass


class CounterfactualSimpleBn(FindCounterfactualSample):
    def __init__(self, predictive_model, flow_model):
        # self.original_instance = original_instance
        self.flow_model = flow_model
        self.predictive_model = predictive_model
        self.distance_loss_func = torch.nn.SmoothL1Loss()
        self.predictive_loss_func = torch.nn.BCELoss()


    @property
    def _flow_model(self):
        return self.flow_model

    @property
    def _predictive_model(self):
        return self.predictive_model
    
    def initialize_latent_representation(self):
        pass 

    def distance_loss(self, x, y):
        return self.distance_loss_func(x, y)

    def prediction_loss(self, x):
        yhat = self._predictive_model(x).reshape(-1)
        yexpected = torch.ones(yhat.shape, dtype=torch.float).reshape(-1).cuda()
        self.predictive_loss_func(yhat, yexpected)
        return self.predictive_loss_func(yhat, yexpected)

    def fair_loss(self):
        return 0 

    def combine_loss(self):
        return self.prediction_loss() + self.distance_loss() + self.fair_loss()

    def make_perturb_loss(self, z_value, delta_value):
        return z_value + delta_value

    def _get_latent_representation_from_flow(self, input_value):
        return get_latent_representation_from_flow(self.flow_model, input_value)
    
    def _original_space_value_from_latent_representation(self, z_value):
        return original_space_value_from_latent_representation(self.flow_model, z_value)

    def simple_find_counterfactual_via_iterations(self, original_instance):
        z_value = self._get_latent_representation_from_flow(original_instance)
        index_ = 0
        while True:
            index_ += 1
            delta_value = torch.rand(z_value.shape[1]).cuda()
            z_hat = self.make_perturb_loss(z_value, delta_value)
            x_hat = self._original_space_value_from_latent_representation(z_hat)
            prediction = self._predictive_model(x_hat)
            if torch.gt(prediction, 0.5)[0]:
                return delta_value
              
    def optimize_loss(self, original_instance):
        original_representation = self._get_latent_representation_from_flow(original_instance)
        counterfactual_representation = nn.Parameter(torch.rand(original_representation.shape).cuda())
        counterfactual_sample = self._original_space_value_from_latent_representation(counterfactual_representation)

        distance = self.distance_loss(original_representation, counterfactual_representation)
        prediction = self.prediction_loss(counterfactual_sample)

        total_loss = distance + prediction
        optimizer = optim.Adam([counterfactual_representation])
        
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        return self._original_space_value_from_latent_representation(counterfactual_representation) 


if __name__ == '__main__':
    DATA_NAME = 'simple_bn'
    # CONFIG_PATH = '/home/trduong/Data/fairCE/configuration/data_catalog.yaml'
    # CONFIG_FOR_PROJECT = '/home/trduong/Data/fairCE/configuration/project_configurations.yaml'
    # configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    # DATA_PATH = configuration_for_proj[DATA_NAME + '_dataset']
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    # encoder_normalize_data_catalog = EncoderNormalizeDataCatalog(data_catalog)
    # data_frame = encoder_normalize_data_catalog.data_frame
    # target = encoder_normalize_data_catalog.target

    # predictive_model_path = configuration_for_proj['predictive_model_' + DATA_NAME]
    # flow_model_path = configuration_for_proj['flow_model_' + DATA_NAME]
    
    # predictive_model = load_pytorch_prediction_model_from_model_path(predictive_model_path)
    # predictive_model = predictive_model.to(DEVICE)

    # features = data_frame.drop(columns = [target], axis = 1).values.astype(np.float32)
    # features = torch.Tensor(features)
    # features = features.to(DEVICE)

    predictive_model, flow_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(DATA_NAME)

    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    features = data_frame.drop(columns = [target], axis = 1).values.astype(np.float32)
    features = torch.Tensor(features)
    features = features.cuda()

    # data_frame = encoder_normalize_data_catalog.data_frame
    # target = encoder_normalize_data_catalog.target
    

    ### 
    predictions = model_prediction(predictive_model, features)
    negative_index = negative_prediction_index(predictions)
    negative_instance_features = negative_prediction_instances(features, negative_index)

    factual_sample = negative_instance_features[0:10, :]
    counterfactual_instance = CounterfactualSimpleBn(predictive_model, flow_model)
    cf_sample = counterfactual_instance.optimize_loss(factual_sample)

    factual_df = pd.DataFrame(factual_sample.cpu().detach().numpy(), columns=list(data_frame.columns)[:-1])
    counterfactual_df = pd.DataFrame(cf_sample.cpu().detach().numpy(), columns=list(data_frame.columns)[:-1])
    
    # print(model_prediction(predictive_model, features))
    # print(model_prediction(predictive_model, cf_sample))
    factual_df[target] = ''
    factual_df[target] = model_prediction(predictive_model, factual_sample).cpu().detach().numpy()

    counterfactual_df[target] = ''
    counterfactual_df[target] = model_prediction(predictive_model, cf_sample).cpu().detach().numpy()



    factual_df.to_csv(configuration_for_proj['result_simple_bn'].format("original_instance.csv"), index=False)
    counterfactual_df.to_csv(configuration_for_proj['result_simple_bn'].format("cf_sample.csv"), index=False)
    