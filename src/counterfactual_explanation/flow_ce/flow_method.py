import timeit
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from counterfactual_explanation.models.classifier import Net
from counterfactual_explanation.utils.mlcatalog import (
    get_latent_representation_from_flow,
    original_space_value_from_latent_representation)
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
        self.distance_loss_func = torch.nn.MSELoss()
        self.predictive_loss_func = torch.nn.BCELoss()
        self.lr = 1e-5
        self.n_epochs = 1000

    @property
    def _flow_model(self):
        return self.flow_model

    @property
    def _predictive_model(self):
        return self.predictive_model

    def initialize_latent_representation(self):
        pass

    def distance_loss(self, representation_factual, representation_counterfactual):
        return self.distance_loss_func(representation_factual, representation_counterfactual)

    def prediction_loss(self, representation_counterfactual):
        counterfactual = self._original_space_value_from_latent_representation(representation_counterfactual)
        yhat = self._predictive_model(counterfactual).reshape(-1)
        yexpected = torch.ones(
            yhat.shape, dtype=torch.float).reshape(-1).cuda()
        self.predictive_loss_func(yhat, yexpected)
        return self.predictive_loss_func(yhat, yexpected)

    def fair_loss(self):
        return 0

    def combine_loss(self, factual, counterfactual):
        return self.prediction_loss(counterfactual) + self.distance_loss(factual, counterfactual)

    def make_perturb_loss(self, z_value, delta_value):
        return z_value + delta_value

    def _get_latent_representation_from_flow(self, input_value):
        return get_latent_representation_from_flow(self.flow_model, input_value)

    def _original_space_value_from_latent_representation(self, z_value):
        return original_space_value_from_latent_representation(self.flow_model, z_value)

    def find_counterfactual_via_iterations(self, factual):
        z_value = self._get_latent_representation_from_flow(factual)
        index_ = 0
        for _ in range(5000):
            index_ += 1
            delta_value = torch.rand(z_value.shape[1]).cuda()
            z_hat = self.make_perturb_loss(z_value, delta_value)
            x_hat = self._original_space_value_from_latent_representation(
                z_hat)
            prediction = self._predictive_model(x_hat)
            if torch.gt(prediction[0], 0.5):
                return x_hat[0]
            return x_hat[0]

    def find_counterfactual_via_gradient_descent(self, factual):
        z_value = self._get_latent_representation_from_flow(factual)
        # index_ = 0
        delta_value = nn.Parameter(torch.rand(z_value.shape[1]).cuda())
        representation_factual = self._get_latent_representation_from_flow(
            factual)

        z_hat = self.make_perturb_loss(z_value, delta_value)
        x_hat = self._original_space_value_from_latent_representation(
            z_hat)
        distance_loss = self.distance_loss(
            representation_factual, z_hat)

        prediction_loss = self.prediction_loss(x_hat)
        total_loss = distance_loss + prediction_loss

        optimizer = optim.Adam([delta_value], lr=0.00001)

        for epoch in tqdm(range(1000)):

            z_hat = self.make_perturb_loss(z_value, delta_value)
            x_hat = self._original_space_value_from_latent_representation(
                z_hat)


            # distance_loss = self.distance_loss(
            #     original_representation, z_hat)

            total_loss = self.combine_loss(representation_factual, z_hat)

            # prediction_loss = self.prediction_loss(x_hat)
            # total_loss = distance_loss + 7*prediction_loss

            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

            if epoch % 1000 == 0:
                print("\n Epoch {}, Loss {:.4f}".format(epoch, total_loss))


        z_hat = self.make_perturb_loss(z_value, delta_value)
        x_hat = self._original_space_value_from_latent_representation(
            z_hat)
        return x_hat[0]
