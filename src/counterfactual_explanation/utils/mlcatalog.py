import torch
from .data_catalog import DataCatalog

def load_pytorch_prediction_model_from_model_path(model_path):
    # print("Load model")
    model = torch.load(model_path)
    model.eval()
    # print("End load")
    return model


def load_pytorch_prediction_model_with_state_dict(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_pytorch_model_to_model_path(model, model_path):
    torch.save(model, model_path)


def get_latent_representation_from_flow(flow, input_value):
    return flow(input_value)


def original_space_value_from_latent_representation(flow, z_value):
    # return flow.inverse(torch.from_numpy(z_value).float()).detach().numpy()
    return flow.inverse(z_value)

def model_prediction(predictive_model, features):
    return predictive_model(features)

def negative_prediction_index(prediction):
    return torch.lt(prediction, 0.5).reshape(-1)

def positive_prediction_index(prediction):
    return torch.gt(prediction, 0.5).reshape(-1)


def prediction_instances(instances, indexes):
    return instances[indexes]


class MLModelCatalog():
    def __init__(self,data: DataCatalog, predictive_model) -> None:
        self.model = predictive_model 
        self._continuous = data.continous
        self._categoricals = data.categoricals

    def predict(self):
        pass
    def predict_proba(self):
        pass


def train_one_epoch_batch_data(flow_model, optimizer, loss_fn, batch_x, batch_y):
    z_value = flow_model(batch_x)
    sldj = flow_model.logdet()
    loss = loss_fn(z_value, sldj, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss