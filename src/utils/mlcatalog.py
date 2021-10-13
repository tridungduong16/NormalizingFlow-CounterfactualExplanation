import torch
from data_catalog import DataCatalog, load_catalog


def load_pytorch_prediction_model_from_model_path(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

class MLModelCatalog():
    def __init__(self,data: DataCatalog, model_path: str) -> None:
        self._model = load_pytorch_prediction_model_from_model_path(model_path)
        self._continuous = data.continous
        self._categoricals = data.categoricals

    def predict(self):
        pass
    def predict_proba(self):
        pass

