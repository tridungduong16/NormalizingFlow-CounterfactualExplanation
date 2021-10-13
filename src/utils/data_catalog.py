from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from helpers import load_target_features_name
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset


class Data(ABC):
    @property
    @abstractmethod
    def categoricals(self):
        pass

    @property
    @abstractmethod
    def continous(self):
        pass

    @property
    @abstractmethod
    def immutables(self):
        pass

    @property
    @abstractmethod
    def target(self):
        pass

    @property
    @abstractmethod
    def raw(self):
        pass


class DataCatalog(Data):
    def __init__(self, data_name: str, data_path: str, configuration_path: str):
        self.name = data_name

        catalog_content = ["continous", "categorical", "immutable", "target"]
        self.catalog = load_target_features_name(configuration_path, data_name, catalog_content)

        for key in ["continous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []

        self._raw = pd.read_csv(data_path)

    @property
    def categoricals(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continous(self) -> List[str]:
        return self.catalog["continous"]

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        return self.catalog["target"]

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy()


class EncoderNormalizeDataCatalog():
    def __init__(self, data: DataCatalog):
        self.data_frame = data._raw
        self.continous = data.continous
        self.categoricals = data.categoricals
        self.scaler = StandardScaler()
        self.target = data.target

        self.encoder = OneHotEncoder(sparse=False)
        self.normalize_continuous_feature()
        self.convert_to_one_hot_encoding_form()
        self.encoded_feature_name = ""

    def normalize_continuous_feature(self):
        self.data_frame[self.continous] = self.scaler.fit_transform(self.data_frame[self.continous])

    def convert_to_one_hot_encoding_form(self):
        encoded_data_frame = self.encoder.fit_transform(self.data_frame[self.categoricals])
        column_name = self.encoder.get_feature_names(self.categoricals)
        self.data_frame[column_name] = pd.DataFrame(encoded_data_frame, columns=column_name)
        self.data_frame = self.data_frame.drop(self.categoricals, axis=1)
        self.encoded_feature_name = column_name

    def convert_from_one_hot_to_original_forms(self):
        pass

    def order_data(self, feature_order) -> pd.DataFrame:
        return self.data_frame[feature_order]


class TensorDatasetTraning(Dataset):

    def __init__(self, feature, target):
        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        image = self.feature[index]
        label = self.target[index]
        return image, label


