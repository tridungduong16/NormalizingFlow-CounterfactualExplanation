import os
from abc import ABC, abstractmethod
from typing import Any, Dict
from typing import List

import pandas as pd
import yaml


def load_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_catalog(filename: str, dataset: str, keys: List[str]):
    with open(os.path.join(filename), "r") as f:
        catalog = yaml.safe_load(f)

    if dataset not in catalog:
        raise KeyError("Dataset not in catalog.")

    for key in keys:
        if key not in catalog[dataset].keys():
            raise KeyError("Important key {} is not in Catalog".format(key))
        if catalog[dataset][key] is None:
            catalog[dataset][key] = []

    return catalog[dataset]


class Data(ABC):
    """
    Abstract class to implement arbitrary datasets, which are provided by the user.
    """

    @property
    @abstractmethod
    def categoricals(self):
        """
        Provides the column names of categorical data.
        Column names do not contain encoded information as provided by a get_dummy() method (e.g., sex_female)

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all categorical columns
        """
        pass

    @property
    @abstractmethod
    def continous(self):
        """
        Provides the column names of continuous data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all continuous columns
        """
        pass

    @property
    @abstractmethod
    def immutables(self):
        """
        Provides the column names of immutable data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all immutable columns
        """
        pass

    @property
    @abstractmethod
    def target(self):
        """
        Provides the name of the label column.

        Returns
        -------
        str
            Target label name
        """
        pass

    @property
    @abstractmethod
    def raw(self):
        """
        The raw Dataframe without encoding or normalization

        Returns
        -------
        pd.DataFrame
            Tabular data with raw information
        """
        pass


class DataCatalog(Data):
    """
    Use already implemented datasets.

    Parameters
    ----------
    data_name : {'adult', 'compas', 'give_me_some_credit'}
        Used to get the correct dataset from online repository.

    Returns
    -------
    None
    """

    def __init__(self, data_name: str, data_path: str, configuration_path: str):
        self.name = data_name

        catalog_content = ["continous", "categorical", "immutable", "target"]
        self.catalog: Dict[str, Any] = load_catalog(  # type: ignore
            configuration_path, data_name, catalog_content
        )

        for key in ["continous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []

        self._raw: pd.DataFrame = load_dataset(data_path)

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


if __name__ == '__main__':
    data_name = 'adult'
    data_path = '/home/trduong/Data/fairCE/reports/results/original.csv'
    config_path = "/home/trduong/Data/fairCE/src/carla/data_catalog.yaml"
    dataset = DataCatalog(data_name, data_path, config_path)
    print(dataset.target)
    print(dataset.raw)
