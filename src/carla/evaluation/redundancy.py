from typing import List

import numpy as np
import pandas as pd
import torch
from carla.models.api import MLModel


def redundancy(
        factuals: pd.DataFrame, counterfactuals: pd.DataFrame, mlmodel: MLModel, predictive_model
) -> List[List[int]]:
    """
    Computes Redundancy measure for every counterfactual

    Parameters
    ----------
    factuals: Encoded and normalized factual samples
    counterfactuals: Encoded and normalized counterfactual samples
    mlmodel: Black-box-model we want to discover

    Returns
    -------
    List with redundancy values per counterfactual sample
    """

    df_enc_norm_fact = factuals.reset_index(drop=True)
    df_cfs = counterfactuals.reset_index(drop=True)

    labels = df_cfs[mlmodel.target]
    df_cfs = df_cfs.drop(mlmodel.target, axis=1)
    df_cfs["redundancy"] = df_cfs.apply(
        lambda x: compute_redundancy(
            df_enc_norm_fact.iloc[x.name].values, x.values, mlmodel, predictive_model, labels.iloc[x.name]
        ),
        axis=1,
    )
    return df_cfs["redundancy"].values.reshape((-1, 1)).tolist()


def compute_redundancy(
        fact: np.ndarray, cf: np.ndarray, mlmodel: MLModel, predictive_model, label_value: int
) -> int:
    red = 0

    # prediction = self._predictive_model(x_hat)
    # if torch.gt(prediction[0], 0.5):
    #     candidates.append(x_hat[0].detach())

    for col_idx in range(cf.shape[0]):  # input array has one-dimensional shape
        if fact[col_idx] != cf[col_idx]:
            temp_cf = np.copy(cf)
            temp_cf[col_idx] = fact[col_idx]
            temp_pred = np.argmax(predictive_model(torch.Tensor(temp_cf.reshape((1, -1))).cuda()).cpu().detach())
            if temp_pred == label_value:
                red += 1

    return red
