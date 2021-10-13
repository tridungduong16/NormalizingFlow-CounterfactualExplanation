from typing import Dict

import yaml
from carla import DataCatalog, MLModelCatalog
from carla.recourse_methods import GrowingSpheres, Clue, Dice, Face, Revise, Wachter
from utils.helpers import load_config, load_setup

"""Setup PATH"""
path1 = "/home/trduong/Data/fairCE/experimental_setup.yaml"
path2 = "/home/trduong/Data/fairCE/project_configurations.yaml"


# def load_setup(path, method, data_name) -> Dict:
#     with open(path, "r") as f:
#         setup_catalog = yaml.safe_load(f)
#     hyperparams = setup_catalog['recourse_methods'][method]["hyperparams"]
#     hyperparams["data_name"] = data_name
#     return hyperparams

def load_method(method, mlmodel, data, hyperparams=None):
    """

    :param method:
    :param mlmodel:
    :param data:
    :param hyperparams:
    :return:
    """

    if method == "clue":
        return Clue(data, mlmodel, hyperparams)
    elif method == "dice":
        return Dice(mlmodel, hyperparams)
    elif "face" in method:
        return Face(mlmodel, hyperparams)
    elif method == "gs":
        return GrowingSpheres(mlmodel)
    elif method == "revise":
        # variable input layer dimension is first time here available
        hyperparams["vae_params"]["layers"] = [
                                                  len(mlmodel.feature_input_order)
                                              ] + hyperparams["vae_params"]["layers"]
        return Revise(mlmodel, data, hyperparams)
    elif "wachter" in method:
        return Wachter(mlmodel, hyperparams)
    else:
        raise ValueError("Recourse method not known")


if __name__ == "__main__":
    """load a catalog dataset"""
    data_name = "adult"
    dataset = DataCatalog(data_name)

    """Load configuration"""
    configuration = load_config(path2)

    """load artificial neural network from catalog"""
    model = MLModelCatalog(dataset, "ann", backend="pytorch")

    """Load methods"""
    method = 'gs'
    gs = load_method(method, model, dataset)

    # method = "clue"
    # hyperparams = load_setup(path1, method, data_name)
    # cl = load_method(method, model, dataset, hyperparams)

    method = 'dice'
    hyperparams = load_setup(path1, method, data_name)
    dc = load_method(method, model, dataset, hyperparams)

    # method = 'revise'
    # hyperparams = load_setup(path1, method, data_name)
    # rv = load_method(method, model, dataset, hyperparams)
    # 
    # method = 'face'
    # hyperparams = load_setup(path1, "face_knn", data_name)
    # fc = load_method(method, model, dataset, hyperparams)

    # sys.exit(1)
    # method = 'wachter'
    # hyperparams = load_setup(path1, "wachter", data_name)
    # wt = load_method(method, model, dataset, hyperparams)

    """get factuals from the data to generate counterfactual examples"""
    factuals = dataset.raw.iloc[:10]
    # print(factuals)
    """generate counterfactual examples"""
    # counterfactuals_gs = gs.get_counterfactuals(factuals)
    counterfactuals_dc = dc.get_counterfactuals(factuals)
    # counterfactuals_rv = rv.get_counterfactuals(factuals)
    # counterfactuals_fc = wt.get_counterfactuals(factuals)
    # print("check")
    # print(factuals.values)
    # factuals = torch.tensor(factuals.values)
    # counterfactuals_wt = wt.get_counterfactuals(factuals)

    """save result"""
    # factuals.to_csv(configuration['result_path'].format('original.csv'), index=False)
    # counterfactuals_gs.to_csv(configuration['result_path'].format('gs.csv'), index=False)
    # counterfactuals_dc.to_csv(configuration['result_path'].format('dc.csv'), index=False)
    # counterfactuals_rv.to_csv(configuration['result_path'].format('rv.csv'), index=False)
    # counterfactuals_fc.to_csv(configuration['result_path'].format('fc.csv'), index=False)
    # counterfactuals_wt.to_csv(configuration['result_path'].format('wt.csv'), index=False)
