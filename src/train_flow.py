import numpy as np
import torch

from flow_ssl import FlowLoss
from flow_ssl.data import make_moons_ssl
from flow_ssl.distributions import SSLGaussMixture
from flow_ssl.realnvp.realnvp import RealNVPTabular
from utils.data_catalog import (DataCatalog, EncoderNormalizeDataCatalog,
                                TensorDatasetTraning)
from utils.helpers import load_configuration_from_yaml
from utils.mlcatalog import (save_pytorch_model_to_model_path,
                             train_one_epoch_batch_data)
from torch.utils.data import DataLoader

if __name__ == "__main__":
    DATA_NAME = 'simple_bn'
    CONFIG_PATH = '/home/trduong/Data/fairCE/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/home/trduong/Data/fairCE/configuration/project_configurations.yaml'
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_dataset']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    encoder_normalize_data_catalog = EncoderNormalizeDataCatalog(data_catalog)
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    feature_names = encoder_normalize_data_catalog.categoricals + encoder_normalize_data_catalog.continous

    LR_INIT = 1e-4
    EPOCHS = 2001
    BATCH_SIZE = 32
    PRINT_FREQ = 10
    MEAN_VALUE = 0.5

    x1_mean = data_frame['x1'].median()
    x2_mean = data_frame['x2'].median()
    x3_mean = data_frame['x3'].median()

    means = torch.tensor([
        np.array([x1_mean, x2_mean, x3_mean]).astype(np.float32)
        ])

    prior = SSLGaussMixture(means=means)
    features = data_frame[feature_names].values.astype(np.float32)
    labels = data_frame[target].values.astype(np.float32)

    features = torch.Tensor(features)
    labels = torch.Tensor(labels).reshape(-1,1)
    train_data = torch.hstack((features,labels))
    train_data = TensorDatasetTraning(train_data)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    flow = RealNVPTabular(num_coupling_layers=5, in_dim=3, num_layers=1, hidden_dim=512)
    loss_fn = FlowLoss(prior)

    
    optimizer = torch.optim.Adam(flow.parameters(), lr=LR_INIT, weight_decay=1e-2)
    for t in range(EPOCHS):
        for local_batch, local_labels in train_loader:
            z = flow(local_batch)
            sldj = flow.logdet()
            loss = loss_fn(z, sldj, local_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if t % PRINT_FREQ == 0:
            print('iter %s:' % t, 'loss = %.3f' % loss)
    
    save_pytorch_model_to_model_path(flow, configuration_for_proj['flow_model_' + DATA_NAME])
