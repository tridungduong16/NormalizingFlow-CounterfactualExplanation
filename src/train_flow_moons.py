import torch
import numpy as np 

from flow_ssl import FlowLoss
from flow_ssl.data import make_moons_ssl
from flow_ssl.distributions import SSLGaussMixture
from flow_ssl.realnvp.realnvp import RealNVPTabular
from utils.mlcatalog import save_pytorch_model_to_model_path

if __name__ == "__main__":
    LR_INIT = 1e-4
    EPOCHS = 2001
    BATCH_SIZE = 32
    PRINT_FREQ = 500
    MEAN_VALUE = 3.5
    means = torch.tensor([[-MEAN_VALUE, -MEAN_VALUE], [MEAN_VALUE, MEAN_VALUE]])
    prior = SSLGaussMixture(means=means)
    data, labels = make_moons_ssl()
    flow = RealNVPTabular(num_coupling_layers=5, in_dim=2, num_layers=1, hidden_dim=512)
    loss_fn = FlowLoss(prior)

    n_ul = np.sum(labels == -1)
    n_l = np.shape(labels)[0] - n_ul

    labeled_data = data[labels != -1]
    labeled_labels = labels[labels != -1]
    unlabeled_data = data[labels == -1]
    unlabeled_labels = labels[labels == -1]

    optimizer = torch.optim.Adam(flow.parameters(), lr=LR_INIT, weight_decay=1e-2)
    for t in range(EPOCHS):
        batch_idx = np.random.choice(n_ul, size=BATCH_SIZE)
        batch_x, batch_y = unlabeled_data[batch_idx], unlabeled_labels[batch_idx]
                
        batch_x = np.vstack([batch_x, labeled_data])
        batch_y = np.hstack([batch_y, labeled_labels])
        batch_x, batch_y = torch.from_numpy(batch_x), torch.from_numpy(batch_y)
        z = flow(batch_x)
        sldj = flow.logdet()
        loss = loss_fn(z, sldj, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % PRINT_FREQ == 0:
            print('iter %s:' % t, 'loss = %.3f' % loss)
        if t == int(EPOCHS * 0.5) or t == int(EPOCHS * 0.8):
            for p in optimizer.param_groups:
                p["lr"] /= 10
    
    MODEL_PATH = "models/moon_flow.pth"
    save_pytorch_model_to_model_path(flow, MODEL_PATH)