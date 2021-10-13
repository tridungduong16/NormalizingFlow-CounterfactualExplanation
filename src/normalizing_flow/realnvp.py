import torch.nn as nn

from coupling_layer import CouplingLayerTabular
from coupling_layer import MaskTabular

from invertible import iSequential

class RealNVPBase(nn.Module):

    def forward(self,x_value):
        return self.body(x_value)

    def logdet(self):
        return self.body.logdet()

    def inverse(self,z_value):
        return self.body.inverse(z_value)

    def nll(self,x_value,y_value=None,label_weight=1.):
        z_value = self(x_value)
        logdet = self.logdet()
        z_value = z_value.reshape((z_value.shape[0], -1))
        prior_ll = self.prior.log_prob(z_value, y_value,label_weight=label_weight)
        nll = -(prior_ll + logdet)
        return nll

class RealNVPTabular(RealNVPBase):
    def __init__(self, in_dim=2, num_coupling_layers=6, hidden_dim=256, num_layers=2, init_zeros=False,dropout=False):
        super(RealNVPTabular, self).__init__()
        self.body = iSequential(*[
                        CouplingLayerTabular(in_dim, hidden_dim, num_layers, MaskTabular(reverse_mask=bool(i%2)),
                            init_zeros=init_zeros,dropout=dropout)
                        for i in range(num_coupling_layers)
                    ])
