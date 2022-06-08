import sys
from stochastic_spiking.neural_models.lsnn import *

thismodule = sys.modules[__name__]

def net(net_name='LSNN'):
    net_model = getattr(thismodule, net_name)
    return net_model
