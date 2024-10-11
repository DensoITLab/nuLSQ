import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.quantizer.uniform import *
from src.quantizer.nonuniform import *

def LSQ_initializer(x, Qparms, Qn, Qp, quantizer, *args):
    dev = x.device
    Qp_on_device = torch.tensor(Qp, dtype=torch.float).to(dev)
    Qn_on_device = torch.tensor(Qn, dtype=torch.float).to(dev)

    if "init_scale" in Qparms.keys():
        scale = Qparms["init_scale"]
    else:
        scale = torch.tensor(0.0).to(dev)


    num_param_s = torch.numel(scale)

    scale     = torch.max(scale, 2 * x.detach().abs().mean() / math.sqrt(Qp_on_device))
    
    Qparms["init_scale"] = scale

