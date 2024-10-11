import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def Const_initializer(x, Qparms, Qn, Qp, quantizer, mode):
    dev = x.device

    if "init_scale" in Qparms.keys():
        scale = Qparms["init_scale"]
    else:
        scale = torch.tensor(0.0).to(dev)

    num_param_s = torch.numel(scale)


    if mode == "symmetric":
        alpha = torch.tensor(3.0).to(dev)
        
    elif mode == "asymmetric":
        alpha = torch.tensor(8.0).to(dev)

    Qparms["init_scale"] = alpha/Qp


