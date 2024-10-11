import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.quantizer.uniform import *
from src.quantizer.nonuniform import *
#Normalized MSE initializer is originally proposed by UniQ
#Reference: https://github.com/phuocphn/uniq
# We developed it for LSQ and nuLSQ/UniQ as well
def NMSE_initializer(x, Qparms, Qn, Qp, quantizer, mode):
    dev = x.device
    Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(dev)
    if "init_scale" in Qparms.keys():
        scale = Qparms["init_scale"]
    else:
        scale = torch.tensor([0.0]).to(dev)

    if mode == "symmetric":
        delta_normal = {1: 0.9956866859435065, 3: 0.5860194414434872, 7: 0.33520061219993685, 15: 0.18813879027991698, 31: 0.10406300944201481, 63: 0.05686767238235839, 127: 0.03076238758025524}
        x_std = x.detach().std()
        
    elif mode == "asymmetric":
        delta_normal = { 3: 0.65076985, 7: 0.35340955, 15: 0.19324868, 31: 0.10548752, 63: 0.0572659, 127: 0.03087133, 255: 0.01652923}
        x_std = (2*((x.detach()**2).mean()))**0.5
    scale_tmp = x_std * delta_normal[Qp]
    if scale_tmp > scale:                
        Qparms["init_scale"] = scale_tmp
