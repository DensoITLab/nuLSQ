import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#----------------------------------------------------------
# LSQ
#----------------------------------------------------------
class MinMax_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(MinMax_quantizer, self).__init__()
        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            if module.first_layer:
                module.x_Qn = 2 ** (num_bits-1) - 1
                module.x_Qp = 2 ** (num_bits-1) - 1
                self.is_weight = True
            else:
                module.x_Qn = 0
                module.x_Qp = 2 ** (num_bits  ) - 1
                self.is_weight = False
        elif mode == "weight":
            module.w_Qn = 2 ** (num_bits-1) - 1
            module.w_Qp = 2 ** (num_bits-1) - 1
            self.is_weight = True

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):

        y, x_max = _MinMaxFixed_quantizer(x, Qn, Qp)
        if self.is_weight:
            self.w_min = - x_max
            self.w_max = x_max
        return y
 
    def calculate_min_max(self, *args):
        
        return self.w_min, self.w_max     

def _MinMaxFixed_quantizer(x, Qn, Qp):

    Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
    Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
    # print(scale)
    
    max = x.max()
    x_q = x.div(max).mul(Qp_on_device).round().div(Qp).mul(max)
    y = (x_q - x).detach() + x

    return  y,  max

