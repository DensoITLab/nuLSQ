import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#----------------------------------------------------------
# LSQ
#----------------------------------------------------------
class LSQ_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(LSQ_quantizer, self).__init__()
        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            if module.first_layer:
                module.x_Qn = 2 ** (num_bits-1)
                module.x_Qp = 2 ** (num_bits-1) - 1
            else:
                module.x_Qn = 0
                module.x_Qp = 2 ** (num_bits  ) - 1
            module.x_scale = nn.Parameter(torch.tensor([0.0]))
            module.x_Qparms['scale'] = module.x_scale
        elif mode == "weight":
            module.w_Qn = 2 ** (num_bits-1)
            module.w_Qp = 2 ** (num_bits-1) - 1
            module.w_scale = nn.Parameter(torch.tensor([0.0]))
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        scale = Qparms['scale']
        y = _LSQ_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode)
        return y
    
    def scale_to_Qparms(self, Qparms, Qn, Qp):
        Qparms["scale"].data = torch.full(Qparms["scale"].size(), Qparms["init_scale"].clone().detach(), device=Qparms["scale"].device)

def _LSQ_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode):

    Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
    Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
    # print(scale)
    assert scale > 0, 'scale = {}, {}, {}'.format(scale, Qn_on_device, Qp_on_device)
    
    # gradient scaling
    if num_elements > 0:
        if grad_scale_mode == "10_fac":
            grad_scale = torch.tensor(10.0).to(x.device)
        elif grad_scale_mode == "LSQ_grad_scale":
            grad_scale = 1.0 / torch.sqrt(num_elements * Qp_on_device) 
        else:
            grad_scale = torch.tensor(1.0).to(x.device)
            
        bw_scale   = scale * grad_scale
                
        scale      = (scale - bw_scale).detach() + bw_scale
    x  = x / scale
    
    x  = torch.min(torch.max(x, -Qn_on_device), Qp_on_device)
    xq = torch.round(x)
    y  = (xq - x).detach() + x
    
    y  = scale * y

    return  y

