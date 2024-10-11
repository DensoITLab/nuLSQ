
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.quantizer.uniform import *
from src.quantizer.nonuniform import *
from src.initializer import *


#----------------------------------------------------------
# Fully connected layer 
#----------------------------------------------------------
class QLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, \
        num_bits=None, w_grad_scale_mode = "LSQ_grad_scale", \
        x_grad_scale_mode = "LSQ_grad_scale",\
        weight_norm = False, w_quantizer = None, x_quantizer = None, \
        w_initializer = None, x_initializer = None, first_layer = False):

        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.num_bits   = num_bits
        self.x_grad_scale_mode = x_grad_scale_mode
        self.w_grad_scale_mode = w_grad_scale_mode
        self.weight_norm = weight_norm
        self.first_layer = first_layer
        self.x_Qparms={}
        self.w_Qparms={}
        self.w_quantizer= w_quantizer(self, num_bits, "weight") if w_quantizer != None else None
        self.x_quantizer= x_quantizer(self, num_bits, "activation") if x_quantizer != None else None
        self.w_initializer= w_initializer
        self.x_initializer= x_initializer
        self.register_buffer('init_state', torch.tensor(False))   

    def forward(self, input):
        return _forward_common(self, input)


#----------------------------------------------------------
# convolutional layer
#----------------------------------------------------------
class QConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, \
        stride=1, padding=0, dilation=1, groups=1, bias=True, \
        num_bits=None, \
        w_grad_scale_mode = "LSQ_grad_scale", \
        x_grad_scale_mode = "LSQ_grad_scale",  \
        weight_norm = None, w_quantizer = None, x_quantizer = None, \
        w_initializer = None, x_initializer = None, first_layer = False):

        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.num_bits   = num_bits
        self.x_grad_scale_mode = x_grad_scale_mode
        self.w_grad_scale_mode = w_grad_scale_mode
        self.weight_norm = weight_norm
        self.first_layer = first_layer
        self.x_Qparms={}
        self.w_Qparms={}
        self.w_quantizer= w_quantizer(self, num_bits, "weight") if w_quantizer != None else None
        self.x_quantizer= x_quantizer(self, num_bits, "activation") if x_quantizer != None else None
        self.w_initializer= w_initializer
        self.x_initializer= x_initializer
        self.register_buffer('init_state', torch.tensor(False))   

    def forward(self, input):
        return _forward_common(self, input)


def _forward_common(module, input):
    if module.weight_norm in ["WN", "LWN"]:
        mean = module.weight.mean()
        std = module.weight.std()
        weight = module.weight.add(-mean).div(std)
    else :
        weight = module.weight
    if module.init_state == False:
        if module.x_quantizer != None and module.x_initializer != None:
            Qparms_to_dev(input, module.x_Qparms)
            if module.first_layer:                
                module.x_initializer(input, module.x_Qparms, module.x_Qn, module.x_Qp, module.x_quantizer, "symmetric")
            else:
                module.x_initializer(input, module.x_Qparms, module.x_Qn, module.x_Qp, module.x_quantizer, "asymmetric")
            module.x_quantizer.scale_to_Qparms(module.x_Qparms, module.x_Qn, module.x_Qp)

        if module.w_quantizer != None and module.w_initializer != None:                
            Qparms_to_dev(weight, module.w_Qparms)
            module.w_initializer(weight, module.w_Qparms, module.w_Qn, module.w_Qp, module.w_quantizer, "symmetric")            
            module.w_quantizer.scale_to_Qparms(module.w_Qparms, module.w_Qn, module.w_Qp)

    if module.x_quantizer != None:
        input  = module.x_quantizer(input,         module.x_Qparms, module.x_Qn, module.x_Qp, input.shape[1], module.x_grad_scale_mode)
    if module.w_quantizer != None:
        weight = module.w_quantizer(weight, module.w_Qparms, module.w_Qn, module.w_Qp, weight.numel(), module.w_grad_scale_mode)

    if module.weight_norm in ["WN", "LWN"]:
        weight = weight.mul(std)

    # compute forward pass
    if   module.__class__.__name__ in ["QLinear"]:
        out = F.linear(input, weight, module.bias)
    elif module.__class__.__name__ in ["QConv2d"]:
        out = F.conv2d(input, weight, module.bias, module.stride, module.padding, module.dilation, module.groups)

    return out

def Qparms_to_dev(x, Qparms):
    dev = x.device
    for iname in Qparms.keys():
        Qparms[iname].data = Qparms[iname].data.to(dev)
    