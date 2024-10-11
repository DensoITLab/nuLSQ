import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#----------------------------------------------------------
# nuLSQ (fastest version)
#----------------------------------------------------------
class Positive_nuLSQ_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(Positive_nuLSQ_quantizer, self).__init__()
        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            module.x_Qn = 0
            module.x_Qp = 2 ** (num_bits  ) - 1
            module.x_scale = nn.Parameter(torch.tensor([0.0] * module.x_Qp))
            module.register_buffer('Ns_x',torch.tensor([0.0] * module.x_Qp))
            module.x_Qparms['scale'] = module.x_scale
        elif mode == "weight":
            raise NotImplementedError('Not applicable for weight')
        
    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode,  box_size=1024*256, counting_num = False):
        scale = Qparms['scale']
        y = _Positive_nuLSQ_fastest_quantizer.apply(x, scale, Qn, Qp, num_elements, grad_scale_mode,  box_size, counting_num)
        return y

    def calculate_min_max(self, Qparms, Qn, Qp):
        w_scale = Qparms["scale"]
        w_min = 0  
        w_max = w_scale.sum()
        return w_min, w_max     

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        if Qparms["init_scale"].size() == Qparms["scale"].size():
            Qparms["scale"].data = Qparms["init_scale"].clone().detach()
        else:
            Qparms["scale"].data = torch.full(Qparms["scale"].size(), Qparms["init_scale"].clone().detach(), device=Qparms["scale"].device)

class _Positive_nuLSQ_fastest_quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, Qn, Qp, num_elements,  x_grad_scale_mode = "none",  box_size=1024*256, counting_num = False):

        device = x.device
        Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
        Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
        num_levels = Qp + 1
        for s in scale:
            assert s >= 0, 'scale = {}, num_levels={}, num_elements={}, box_size={}, gradscale_mode ={}'\
            .format(scale, num_levels, num_elements, box_size, x_grad_scale_mode)


        if num_elements > 0:
            if x_grad_scale_mode == "LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            elif x_grad_scale_mode == "10fac_LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / (10*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            else:
                grad_scale = torch.tensor(1.0).to(x.device)

        cumsum_scale = torch.cumsum(scale, dim=0)

        idx1         = torch.searchsorted(cumsum_scale-scale/2, x)
        idx2         = torch.searchsorted(cumsum_scale        , x)

        y            = torch.cat([torch.tensor([0]).to(device), cumsum_scale])[idx1]



        ratio = torch.ones_like(scale) 
        ctx.save_for_backward(x, scale, grad_scale, cumsum_scale, idx2, torch.tensor(box_size), ratio)

        return y

    @staticmethod
    def backward(ctx, dLdy):
        x, scale, grad_scale, cumsum_scale, idx, box_size, ratio = ctx.saved_tensors        
        device = x.device

        flag_high    = x > scale.sum()
        flag_low     = x < 0
        flag_outside = flag_high | flag_low
        dLdx         = torch.where(flag_outside, torch.tensor(0, dtype=x.dtype, device = device), dLdy)

        shift_x   = x - torch.cat([torch.tensor([0]).to(device), cumsum_scale])[idx]
        dev_scale = (torch.cat([-1/scale, torch.tensor([0]).to(device)])) [idx]
        d         = shift_x * dev_scale

        rounded_d = d - torch.round(d)

        dyds_body = torch.where(flag_outside, torch.tensor(0, dtype=x.dtype, device = device), rounded_d)

        box_size = int(box_size) # magic parameter for parallelize scatter_add_

        num_levels = scale.numel() + 1
        N          = int(box_size / num_levels)
        M_residual = int(x.numel() % box_size)
        M_parallel = x.numel() - M_residual

        idx1, idx2 = torch.split(               idx.view(-1), (M_parallel, M_residual))
        val1, val2 = torch.split((dLdy * dyds_body).view(-1), (M_parallel, M_residual))

        dLds1 = torch.zeros((num_levels, N)).to(device)
        idx1  = idx1.view(-1, N)
        val1  = val1.view(-1, N)
        dLds1.scatter_add_(0, idx1, val1)
        dLds1 = dLds1.sum(1).view(-1)
        dLds1 = dLds1[:-1]

        dLds2 = torch.zeros(num_levels).to(device)
        if M_residual > 0:
            dLds2.scatter_add_(0, idx2, val2)
        dLds2 = dLds2[:-1]

        dLds  = (dLds1 + dLds2)*grad_scale
        dLds += torch.where(flag_high, dLdy, torch.tensor(0, dtype=x.dtype, device = device)).sum().view(-1)*grad_scale

        return dLdx, dLds, None, None, None, None, None, None

#----------------------------------------------------------
# LSQ (non uniform version for weight)
# Note order of scale: (negative: small to large magnitude, positive: small to large magnitude)
#----------------------------------------------------------
class Symmetric_nuLSQ_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(Symmetric_nuLSQ_quantizer, self).__init__()
        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            module.x_Qn = 2 ** (num_bits-1)
            module.x_Qp = 2 ** (num_bits-1) - 1
            module.x_scale = nn.Parameter(torch.tensor([0.0]* (module.x_Qp + module.x_Qn)))
            module.x_Qparms['scale'] = module.x_scale

        elif mode == "weight":
            module.w_Qn = 2 ** (num_bits-1)
            module.w_Qp = 2 ** (num_bits-1) - 1
            module.w_scale = nn.Parameter(torch.tensor([0.0]* (module.w_Qp + module.w_Qn)))
            module.w_Qparms['scale'] = module.w_scale
    
    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode = "none", box_size=1024*256, counting_num = False):
        scale = Qparms['scale']
        return _Symmetric_nuLSQ_fastest_quantizer.apply(x, scale, Qn, Qp, num_elements, grad_scale_mode, box_size, counting_num)

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        if Qparms["init_scale"].size() == Qparms["scale"].size():
            Qparms["scale"].data = Qparms["init_scale"].clone().detach()
        else:
            Qparms["scale"].data = torch.full(Qparms["scale"].size(), Qparms["init_scale"].clone().detach(), device=Qparms["scale"].device)



class _Symmetric_nuLSQ_fastest_quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, Qn, Qp, num_elements,  grad_scale_mode, box_size = 1024*256, counting_num = False):

        device = x.device
        Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
        Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
        neg_scale = scale[:Qn]
        pos_scale = scale[Qn:]
        for neg_s in neg_scale:
            assert neg_s > 0, 'scale in negative region = {}'.format(neg_scale)
        for pos_s in pos_scale:
            assert pos_s > 0, 'scale in positive region = {}'.format(pos_scale)

        if num_elements > 0:
            if grad_scale_mode == "wo_grad_scale":
                grad_scale_neg = torch.tensor(1.0).to(x.device)
                grad_scale_pos = torch.tensor(1.0).to(x.device)
            elif grad_scale_mode == "LSQ_grad_scale":
                grad_scale_neg = 2.0 / torch.sqrt(num_elements * Qn_on_device) 
                grad_scale_pos = 2.0 / torch.sqrt(num_elements * Qp_on_device) 
            else:
                grad_scale_neg = torch.tensor(1.0).to(x.device)
                grad_scale_pos = torch.tensor(1.0).to(x.device)

        y = torch.zeros_like(x)

        cumsum_pos_scale = torch.cumsum(pos_scale, dim=0)
        cumsum_neg_scale = torch.cumsum(neg_scale, dim=0)

        idx1_pos         = torch.searchsorted(cumsum_pos_scale-pos_scale/2, x)
        idx2_pos         = torch.searchsorted(cumsum_pos_scale        , x)

        y            = torch.cat([torch.tensor([0]).to(device), cumsum_pos_scale])[idx1_pos]

        idx1_neg         = torch.searchsorted(cumsum_neg_scale-neg_scale/2, -x)
        idx2_neg         = torch.searchsorted(cumsum_neg_scale        , -x)
        y            = y + torch.cat([torch.tensor([0]).to(device), -cumsum_neg_scale])[idx1_neg]

        ctx.save_for_backward(x, neg_scale, pos_scale, grad_scale_neg, grad_scale_pos, cumsum_neg_scale, cumsum_pos_scale, idx2_neg, idx2_pos, torch.tensor(box_size))

        return y

    @staticmethod
    def backward(ctx, dLdy):
        x, neg_scale, pos_scale, grad_scale_neg, grad_scale_pos, cumsum_neg_scale, cumsum_pos_scale, idx_neg, idx_pos, box_size = ctx.saved_tensors
        device = x.device

        flag_high   = x > pos_scale.sum()
        flag_neg = x < 0
        flag_low    = x < -neg_scale.sum()
        flag_pos_outside = flag_high | flag_neg
        flag_neg_outside = flag_high | (~flag_neg)
        flag_outside = flag_high | flag_low
        dLdx         = torch.where(flag_outside, torch.tensor(0, dtype=x.dtype, device = device), dLdy)        

        # derivative in positive region
        shift_x   = x - torch.cat([torch.tensor([0]).to(device), cumsum_pos_scale])[idx_pos]
        dev_scale = (torch.cat([-1/pos_scale, torch.tensor([0]).to(device)])) [idx_pos]
        d         = shift_x * dev_scale

        rounded_d = d - torch.round(d)

        dyds_body = torch.where(flag_pos_outside, torch.tensor(0, dtype=x.dtype, device = device), rounded_d)

        box_size = int(box_size) # magic parameter for parallelize scatter_add_

        num_levels = pos_scale.numel() + 1
        N          = int(box_size / num_levels)
        M_residual = int(x.numel() % box_size)
        M_parallel = x.numel() - M_residual

        idx1, idx2 = torch.split(               idx_pos.view(-1), (M_parallel, M_residual))
        val1, val2 = torch.split((dLdy * dyds_body).view(-1), (M_parallel, M_residual))

        dLds1 = torch.zeros((num_levels, N)).to(device)
        idx1  = idx1.view(-1, N)
        val1  = val1.view(-1, N)
        dLds1.scatter_add_(0, idx1, val1)
        dLds1 = dLds1.sum(1).view(-1)
        dLds1 = dLds1[:-1]

        dLds2 = torch.zeros(num_levels).to(device)
        if M_residual > 0:
            dLds2.scatter_add_(0, idx2, val2)
        dLds2 = dLds2[:-1]

        dLds_p  = (dLds1 + dLds2)*grad_scale_pos
        dLds_p += torch.where(flag_high, dLdy, torch.tensor(0, dtype=x.dtype, device = device)).sum().view(-1)*grad_scale_pos



        # derivative in negative region
        shift_x   = - x - torch.cat([torch.tensor([0]).to(device), cumsum_neg_scale])[idx_neg]
        dev_scale = (torch.cat([1/neg_scale, torch.tensor([0]).to(device)])) [idx_neg]#isn't minus meeded??
        d         = shift_x * dev_scale  

        rounded_d = d - torch.round(d)

        dyds_body = torch.where(flag_neg_outside, torch.tensor(0, dtype=x.dtype, device = device), rounded_d)

        box_size = int(box_size) # magic parameter for parallelize scatter_add_

        num_levels = neg_scale.numel() + 1
        box_size = int(box_size * num_levels/ (neg_scale.numel() + pos_scale.numel() +1))
        N          = int(box_size / num_levels)
        M_residual = int(x.numel() % box_size)
        M_parallel = x.numel() - M_residual

        idx1, idx2 = torch.split(               idx_neg.view(-1), (M_parallel, M_residual))
        val1, val2 = torch.split((dLdy * dyds_body).view(-1), (M_parallel, M_residual))

        dLds1 = torch.zeros((num_levels, N)).to(device)
        idx1  = idx1.view(-1, N)
        val1  = val1.view(-1, N)
        dLds1.scatter_add_(0, idx1, val1)
        dLds1 = dLds1.sum(1).view(-1)
        dLds1 = dLds1[:-1]

        dLds2 = torch.zeros(num_levels).to(device)
        if M_residual > 0:
            dLds2.scatter_add_(0, idx2, val2)
        dLds2 = dLds2[:-1]

        dLds_n  = (dLds1 + dLds2)*grad_scale_neg
        dLds_n -= torch.where(flag_low, dLdy, torch.tensor(0, dtype=x.dtype, device = device)).sum().view(-1)*grad_scale_neg
        dLds = torch.cat((dLds_n, dLds_p), dim=0)

        return dLdx, dLds, None, None, None, None, None, None
