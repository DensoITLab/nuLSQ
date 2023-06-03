##########################################################################
# Learned Step Size Quantization (ICLR2020) and its generalization
# * Implemented by Mitsuru Ambai
# * Denso IT Laboratory, Inc.
##########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils

#----------------------------------------------------------
# LSQ
#----------------------------------------------------------
def _quantize_LSQ(x, scale, Qn, Qp, num_elements,grad_scale_mode):

    Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
    Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
    assert scale > 0, 'scale = {}, {}, {}'.format(scale, Qn_on_device, Qp_on_device)
    
    # gradient scaling
    if num_elements > 0:
        if grad_scale_mode == "wo_grad_scale":
            grad_scale = torch.tensor(1.0).to(x.device)
        elif grad_scale_mode == "10_fac":
            grad_scale = torch.tensor(10.0).to(x.device)
        elif grad_scale_mode == "LSQ_grad_scale":
            grad_scale = 1.0 / torch.sqrt(num_elements * Qp_on_device) 
        elif grad_scale_mode == "10dfac_LSQ_grad_scale":
            grad_scale = 1.0 / (10*torch.sqrt(num_elements * Qp_on_device) )

        bw_scale   = scale * grad_scale
        scale      = (scale - bw_scale).detach() + bw_scale
    x  = x / scale
    
    x  = torch.min(torch.max(x, -Qn_on_device), Qp_on_device)
    xq = torch.round(x)
    y  = (xq - x).detach() + x
    
    y  = scale * y

    return y


#----------------------------------------------------------
# floor LSQ
#----------------------------------------------------------
def _quantize_floorLSQ(x, scale, Qn, Qp, num_elements,grad_scale_mode):

    Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
    Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
    assert scale > 0, 'scale = {}, {}, {}'.format(scale, Qn_on_device, Qp_on_device)
    
    # gradient scaling
    if num_elements > 0:
        if grad_scale_mode == "wo_grad_scale":
            grad_scale = torch.tensor(1.0).to(x.device)
        elif grad_scale_mode == "10_fac":
            grad_scale = torch.tensor(10.0).to(x.device)
        elif grad_scale_mode == "LSQ_grad_scale":
            grad_scale = 1.0 / torch.sqrt(num_elements * Qp_on_device) 
        elif grad_scale_mode == "floorLSQ_grad_scale":
            grad_scale = 1.0 / torch.sqrt(1000000*num_elements * Qp_on_device)
        elif grad_scale_mode == "floorLSQ_grad_scale_2":
            grad_scale = 1.0 / (num_elements * Qp_on_device)**3/2
        elif grad_scale_mode == "floorLSQ_grad_scale_3":
            grad_scale = 1.0 / (num_elements * Qp_on_device)
        elif grad_scale_mode == "floorLSQ_grad_scale_square":
            grad_scale = 1.0 / (num_elements * Qp_on_device)**2 

        bw_scale   = scale * grad_scale        
        scale      = (scale - bw_scale).detach() + bw_scale
    x  = x / scale 
    
    x  = torch.min(torch.max(x-1/2, -Qn_on_device ), Qp_on_device )
    xq = torch.round(x )
    y  = (xq - x).detach() + x
    flag   = (x < Qp_on_device) &  (x > -Qn_on_device)
    
    y  = scale * y  - 1/2 * (scale*flag).detach() + 1/2 * scale * flag
    # y  = scale * y  

    return y

#----------------------------------------------------------
# ceil LSQ
#----------------------------------------------------------
def _quantize_ceilLSQ(x, scale, Qn, Qp, num_elements,grad_scale_mode):

    Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
    Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
    assert scale > 0, 'scale = {}, {}, {}'.format(scale, Qn_on_device, Qp_on_device)
    
    # gradient scaling
    if num_elements > 0:
        if grad_scale_mode == "wo_grad_scale":
            grad_scale = torch.tensor(1.0).to(x.device)
        elif grad_scale_mode == "10_fac":
            grad_scale = torch.tensor(10.0).to(x.device)
        elif grad_scale_mode == "LSQ_grad_scale":
            grad_scale = 1.0 / torch.sqrt(num_elements * Qp_on_device) 
        bw_scale   = scale * grad_scale        
        scale      = (scale - bw_scale).detach() + bw_scale

    x  = x / scale
    
    x  = torch.min(torch.max(x, -Qn_on_device), Qp_on_device)
    xq = torch.ceil(x)
    y  = (xq - x).detach() + x
    
    y  = scale * y 

    return y


def _quantize_SoftPlus_LSQ(x, scale, Qn, Qp, num_elements):

    Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
    Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
    
    # gradient scaling
    if num_elements > 0:
        grad_scale = 1.0 / torch.sqrt(num_elements * Qp_on_device) 

        grad_alpha = 0.01
        bw_scale   = scale * grad_scale * grad_alpha
        
        scale      = (scale - bw_scale).detach() + bw_scale 

    x  = x / torch.log(1 + torch.exp(scale))
    x  = torch.min(torch.max(x, -Qn_on_device), Qp_on_device)
    xq = torch.round(x)
    y  = (xq - x).detach() + x
    y  = torch.log(1 + torch.exp(scale)) * y

    return y

def _quantize_Exp_LSQ(x, scale, Qn, Qp, num_elements):

    Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
    Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
    
    # gradient scaling
    if num_elements > 0:
        grad_scale = 1.0        
        # gradient scaling for exp LSQ & softplus LSQ
        grad_alpha = (grad_scale / ((torch.exp(scale))**2 + 0.01)).detach()
        bw_scale   = scale * grad_alpha
        
        scale      = (scale - bw_scale).detach() + bw_scale 

    x  = x / torch.exp(scale)
    x  = torch.min(torch.max(x, -Qn_on_device), Qp_on_device)
    xq = torch.round(x)
    y  = (xq - x).detach() + x
    y  = torch.exp(scale) * y

    return y

#----------------------------------------------------------
# LSQ
#----------------------------------------------------------
def _quantize_shiftLSQ(x, b, scale, Qn, Qp, num_elements,grad_scale_mode):

    Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
    Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
    assert scale > 0, 'scale = {}, {}, {}'.format(scale, Qn_on_device, Qp_on_device)
    #if scale < 0:
        #scale = (-1*scale).detach()
        #print('scale = {}'.format(scale))
    
    # gradient scaling
    if num_elements > 0:
        if grad_scale_mode == "wo_grad_scale":
            grad_scale = torch.tensor(1.0).to(x.device)
            # print(grad_scale)
        elif grad_scale_mode == "10_fac":
            grad_scale = torch.tensor(10.0).to(x.device)
            # print(grad_scale)
        elif grad_scale_mode == "LSQ_grad_scale":
            grad_scale = 1.0 / torch.sqrt(num_elements * Qp_on_device) 
        elif grad_scale_mode == "10dfac_LSQ_grad_scale":
            grad_scale = 1.0 / (10*torch.sqrt(num_elements * Qp_on_device) )
        #grad_scale = 1.0
        # default gradient scaling for LSQ
        bw_scale   = scale * grad_scale
        
        scale      = (scale - bw_scale).detach() + bw_scale
    x  = (x - b) / scale
    
    x  = torch.min(torch.max(x, -Qn_on_device), Qp_on_device)
    xq = torch.round(x)
    y  = (xq - x).detach() + x
    y  = scale * y + b

    return y



#----------------------------------------------------------
# LSQ (non uniform version for activation)
#----------------------------------------------------------
class _quantize_LSQ_non_uniform_act_core(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, num_bits, num_elements, x_grad_scale_mode):

        num_levels = 2 ** num_bits
        for s in scale:
            assert s > 0, 'scale = {}'.format(scale)
        if num_elements > 0:
            if x_grad_scale_mode == "wo_grad_scale":
                grad_scale = torch.tensor(1.0).to(x.device)
            elif x_grad_scale_mode == "LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            elif x_grad_scale_mode == "10fac_LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / (10*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            else:
                print("Not implemented:", x_grad_scale_mode)
        else:
            grad_scale = torch.tensor(1.0).to(x.device)

        y = torch.zeros_like(x)
        Ns_x = torch.zeros_like(scale)
        cumsum_scale = torch.tensor([0], dtype=torch.float).to(x.device)
        for i in range(num_levels-1):
            offset  = cumsum_scale + scale[i]/2
            y      += scale[i] * torch.heaviside(x - offset, torch.tensor(0.0, dtype=torch.float).to(x.device))
            Ns_x[i] = ((x > cumsum_scale) & (x < cumsum_scale + scale[i])).float().sum()
            cumsum_scale += scale[i]
            # assert Ns_x[i] > 0, '(i, scale, Ns_x) = {}, {}, {}'.format(i, scale, Ns_x[i])
        if x_grad_scale_mode == "nuLSQ_grad_scale":
            ratio = Ns_x/(Ns_x.sum())
        else:
            ratio = torch.ones_like(scale) 
        ctx.save_for_backward(x, scale, grad_scale, ratio )
        return y, Ns_x

    @staticmethod
    def backward(ctx, dLdy, temp_Ns_x):
        x, scale, grad_scale, ratio= ctx.saved_tensors
        
        num_levels = torch.numel(scale) + 1
        # cum_scale = scale.cumsum(dim=0).tolist()
        # N_w = (x > 0).float().sum()
        # N_s = ((x > 0) & (x < cum_scale[0])).float().sum()
        

        flag_high   = x > scale.sum()
        flag_low    = x < 0
        flag_middle = (~flag_high) & (~flag_low)

        dydx = torch.ones_like(x) * flag_middle

        dLds = torch.zeros_like(scale)
        cumsum_scale1 = torch.tensor(0, dtype=torch.float).to(x.device)
        cumsum_scale2 = scale[0].clone()
        cumsum_scale3 = scale.sum()
        # Ns_tot = ((x > 0 ) & (x< cumsum_scale3)).float().sum()
        for i in range(num_levels-1):
            # if i > 0:
            #     N_s = ((x >= cum_scale[i-1]) & (x < cum_scale[i])).float().sum()
            flag_body  = (cumsum_scale1 < x) & (x < cumsum_scale2)
            flag_upper =  cumsum_scale3 < x

            shift_x = x - cumsum_scale1
            d1 = torch.heaviside(shift_x - scale[i]/2, torch.tensor(0.0, dtype=torch.float).to(x.device))
            d2 = -shift_x / scale[i]
            d  = d1 + d2  

            dyds_body  = flag_body  * (d)
            dyds_upper = flag_upper * (1.0)
            dyds       = (dyds_body + dyds_upper)/torch.sqrt(ratio[i])


            if i < num_levels-2:
                cumsum_scale1 += scale[i+0]
                cumsum_scale2 += scale[i+1]
                
            dLds[i] = (dLdy * dyds).sum().view(-1) * grad_scale

        return dLdy * dydx, dLds, None, None, None


def _quantize_LSQ_non_uniform_act(x, scale, num_bits, num_elements, x_grad_scale_mode):
    func = _quantize_LSQ_non_uniform_act_core.apply
    return func(x, scale, num_bits, num_elements, x_grad_scale_mode)

#----------------------------------------------------------
# LSQ (non uniform fast test version for activation)
#----------------------------------------------------------
class _quantize_LSQ_non_uniform_fast_act_core(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, num_bits, num_elements, x_grad_scale_mode):

        num_levels = 2 ** num_bits
        q_0 = torch.tensor(0.0, dtype=torch.float).to(x.device)
        q_1 = torch.tensor(1.0, dtype=torch.float).to(x.device)
        for s in scale:
            assert s > 0, 'scale = {}'.format(scale)
        if num_elements > 0:
            if x_grad_scale_mode == "wo_grad_scale":
                grad_scale = torch.tensor(1.0).to(x.device)
            elif x_grad_scale_mode == "LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            elif x_grad_scale_mode == "10fac_LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / (10*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            else:
                print("Not implemented:", x_grad_scale_mode)
        else:
            grad_scale = torch.tensor(1.0).to(x.device)

        y = torch.zeros_like(x)
        Ns_x = torch.zeros_like(scale)
        cumsum_scale = torch.tensor([0], dtype=torch.float).to(x.device)
        for i in range(num_levels-1):
            offset  = cumsum_scale + scale[i]/2
            y      +=  (x > offset).float() * scale[i]
            Ns_x[i] = ((x > cumsum_scale) & (x < cumsum_scale + scale[i])).float().sum()
            cumsum_scale += scale[i]
            # assert Ns_x[i] > 0, '(i, scale, Ns_x) = {}, {}, {}'.format(i, scale, Ns_x[i])
        if x_grad_scale_mode == "nuLSQ_grad_scale":
            ratio = Ns_x/(Ns_x.sum())
        else:
            ratio = torch.ones_like(scale) 
        ctx.save_for_backward(x, scale, grad_scale, ratio )
        return y, Ns_x

    @staticmethod
    def backward(ctx, dLdy, temp_Ns_x):
        x, scale, grad_scale, ratio= ctx.saved_tensors
        num_levels = torch.numel(scale) + 1
        
        threshold = scale.sum()
        flag_high   = x > threshold
        flag_low    = x < 0
        flag_middle = (~flag_high) & (~flag_low)

        dydx = torch.ones_like(x) * flag_middle

        dLds = torch.zeros_like(scale)
        cumsum_scale1 = torch.tensor(0, dtype=torch.float).to(x.device)
        cumsum_scale2 = scale[0].clone()
        
        for i in range(num_levels-1):
            flag_body  = (cumsum_scale1 < x) & (x < cumsum_scale2)
            shift_x = x - cumsum_scale1
            d2 = -shift_x / scale[i]
            d1 = torch.round(-d2)
            d  = d1 + d2  

            dyds_body  = flag_body  * (d)
            dyds_upper = flag_high * (1.0)
            dyds       = (dyds_body + dyds_upper)/torch.sqrt(ratio[i])


            if i < num_levels-2:
                cumsum_scale1 += scale[i+0]
                cumsum_scale2 += scale[i+1]
                
            dLds[i] = (dLdy * dyds).sum().view(-1) * grad_scale

        return dLdy * dydx, dLds, None, None, None


def _quantize_LSQ_non_uniform_fast_act(x, scale, num_bits, num_elements, x_grad_scale_mode):
    func = _quantize_LSQ_non_uniform_fast_act_core.apply
    return func(x, scale, num_bits, num_elements, x_grad_scale_mode)


#----------------------------------------------------------
# LSQ (non uniform fast test autograd version for activation)
#----------------------------------------------------------

def _quantize_LSQ_non_uniform_autograd_act(x, scale, num_bits, num_elements, x_grad_scale_mode):

    num_levels = 2 ** num_bits
    q_0 = torch.tensor(0.0, dtype=torch.float).to(x.device)
    q_1 = torch.tensor(1.0, dtype=torch.float).to(x.device)
    # print(scale)
    for s in scale:
        assert s > 0, 'scale = {}'.format(scale)
    if num_elements > 0:
        if x_grad_scale_mode == "wo_grad_scale":
            grad_scale = torch.tensor(1.0).to(x.device)
        elif x_grad_scale_mode == "LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
            grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
        elif x_grad_scale_mode == "10fac_LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
            grad_scale = 1.0 / (10*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
        else:
            print("Not implemented:", x_grad_scale_mode)
    else:
        grad_scale = torch.tensor(1.0).to(x.device)

    bw_scale   = scale * grad_scale
    scale      = (scale - bw_scale).detach() + bw_scale

    y = torch.zeros_like(x)
    Ns_x = torch.zeros_like(scale)

    cumsum_scale = torch.cumsum(scale, dim=0) - scale
    for i in range(num_levels-1):
        tmp = (x - cumsum_scale[i])/scale[i]            
        tmp = torch.min(torch.max(tmp,q_0),q_1)
        xq = torch.round(tmp)

        # tmp = torch.round( \
        #     torch.min(torch.max((x - cumsum_scale)/scale[i],q_0),q_1))
        y   = y + ((xq - tmp).detach() + tmp) * scale[i]

        # y  =  y + ((torch.round( \
        #              torch.min(torch.max((x - cumsum_scale[i])/scale[i],q_0),q_1)) \
        #             - torch.min(torch.max((x - cumsum_scale[i])/scale[i],q_0),q_1) \
        #              ).detach() \
        #             + torch.min(torch.max((x - cumsum_scale[i])/scale[i],q_0),q_1)) * scale[i]
        Ns_x[i] = ((x > cumsum_scale[i]) & (x < cumsum_scale[i] + scale[i])).float().sum()
        
        # assert Ns_x[i] > 0, '(i, scale, Ns_x) = {}, {}, {}'.format(i, scale, Ns_x[i])
    if x_grad_scale_mode == "nuLSQ_grad_scale":
        print("Not implemented yet")
    return y, Ns_x

#----------------------------------------------------------
# LSQ (non uniform version for weight)
#----------------------------------------------------------
class _quantize_LSQ_non_uniform_weight_core(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, neg_scale, pos_scale, Qn, Qp, num_elements, grad_scale_mode):

        Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
        Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
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

        ctx.save_for_backward(x, neg_scale, pos_scale, grad_scale_neg, grad_scale_pos, Qn_on_device, Qp_on_device)

        y = torch.zeros_like(x)
        cumsum_scale = torch.tensor([0], dtype=torch.float).to(x.device)
        for i in range(Qp):
            offset  = cumsum_scale + pos_scale[i]/2
            y      += pos_scale[i] * torch.heaviside(x - offset, torch.tensor(0.0, dtype=torch.float).to(x.device))
            cumsum_scale += pos_scale[i]
        cumsum_scale = torch.tensor([0], dtype=torch.float).to(x.device)
        for i in range(Qn):
            offset  = cumsum_scale + neg_scale[i]/2
            y      -= neg_scale[i] * torch.heaviside(-x - offset, torch.tensor(0.0, dtype=torch.float).to(x.device))
            cumsum_scale += neg_scale[i]
        return y

    @staticmethod
    def backward(ctx, dLdy):
        x, neg_scale, pos_scale, grad_scale_neg, grad_scale_pos, Qn_on_device, Qp_on_device = ctx.saved_tensors

        flag_high   = x > pos_scale.sum()
        flag_low    = x < -neg_scale.sum()
        flag_middle = (~flag_high) & (~flag_low)
        
#         cum_neg_scale = neg_scale.cumsum(dim=0).tolist()
#         cum_pos_scale = pos_scale.cumsum(dim=0).tolist()
        
#         pN_s = ((x > 0) & (x <= cum_pos_scale[0])).float().sum()
#         nN_s = ((x < 0) & (x >= -cum_neg_scale[0])).float().sum()

        dydx = torch.ones_like(x) * flag_middle
        # update for positive scale
        dLds_p = torch.zeros_like(pos_scale)
        cumsum_pos_scale1 = torch.tensor(0, dtype=torch.float).to(x.device)
        cumsum_pos_scale2 = pos_scale[0].clone()
        cumsum_pos_scale3 = pos_scale.sum()
        for i in range(int(Qp_on_device)):
                
#             if i > 0:
#                 pN_s = ((x > cum_pos_scale[i-1]) & (x <= cum_pos_scale[i])).float().sum()
            
            flag_body  = (cumsum_pos_scale1 < x) & (x < cumsum_pos_scale2)
            flag_upper =  cumsum_pos_scale3 < x

            shift_x = x - cumsum_pos_scale1
            d1 = torch.heaviside(shift_x - pos_scale[i]/2, torch.tensor(0.0, dtype=torch.float).to(x.device))
            d2 = -shift_x / pos_scale[i]
            d  = d1 + d2

            dyds_body  = flag_body  * (d)
            dyds_upper = flag_upper * (1.0)
            dyds       = dyds_body + dyds_upper

            if i < Qp_on_device-1:
                cumsum_pos_scale1 += pos_scale[i+0]
                cumsum_pos_scale2 += pos_scale[i+1]
                
#             dLds_p[i] = (dLdy * dyds).sum().view(-1) * torch.max(torch.tensor(0.001), (1.0 / torch.sqrt(pN_s * Qp_on_device)))
            dLds_p[i] = (dLdy * dyds).sum().view(-1) * grad_scale_pos


        # update for negative scale        
        dLds_n = torch.zeros_like(neg_scale)
        cumsum_neg_scale1 = torch.tensor(0, dtype=torch.float).to(x.device)
        cumsum_neg_scale2 = neg_scale[0].clone()
        cumsum_neg_scale3 = neg_scale.sum()
        for i in range(int(Qn_on_device)):
#             if i > 0:
#                 nN_s = ((x < -cum_neg_scale[i-1]) & (x >= -cum_neg_scale[i])).float().sum()
                
            flag_body  = (cumsum_neg_scale1 < -x) & (-x < cumsum_neg_scale2)
            flag_lower =  cumsum_neg_scale3 < -x

            shift_x = -x - cumsum_neg_scale1
            d1 = -torch.heaviside(shift_x - neg_scale[i]/2, torch.tensor(0.0, dtype=torch.float).to(x.device))
            d2 = shift_x / neg_scale[i]
            d  = d1 + d2

            dyds_body  = flag_body  * (d)
            dyds_lower = flag_lower * (-1.0)
            dyds       = dyds_body + dyds_lower

            if i < Qn_on_device-1:
                cumsum_neg_scale1 += neg_scale[i+0]
                cumsum_neg_scale2 += neg_scale[i+1]

#             dLds_n[i] = (dLdy * dyds).sum().view(-1) * torch.max(torch.tensor(0.001), (1.0 / torch.sqrt(nN_s * Qn_on_device)))
            dLds_n[i] = (dLdy * dyds).sum().view(-1) * grad_scale_neg


        return dLdy * dydx, dLds_n, dLds_p, None, None, None, None


def _quantize_LSQ_non_uniform_weight(x, neg_scale, pos_scale, Qn, Qp, num_elements, w_grad_scale_mode):
    # Note: This function assumes x >= 0. Therefore, only available for activations not for weights.
    func = _quantize_LSQ_non_uniform_weight_core.apply
    return func(x, neg_scale, pos_scale, Qn, Qp, num_elements, w_grad_scale_mode)

#----------------------------------------------------------
# LSQ (non uniform version for activation)
#----------------------------------------------------------
class _quantize_LSQ_non_uniform_nunlocal_update_act_core(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, num_bits, num_elements, x_grad_scale_mode):

        num_levels = 2 ** num_bits
        for s in scale:
            assert s > 0, 'scale = {}'.format(scale)
        if num_elements > 0:
            # grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            if x_grad_scale_mode == "wo_grad_scale":
                grad_scale = torch.tensor(1.0).to(x.device)
            elif x_grad_scale_mode == "LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            elif x_grad_scale_mode == "10fac_LSQ_grad_scale" :
                grad_scale = 1.0 / (10*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            elif x_grad_scale_mode == "100fac_LSQ_grad_scale" :
                grad_scale = 1.0 / (100*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            else:
                print("Not implemented:", x_grad_scale_mode)
        else:
            grad_scale = torch.tensor(1.0).to(x.device)

        # num_elements = torch.tensor(num_elements).to(x.device)
        # ctx.save_for_backward(x, scale, num_elements)

        y = torch.zeros_like(x)
        Ns_x = torch.zeros_like(scale)
        cumsum_scale = torch.tensor([0], dtype=torch.float).to(x.device)
        for i in range(num_levels-1):
            offset  = cumsum_scale + scale[i]/2
            y      += scale[i] * torch.heaviside(x - offset, torch.tensor(0.0, dtype=torch.float).to(x.device))
            Ns_x[i] = ((x > cumsum_scale) & (x < cumsum_scale + scale[i])).float().sum()
            cumsum_scale += scale[i]
            assert Ns_x[i] > 0, '(i, scale, Ns_x) = {}, {}, {}'.format(i, scale, Ns_x[i])
        if x_grad_scale_mode == "nuLSQ_grad_scale":
            ratio = Ns_x/(Ns_x.sum())
        else:
            ratio = torch.ones_like(scale) 
        # ctx.Ns_x = Ns_x
        ctx.save_for_backward(x, scale, grad_scale, ratio )
        return y, Ns_x

    @staticmethod
    def backward(ctx, dLdy, temp_Ns_x):
        x, scale, grad_scale, ratio= ctx.saved_tensors
        # x, scale, num_elements = ctx.saved_tensors
        num_levels = torch.numel(scale) + 1
        # cum_scale = scale.cumsum(dim=0).tolist()
        # N_w = (x > 0).float().sum()
        # N_s = ((x > 0) & (x < cum_scale[0])).float().sum()
        

        flag_high   = x > scale.sum()
        flag_low    = x < 0
        flag_middle = (~flag_high) & (~flag_low)

        dydx = torch.ones_like(x) * flag_middle

        dLds = torch.zeros_like(scale)
        cumsum_scale1 = torch.tensor(0, dtype=torch.float).to(x.device)
        cumsum_scale2 = scale[0].clone()
        cumsum_scale3 = scale.sum()
        # Ns_tot = ((x > 0 ) & (x< cumsum_scale3)).float().sum()
        flag_upper =  cumsum_scale3 < x
        for i in range(num_levels-1):
            # if i > 0:
            #     N_s = ((x >= cum_scale[i-1]) & (x < cum_scale[i])).float().sum()
            flag_body  = (cumsum_scale1 < x) & (x < cumsum_scale2)

            shift_x = x - cumsum_scale1
            d1 = torch.heaviside(shift_x - scale[i]/2, torch.tensor(0.0, dtype=torch.float).to(x.device))
            d2 = -shift_x / scale[i]
            d  = d1 + d2  

            dyds_body  = flag_body  * (d)/torch.sqrt(ratio[i])
            dyds_upper = flag_upper * (1.0)/torch.sqrt(ratio[i])
            dyds       = (dyds_body + dyds_upper)


            if i < num_levels-2:
                cumsum_scale1 += scale[i+0]
                cumsum_scale2 += scale[i+1]
                
#             dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt((N_s/N_w) * (num_levels-1))) * torch.sqrt(torch.tensor((num_levels - 1), dtype=torch.float))
            # dLds[i] = (dLdy * dyds).sum().view(-1)
            dLds_body = (dLdy * dyds_body).sum().view(-1 ) * grad_scale/(i+1)
            dLds[i] = dLds_body + (dLdy * dyds_upper).sum().view(-1 ) * grad_scale
            if i > 0:
                for j in range(i):# found bug: should be i-1 -> i
                    dLds[j] = dLds_body + dLds[j]

            #print(i, (1.0 / torch.sqrt(N_s * (num_levels-1))))
            # dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt(num_elements * (N_s/N_w)))
#             dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt((N_s/N_w) * (num_levels-1)))

        return dLdy * dydx, dLds, None, None, None


def _quantize_LSQ_non_uniform_nunlocal_update_act(x, scale, num_bits, num_elements, x_grad_scale_mode):
    # Note: This function assumes x >= 0. Therefore, only available for activations not for weights.
    func = _quantize_LSQ_non_uniform_nunlocal_update_act_core.apply
    return func(x, scale, num_bits, num_elements, x_grad_scale_mode)


#----------------------------------------------------------
# LSQ (non uniform version for activation)
#----------------------------------------------------------
class _quantize_LSQ_non_uniform_nunlocal_update_II_act_core(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, num_bits, num_elements, x_grad_scale_mode):

        num_levels = 2 ** num_bits
        for s in scale:
            assert s > 0, 'scale = {}'.format(scale)
        if num_elements > 0:
            # grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            if x_grad_scale_mode == "wo_grad_scale":
                grad_scale = torch.tensor(1.0).to(x.device)
            elif x_grad_scale_mode == "LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            elif x_grad_scale_mode == "10fac_LSQ_grad_scale" :
                grad_scale = 1.0 / (10*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            elif x_grad_scale_mode == "100fac_LSQ_grad_scale" :
                grad_scale = 1.0 / (100*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            else:
                print("Not implemented:", x_grad_scale_mode)
        else:
            grad_scale = torch.tensor(1.0).to(x.device)

        # num_elements = torch.tensor(num_elements).to(x.device)
        # ctx.save_for_backward(x, scale, num_elements)

        y = torch.zeros_like(x)
        Ns_x = torch.zeros_like(scale)
        cumsum_scale = torch.tensor([0], dtype=torch.float).to(x.device)
        for i in range(num_levels-1):
            offset  = cumsum_scale + scale[i]/2
            y      += scale[i] * torch.heaviside(x - offset, torch.tensor(0.0, dtype=torch.float).to(x.device))
            Ns_x[i] = ((x > cumsum_scale) & (x < cumsum_scale + scale[i])).float().sum()
            cumsum_scale += scale[i]
            assert Ns_x[i] > 0, '(i, scale, Ns_x) = {}, {}, {}'.format(i, scale, Ns_x[i])
        if x_grad_scale_mode == "nuLSQ_grad_scale":
            ratio = Ns_x/(Ns_x.sum())
        else:
            ratio = torch.ones_like(scale) 
        # ctx.Ns_x = Ns_x
        ctx.save_for_backward(x, scale, grad_scale, ratio )
        return y, Ns_x

    @staticmethod
    def backward(ctx, dLdy, temp_Ns_x):
        x, scale, grad_scale, ratio= ctx.saved_tensors
        # x, scale, num_elements = ctx.saved_tensors
        num_levels = torch.numel(scale) + 1
        # cum_scale = scale.cumsum(dim=0).tolist()
        # N_w = (x > 0).float().sum()
        # N_s = ((x > 0) & (x < cum_scale[0])).float().sum()
        

        flag_high   = x > scale.sum()
        flag_low    = x < 0
        flag_middle = (~flag_high) & (~flag_low)

        dydx = torch.ones_like(x) * flag_middle

        dLds = torch.zeros_like(scale)
        cumsum_scale1 = torch.tensor(0, dtype=torch.float).to(x.device)
        cumsum_scale2 = scale[0].clone()
        cumsum_scale3 = scale.sum()
        # Ns_tot = ((x > 0 ) & (x< cumsum_scale3)).float().sum()
        flag_upper =  cumsum_scale3 < x
        for i in range(num_levels-1):
            # if i > 0:
            #     N_s = ((x >= cum_scale[i-1]) & (x < cum_scale[i])).float().sum()
            flag_body  = (cumsum_scale1 < x) & (x < cumsum_scale2)

            shift_x = x - cumsum_scale1
            d1 = torch.heaviside(shift_x - scale[i]/2, torch.tensor(0.0, dtype=torch.float).to(x.device))
            d2 = -shift_x / scale[i]
            d  = d1 + d2  

            dyds_body  = flag_body  * (d)/torch.sqrt(ratio[i])
            dyds_upper = flag_upper * (1.0)/torch.sqrt(ratio[i])
            dyds       = (dyds_body + dyds_upper)


            if i < num_levels-2:
                cumsum_scale1 += scale[i+0]
                cumsum_scale2 += scale[i+1]
                
#             dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt((N_s/N_w) * (num_levels-1))) * torch.sqrt(torch.tensor((num_levels - 1), dtype=torch.float))
            # dLds[i] = (dLdy * dyds).sum().view(-1)
            dLds_body = (dLdy * dyds).sum().view(-1 ) * grad_scale/(i+1)
            for j in range(i+1):
                dLds[j] = dLds_body + dLds[j]

            #print(i, (1.0 / torch.sqrt(N_s * (num_levels-1))))
            # dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt(num_elements * (N_s/N_w)))
#             dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt((N_s/N_w) * (num_levels-1)))

        return dLdy * dydx, dLds, None, None, None


def _quantize_LSQ_non_uniform_nunlocal_update_II_act(x, scale, num_bits, num_elements, x_grad_scale_mode):
    # Note: This function assumes x >= 0. Therefore, only available for activations not for weights.
    func = _quantize_LSQ_non_uniform_nunlocal_update_II_act_core.apply
    return func(x, scale, num_bits, num_elements, x_grad_scale_mode)


#----------------------------------------------------------
# LSQ (non uniform version for activation) stronger effect of dyds_upper than type-II
#----------------------------------------------------------
class _quantize_LSQ_non_uniform_nunlocal_update_III_act_core(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, num_bits, num_elements, x_grad_scale_mode):

        num_levels = 2 ** num_bits
        for s in scale:
            assert s > 0, 'scale = {}'.format(scale)
        if num_elements > 0:
            # grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            if x_grad_scale_mode == "wo_grad_scale":
                grad_scale = torch.tensor(1.0).to(x.device)
            elif x_grad_scale_mode == "LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            elif x_grad_scale_mode == "10fac_LSQ_grad_scale" :
                grad_scale = 1.0 / (10*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            elif x_grad_scale_mode == "100fac_LSQ_grad_scale" :
                grad_scale = 1.0 / (100*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            else:
                print("Not implemented:", x_grad_scale_mode)
        else:
            grad_scale = torch.tensor(1.0).to(x.device)

        # num_elements = torch.tensor(num_elements).to(x.device)
        # ctx.save_for_backward(x, scale, num_elements)

        y = torch.zeros_like(x)
        Ns_x = torch.zeros_like(scale)
        cumsum_scale = torch.tensor([0], dtype=torch.float).to(x.device)
        for i in range(num_levels-1):
            offset  = cumsum_scale + scale[i]/2
            y      += scale[i] * torch.heaviside(x - offset, torch.tensor(0.0, dtype=torch.float).to(x.device))
            Ns_x[i] = ((x > cumsum_scale) & (x < cumsum_scale + scale[i])).float().sum()
            cumsum_scale += scale[i]
            assert Ns_x[i] > 0, '(i, scale, Ns_x) = {}, {}, {}'.format(i, scale, Ns_x[i])
        if x_grad_scale_mode == "nuLSQ_grad_scale":
            ratio = Ns_x/(Ns_x.sum())
        else:
            ratio = torch.ones_like(scale) 
        # ctx.Ns_x = Ns_x
        ctx.save_for_backward(x, scale, grad_scale, ratio )
        return y, Ns_x

    @staticmethod
    def backward(ctx, dLdy, temp_Ns_x):
        x, scale, grad_scale, ratio= ctx.saved_tensors
        # x, scale, num_elements = ctx.saved_tensors
        num_levels = torch.numel(scale) + 1
        # cum_scale = scale.cumsum(dim=0).tolist()
        # N_w = (x > 0).float().sum()
        # N_s = ((x > 0) & (x < cum_scale[0])).float().sum()
        

        flag_high   = x > scale.sum()
        flag_low    = x < 0
        flag_middle = (~flag_high) & (~flag_low)

        dydx = torch.ones_like(x) * flag_middle

        dLds = torch.zeros_like(scale)
        cumsum_scale1 = torch.tensor(0, dtype=torch.float).to(x.device)
        cumsum_scale2 = scale[0].clone()
        cumsum_scale3 = scale.sum()
        # Ns_tot = ((x > 0 ) & (x< cumsum_scale3)).float().sum()
        flag_upper =  cumsum_scale3 < x
        for i in range(num_levels-1):
            # if i > 0:
            #     N_s = ((x >= cum_scale[i-1]) & (x < cum_scale[i])).float().sum()
            flag_body  = (cumsum_scale1 < x) & (x < cumsum_scale2)

            shift_x = x - cumsum_scale1
            d1 = torch.heaviside(shift_x - scale[i]/2, torch.tensor(0.0, dtype=torch.float).to(x.device))
            d2 = -shift_x / scale[i]
            d  = d1 + d2  

            dyds_body  = flag_body  * (d)/torch.sqrt(ratio[i])
            dyds_upper = flag_upper * (1.0)/torch.sqrt(ratio[i])
            # dyds       = (dyds_body + dyds_upper)


            if i < num_levels-2:
                cumsum_scale1 += scale[i+0]
                cumsum_scale2 += scale[i+1]
                
#             dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt((N_s/N_w) * (num_levels-1))) * torch.sqrt(torch.tensor((num_levels - 1), dtype=torch.float))
            # dLds[i] = (dLdy * dyds).sum().view(-1)
            dLds_body = (dLdy * (dyds_body/(i+1) +  dyds_upper)).sum().view(-1 ) * grad_scale
            for j in range(i+1):
                dLds[j] = dLds_body + dLds[j]

            #print(i, (1.0 / torch.sqrt(N_s * (num_levels-1))))
            # dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt(num_elements * (N_s/N_w)))
#             dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt((N_s/N_w) * (num_levels-1)))

        return dLdy * dydx, dLds, None, None, None


def _quantize_LSQ_non_uniform_nunlocal_update_III_act(x, scale, num_bits, num_elements, x_grad_scale_mode):
    # Note: This function assumes x >= 0. Therefore, only available for activations not for weights.
    func = _quantize_LSQ_non_uniform_nunlocal_update_III_act_core.apply
    return func(x, scale, num_bits, num_elements, x_grad_scale_mode)

#----------------------------------------------------------
# LSQ (non uniform version for activation) naiive extension of nuLSQ to nonlocal
#----------------------------------------------------------
class _quantize_LSQ_non_uniform_nunlocal_update_IV_act_core(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, num_bits, num_elements, x_grad_scale_mode):

        num_levels = 2 ** num_bits
        for s in scale:
            assert s > 0, 'scale = {}'.format(scale)
        if num_elements > 0:
            # grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            if x_grad_scale_mode == "wo_grad_scale":
                grad_scale = torch.tensor(1.0).to(x.device)
            elif x_grad_scale_mode == "LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            elif x_grad_scale_mode == "10fac_LSQ_grad_scale" :
                grad_scale = 1.0 / (10*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            elif x_grad_scale_mode == "100fac_LSQ_grad_scale" :
                grad_scale = 1.0 / (100*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            else:
                print("Not implemented:", x_grad_scale_mode)
        else:
            grad_scale = torch.tensor(1.0).to(x.device)

        # num_elements = torch.tensor(num_elements).to(x.device)
        # ctx.save_for_backward(x, scale, num_elements)

        y = torch.zeros_like(x)
        Ns_x = torch.zeros_like(scale)
        cumsum_scale = torch.tensor([0], dtype=torch.float).to(x.device)
        for i in range(num_levels-1):
            offset  = cumsum_scale + scale[i]/2
            y      += scale[i] * torch.heaviside(x - offset, torch.tensor(0.0, dtype=torch.float).to(x.device))
            Ns_x[i] = ((x > cumsum_scale) & (x < cumsum_scale + scale[i])).float().sum()
            cumsum_scale += scale[i]
            assert Ns_x[i] > 0, '(i, scale, Ns_x) = {}, {}, {}'.format(i, scale, Ns_x[i])
        if x_grad_scale_mode == "nuLSQ_grad_scale":
            ratio = Ns_x/(Ns_x.sum())
        else:
            ratio = torch.ones_like(scale) 
        # ctx.Ns_x = Ns_x
        ctx.save_for_backward(x, scale, grad_scale, ratio )
        return y, Ns_x

    @staticmethod
    def backward(ctx, dLdy, temp_Ns_x):
        x, scale, grad_scale, ratio= ctx.saved_tensors
        # x, scale, num_elements = ctx.saved_tensors
        num_levels = torch.numel(scale) + 1
        # cum_scale = scale.cumsum(dim=0).tolist()
        # N_w = (x > 0).float().sum()
        # N_s = ((x > 0) & (x < cum_scale[0])).float().sum()
        

        flag_high   = x > scale.sum()
        flag_low    = x < 0
        flag_middle = (~flag_high) & (~flag_low)

        dydx = torch.ones_like(x) * flag_middle

        dLds = torch.zeros_like(scale)
        cumsum_scale1 = torch.tensor(0, dtype=torch.float).to(x.device)
        cumsum_scale2 = scale[0].clone()
        cumsum_scale3 = scale.sum()
        # Ns_tot = ((x > 0 ) & (x< cumsum_scale3)).float().sum()
        flag_upper =  cumsum_scale3 < x
        for i in range(num_levels-1):
            # if i > 0:
            #     N_s = ((x >= cum_scale[i-1]) & (x < cum_scale[i])).float().sum()
            flag_body  = (cumsum_scale1 < x) & (x < cumsum_scale2)

            shift_x = x - cumsum_scale1
            d1 = torch.heaviside(shift_x - scale[i]/2, torch.tensor(0.0, dtype=torch.float).to(x.device))
            d2 = -shift_x / scale[i]
            d  = d1 + d2  

            dyds_body  = flag_body  * (d)/torch.sqrt(ratio[i])
            dyds_upper = flag_upper * (1.0)/torch.sqrt(ratio[i])
            dyds       = (dyds_body + dyds_upper)


            if i < num_levels-2:
                cumsum_scale1 += scale[i+0]
                cumsum_scale2 += scale[i+1]
                
#             dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt((N_s/N_w) * (num_levels-1))) * torch.sqrt(torch.tensor((num_levels - 1), dtype=torch.float))
            # dLds[i] = (dLdy * dyds).sum().view(-1)
            dLds_body = (dLdy * dyds).sum().view(-1 ) * grad_scale
            dLds[i] = dLds[i] + dLds_body
            if i > 0:
                for j in range(i):
                    dLds[j] = dLds_body/(i+1) + dLds[j]

            #print(i, (1.0 / torch.sqrt(N_s * (num_levels-1))))
            # dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt(num_elements * (N_s/N_w)))
#             dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt((N_s/N_w) * (num_levels-1)))

        return dLdy * dydx, dLds, None, None, None


def _quantize_LSQ_non_uniform_nunlocal_update_IV_act(x, scale, num_bits, num_elements, x_grad_scale_mode):
    # Note: This function assumes x >= 0. Therefore, only available for activations not for weights.
    func = _quantize_LSQ_non_uniform_nunlocal_update_IV_act_core.apply
    return func(x, scale, num_bits, num_elements, x_grad_scale_mode)

#----------------------------------------------------------
# LSQ (non uniform version for activation) naiive extension of nuLSQ to nonlocal
#----------------------------------------------------------
class _quantize_LSQ_non_uniform_nunlocal_update_V_act_core(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, num_bits, num_elements, x_grad_scale_mode):

        num_levels = 2 ** num_bits
        for s in scale:
            assert s > 0, 'scale = {}, grad_mode:{}'.format(scale, x_grad_scale_mode)
        if num_elements > 0:
            # grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            if x_grad_scale_mode == "wo_grad_scale":
                grad_scale = torch.tensor(1.0).to(x.device)
            elif x_grad_scale_mode == "LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            elif x_grad_scale_mode == "10dfac_LSQ_grad_scale" :
                grad_scale = 1.0 / (10*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            elif x_grad_scale_mode == "100fac_LSQ_grad_scale" :
                grad_scale = 1.0 / (100*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            else:
                print("Not implemented:", x_grad_scale_mode)
        else:
            grad_scale = torch.tensor(1.0).to(x.device)

        # num_elements = torch.tensor(num_elements).to(x.device)
        # ctx.save_for_backward(x, scale, num_elements)

        y = torch.zeros_like(x)
        Ns_x = torch.zeros_like(scale)
        cumsum_scale = torch.tensor([0], dtype=torch.float).to(x.device)
        for i in range(num_levels-1):
            offset  = cumsum_scale + scale[i]/2
            y      += scale[i] * torch.heaviside(x - offset, torch.tensor(0.0, dtype=torch.float).to(x.device))
            Ns_x[i] = ((x > cumsum_scale) & (x < cumsum_scale + scale[i])).float().sum()
            cumsum_scale += scale[i]
            # assert Ns_x[i] > 0, '(i, scale, Ns_x) = {}, {}, {}'.format(i, scale, Ns_x[i])
        if x_grad_scale_mode == "nuLSQ_grad_scale":
            ratio = Ns_x/(Ns_x.sum())
        else:
            ratio = torch.ones_like(scale) 
            # ratio[2] = ratio[2]*0.5
        # ctx.Ns_x = Ns_x
        ctx.save_for_backward(x, scale, grad_scale, ratio )
        return y, Ns_x

    @staticmethod
    def backward(ctx, dLdy, temp_Ns_x):
        x, scale, grad_scale, ratio= ctx.saved_tensors
        # x, scale, num_elements = ctx.saved_tensors
        num_levels = torch.numel(scale) + 1
        # cum_scale = scale.cumsum(dim=0).tolist()
        # N_w = (x > 0).float().sum()
        # N_s = ((x > 0) & (x < cum_scale[0])).float().sum()
        

        flag_high   = x > scale.sum()
        flag_low    = x < 0
        flag_middle = (~flag_high) & (~flag_low)

        dydx = torch.ones_like(x) * flag_middle

        dLds = torch.zeros_like(scale)
        cumsum_scale1 = torch.tensor(0, dtype=torch.float).to(x.device)
        cumsum_scale2 = scale[0].clone()
        cumsum_scale3 = scale.sum()
        # Ns_tot = ((x > 0 ) & (x< cumsum_scale3)).float().sum()
        flag_upper =  cumsum_scale3 < x
        for i in range(num_levels-1):
            # if i > 0:
            #     N_s = ((x >= cum_scale[i-1]) & (x < cum_scale[i])).float().sum()
            flag_body  = (cumsum_scale1 < x) & (x < cumsum_scale2)

            shift_x = x - cumsum_scale1
            d1 = torch.heaviside(shift_x - scale[i]/2, torch.tensor(0.0, dtype=torch.float).to(x.device))
            d2 = -shift_x / scale[i]
            d  = d1 + d2  

            dyds_body  = flag_body  * (d)/torch.sqrt(ratio[i])
            dyds_upper = flag_upper * (1.0)/torch.sqrt(ratio[i])
            dyds       = (dyds_body + dyds_upper)


            if i < num_levels-2:
                cumsum_scale1 += scale[i+0]
                cumsum_scale2 += scale[i+1]
                
#             dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt((N_s/N_w) * (num_levels-1))) * torch.sqrt(torch.tensor((num_levels - 1), dtype=torch.float))
            # dLds[i] = (dLdy * dyds).sum().view(-1)
            dLds_body = (dLdy * dyds).sum().view(-1 ) * grad_scale
            # dLds[i] = dLds[i] + dLds_body
            for j in range(i+1):
                dLds[j] = dLds_body + dLds[j]

            #print(i, (1.0 / torch.sqrt(N_s * (num_levels-1))))
            # dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt(num_elements * (N_s/N_w)))
#             dLds[i] = (dLdy * dyds).sum().view(-1) * (1.0 / torch.sqrt((N_s/N_w) * (num_levels-1)))

        return dLdy * dydx, dLds, None, None, None


def _quantize_LSQ_non_uniform_nunlocal_update_V_act(x, scale, num_bits, num_elements, x_grad_scale_mode):
    # Note: This function assumes x >= 0. Therefore, only available for activations not for weights.
    func = _quantize_LSQ_non_uniform_nunlocal_update_V_act_core.apply
    return func(x, scale, num_bits, num_elements, x_grad_scale_mode)

#----------------------------------------------------------
# LSQ (non uniform version for activation) same update as LSQ
#----------------------------------------------------------
class _check_mode_quantize_LSQ_non_uniform_nunlocal_update_act_core(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, num_bits, num_elements, x_grad_scale_mode):

        num_levels = 2 ** num_bits
        for s in scale:
            assert s > 0, 'scale = {}'.format(scale)
        if num_elements > 0:
            if x_grad_scale_mode == "wo_grad_scale":
                grad_scale = torch.tensor(1.0).to(x.device)
            elif x_grad_scale_mode == "LSQ_grad_scale" or x_grad_scale_mode == "nuLSQ_grad_scale":
                grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float))
            elif x_grad_scale_mode == "10fac_LSQ_grad_scale" :
                grad_scale = 1.0 / (10*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            elif x_grad_scale_mode == "100fac_LSQ_grad_scale" :
                grad_scale = 1.0 / (100*torch.sqrt(torch.tensor(num_elements * (num_levels - 1), dtype=torch.float)))
            else:
                print("Not implemented:", x_grad_scale_mode)
        else:
            grad_scale = torch.tensor(1.0).to(x.device)

        y = torch.zeros_like(x)
        Ns_x = torch.zeros_like(scale)
        cumsum_scale = torch.tensor([0], dtype=torch.float).to(x.device)
        for i in range(num_levels-1):
            offset  = cumsum_scale + scale[i]/2
            y      += scale[i] * torch.heaviside(x - offset, torch.tensor(0.0, dtype=torch.float).to(x.device))
            Ns_x[i] = ((x > cumsum_scale) & (x < cumsum_scale + scale[i])).float().sum()
            cumsum_scale += scale[i]
            assert Ns_x[i] > 0, '(i, scale, Ns_x) = {}, {}, {}'.format(i, scale, Ns_x[i])
        if x_grad_scale_mode == "nuLSQ_grad_scale":
            ratio = Ns_x/(Ns_x.sum())
        else:
            ratio = torch.ones_like(scale) 
        ctx.save_for_backward(x, scale, grad_scale, ratio )
        return y, Ns_x

    @staticmethod
    def backward(ctx, dLdy, temp_Ns_x):
        x, scale, grad_scale, ratio= ctx.saved_tensors
        # x, scale, num_elements = ctx.saved_tensors
        num_levels = torch.numel(scale) + 1        

        flag_high   = x > scale.sum()
        flag_low    = x < 0
        flag_middle = (~flag_high) & (~flag_low)

        dydx = torch.ones_like(x) * flag_middle

        dLds = torch.zeros_like(scale)
        cumsum_scale1 = torch.tensor(0, dtype=torch.float).to(x.device)
        cumsum_scale2 = scale[0].clone()
        cumsum_scale3 = scale.sum()
        flag_upper =  cumsum_scale3 < x
        for i in range(num_levels-1):
            flag_body  = (cumsum_scale1 < x) & (x < cumsum_scale2)

            shift_x = x - cumsum_scale1
            d1 = torch.heaviside(shift_x - scale[i]/2, torch.tensor(0.0, dtype=torch.float).to(x.device))
            d2 = -shift_x / scale[i]
            d  = d1 + d2  

            dyds_body  = flag_body  * (d)/torch.sqrt(ratio[i])
            dyds_upper = flag_upper * (1.0)/torch.sqrt(ratio[i])
            dyds       = (dyds_body + dyds_upper)


            if i < num_levels-2:
                cumsum_scale1 += scale[i+0]
                cumsum_scale2 += scale[i+1]
                
            dLds_body = (dLdy * dyds_body).sum().view(-1 ) * grad_scale
            dLds[i] = dLds[i] + (dLdy * dyds_upper).sum().view(-1 ) * grad_scale
            for j in range(num_levels-1):
                dLds[j] = dLds_body + dLds[j]

        return dLdy * dydx, dLds, None, None, None


def _check_mode_quantize_LSQ_non_uniform_nunlocal_update_act(x, scale, num_bits, num_elements, x_grad_scale_mode):
    func = _check_mode_quantize_LSQ_non_uniform_nunlocal_update_act_core.apply
    return func(x, scale, num_bits, num_elements, x_grad_scale_mode)

#----------------------------------------------------------
# forward function for QLinear and QConv2d 
#----------------------------------------------------------
def _forward_common(module, input):
    # print(module.init_state)
    if module.init_state == False:
        # step size initialization of LSQ
        num_param_x = torch.numel(module.x_scale)
        dev         = input.device
        #LSQ+ initialization
        #w_std, w_mean = torch.std_mean(module.weight.detach().abs(), unbiased=False)
        #w_scale = float(torch.max(torch.abs(w_mean-3*w_std),torch.abs(w_mean+3*w_std))/2**(module.num_bits-1))
        #w_scale = 0.1W
        
        #mina = torch.min(input.detach())
        #maxi = torch.max(input.detach())
        #x_scale = (maxi - mina) / (module.x_Qp - module.x_Qn)
        #x_scale = 1.0
        
        
        # step size initialization for softplus LSQ
        if module.mode == "SoftPlus_LSQ":
            x_init = 2 * input.detach().abs().mean() / math.sqrt(module.x_Qp)
            w_init = 2 * module.weight.detach().abs().mean() / math.sqrt(module.w_Qp)
            x_scale     = float(math.log(math.exp(x_init) - 1))
            w_scale     = float(math.log(math.exp(w_init) - 1))
        
        # step size initialization for exp LSQ
        elif module.mode == "Exp_LSQ":
            x_scale     = float(math.log(2 * input.detach().abs().mean() / math.sqrt(module.x_Qp)))
            w_scale     = float(math.log(2 * module.weight.detach().abs().mean() / math.sqrt(module.w_Qp)))
        
        elif module.mode == "floorLSQ" or module.mode == "ceilLSQ"  or module.mode == "W_floorLSQ_A_LSQ":
            xmin = torch.min(input.detach())
            xmax = torch.max(input.detach())
            wmin = torch.min(module.weight.detach())
            wmax = torch.max(module.weight.detach())
            x_scale = (xmax - xmin) / (module.x_Qp + module.x_Qn)
            w_scale     = (wmax - wmin) / (module.x_Qp + module.x_Qn)
            print(x_scale, w_scale)
            # w_scale     = float(2 * module.weight.detach().abs().mean() / math.sqrt(module.w_Qp))
        elif module.mode == "shiftLSQ_first_layer":
            shift_const = (0.4914 + 0.4822 + 0.4465)/(0.2023 + 0.1994 + 0.2010)
            x_scale     = float(2 * (input.detach() - shift_const).abs().mean() / math.sqrt(module.x_Qp))
            w_scale     = float(2 * module.weight.detach().abs().mean() / math.sqrt(module.w_Qp))
        # step size initialization for default LSQ
        else:
            # if module.num_bits < 8:
            # if module.mode == "LSQ_first_layer":
            #     print(input.size())
            #     print(input)
            x_scale     = float(2 * input.detach().abs().mean() / math.sqrt(module.x_Qp))
            w_scale     = float(2 * module.weight.detach().abs().mean() / math.sqrt(module.w_Qp))
            # print(input)
            # else:
            #     print("8bits case")
            #     xmin = torch.min(input.detach())
            #     xmax = torch.max(input.detach())
            #     wmin = torch.min(module.weight.detach())
            #     wmax = torch.max(module.weight.detach())
            #     x_scale = (xmax - xmin) / (module.x_Qp + module.x_Qn)
            #     w_scale     = (wmax - wmin) / (module.x_Qp + module.x_Qn)
        # print('before Init_x:', module.x_scale)
        # print('before Init_w', module.w_scale)

        module.x_scale.data = torch.tensor([x_scale] * num_param_x).to(dev).clone()
        module.w_scale.data = torch.tensor([w_scale]              ).to(dev).clone()
        # print('Init_x:', module.x_scale)
        # # print('x = {}'.format(torch.max(input)))
        # # assert module.x_scale <2 , 'x = {}'.format(torch.max(input))
        # print('Init_w', module.w_scale)
        module.init_state.data = torch.tensor(True).to(dev).clone()
    
    if module.init_state_for_nonuniform_weight == False:
        # step size initialization of LSQ
        dev         = input.device
        #w_std, w_mean = torch.std_mean(module.weight, unbiased=False)
        #w_scale = float(torch.max(torch.abs(w_mean-3*w_std),torch.abs(w_mean+3*w_std))/2**(module.num_bits-1))

        #w_neg_scale     = float(2 * module.weight.abs().mean() / math.sqrt(module.w_Qn))
        
        if module.mode != "LSQ_non_uniform_first_layer":
            num_param_x = torch.numel(module.x_scale)
            x_scale     = float(2 * input.detach().abs().mean() / math.sqrt(module.x_Qp))
            w_scale     = float(2 * module.weight.detach().abs().mean() / math.sqrt(module.w_Qp))
            module.x_scale.data = torch.tensor([x_scale] * num_param_x).to(dev).clone()
            module.w_pos_scale.data = torch.tensor([w_scale] * module.w_Qp ).to(dev).clone()
            module.w_neg_scale.data = torch.tensor([w_scale] * module.w_Qn ).to(dev).clone()
        else:
            x_scale     = float(2 * input.detach().abs().mean() / math.sqrt(module.x_Qp))
            w_scale     = float(2 * module.weight.detach().abs().mean() / math.sqrt(module.w_Qp))
            # x_scale = utils._find_step_size_by_minimizing_quantization_error(input.detach(), x_scale, module.x_Qn, module.x_Qp, "LogCosh")
            # w_scale = _find_step_size_by_minimizing_quantization_error(input.detach(), x_scale.detach(), module.x_Qn, module.x_Qp, "L2")

            
            module.x_pos_scale.data = torch.tensor([x_scale] * module.x_Qp ).to(dev).clone()
            module.x_neg_scale.data = torch.tensor([x_scale] * module.x_Qn ).to(dev).clone()
            module.w_pos_scale.data = torch.tensor([w_scale] * module.w_Qp ).to(dev).clone()
            module.w_neg_scale.data = torch.tensor([w_scale] * module.w_Qn ).to(dev).clone()
        
        module.init_state_for_nonuniform_weight.data = torch.tensor(True).to(dev).clone()
    
    if module.mode == "real":
        # input : real-valued
        # weight: real-valued
        weight = module.weight
    elif module.mode == "LSQ":
        # input :     uniform quantization
        # weight:     uniform quantization
        input  = _quantize_LSQ(input,         module.x_scale, module.x_Qn, module.x_Qp, input.shape[1],module.x_grad_scale_mode)
        weight = _quantize_LSQ(module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
    elif module.mode == "LSQ_first_layer":
        input  = _quantize_LSQ(input,         module.x_scale, module.x_Qn, module.x_Qp, input.shape[1],"LSQ_grad_scale")
        weight = _quantize_LSQ(module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),"LSQ_grad_scale")
    elif module.mode == "shiftLSQ_first_layer":
        shift_const = (0.4914 + 0.4822 + 0.4465)/(0.2023 + 0.1994 + 0.2010)
        input  = _quantize_shiftLSQ(input,       shift_const,  module.x_scale, module.x_Qn, module.x_Qp, input.shape[1],"LSQ_grad_scale")
        weight = _quantize_LSQ(module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),"LSQ_grad_scale")        
    elif module.mode == "floorLSQ" :
        input  = _quantize_floorLSQ(input,         module.x_scale, module.x_Qn, module.x_Qp, input.numel(),module.x_grad_scale_mode)
        weight = _quantize_floorLSQ(module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
    elif module.mode == "ceilLSQ" :
        input  = _quantize_ceilLSQ(input,         module.x_scale, module.x_Qn, module.x_Qp, input.shape[1],module.x_grad_scale_mode)
        weight = _quantize_ceilLSQ(module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
    elif module.mode == "W_floorLSQ_A_LSQ" :
        input  = _quantize_LSQ(input,         module.x_scale, module.x_Qn, module.x_Qp, input.shape[1],module.x_grad_scale_mode)
        weight = _quantize_floorLSQ(module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
    elif module.mode == "SoftPlus_LSQ":
        input  = _quantize_SoftPlus_LSQ(input,         module.x_scale, module.x_Qn, module.x_Qp, input.shape[1])
        weight = _quantize_SoftPlus_LSQ(module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel())
    elif module.mode == "Exp_LSQ":
        input  = _quantize_Exp_LSQ(input,         module.x_scale, module.x_Qn, module.x_Qp, input.shape[1])
        weight = _quantize_Exp_LSQ(module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel())
    elif module.mode == "LSQ_non_uniform_only_activation":
        input, Ns_x  = _quantize_LSQ_non_uniform_act(input,         module.x_scale, module.num_bits, input.shape[1], module.x_grad_scale_mode)
        weight = _quantize_LSQ            (module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
        module.Ns_x.data += Ns_x
    elif module.mode == "LSQ_non_uniform_only_activation_fast":
        input, Ns_x  = _quantize_LSQ_non_uniform_fast_act(input,         module.x_scale, module.num_bits, input.shape[1], module.x_grad_scale_mode)
        weight = _quantize_LSQ            (module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
        module.Ns_x.data += Ns_x
    elif module.mode == "LSQ_non_uniform_only_activation_auto_grad":
        input, Ns_x  = _quantize_LSQ_non_uniform_autograd_act(input,         module.x_scale, module.num_bits, input.shape[1], module.x_grad_scale_mode)
        weight = _quantize_LSQ            (module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
        module.Ns_x.data += Ns_x
    elif module.mode == "LSQ_non_uniform_non_local_only_activation":
        input, Ns_x  = _quantize_LSQ_non_uniform_nunlocal_update_act(input,         module.x_scale, module.num_bits, input.shape[1], module.x_grad_scale_mode)
        weight = _quantize_LSQ            (module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
        module.Ns_x.data += Ns_x
    elif module.mode == "LSQ_non_uniform_non_local_only_activation_II":
        input, Ns_x  = _quantize_LSQ_non_uniform_nunlocal_update_II_act(input,         module.x_scale, module.num_bits, input.shape[1], module.x_grad_scale_mode)
        weight = _quantize_LSQ            (module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
        module.Ns_x.data += Ns_x
    elif module.mode == "LSQ_non_uniform_non_local_only_activation_III":
        input, Ns_x  = _quantize_LSQ_non_uniform_nunlocal_update_III_act(input,         module.x_scale, module.num_bits, input.shape[1], module.x_grad_scale_mode)
        weight = _quantize_LSQ            (module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
        module.Ns_x.data += Ns_x
    elif module.mode == "LSQ_non_uniform_non_local_only_activation_IV":
        input, Ns_x  = _quantize_LSQ_non_uniform_nunlocal_update_IV_act(input,         module.x_scale, module.num_bits, input.shape[1], module.x_grad_scale_mode)
        weight = _quantize_LSQ            (module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
        module.Ns_x.data += Ns_x
    elif module.mode == "LSQ_non_uniform_non_local_only_activation_V":
        input, Ns_x  = _quantize_LSQ_non_uniform_nunlocal_update_V_act(input,         module.x_scale, module.num_bits, input.shape[1], module.x_grad_scale_mode)
        weight = _quantize_LSQ            (module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
        module.Ns_x.data += Ns_x

    elif module.mode == "check_LSQ_non_uniform_non_local_only_activation":
        input, Ns_x  =  _check_mode_quantize_LSQ_non_uniform_nunlocal_update_act(input,         module.x_scale, module.num_bits, input.shape[1], module.x_grad_scale_mode)
        weight = _quantize_LSQ            (module.weight, module.w_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
        module.Ns_x.data += Ns_x

    elif module.mode == "LSQ_non_uniform_only_weight":
        input  = _quantize_LSQ(input,         module.x_scale, module.x_Qn, module.x_Qp, input.shape[1],module.x_grad_scale_mode, module.x_grad_scale_mode)
        weight = _quantize_LSQ_non_uniform_weight(module.weight,module.w_neg_scale,  module.w_pos_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
    elif module.mode == "LSQ_non_uniform_both_activation_weight":
        input  = _quantize_LSQ_non_uniform_act(input,         module.x_scale, module.num_bits, input.shape[1], module.x_grad_scale_mode)
        weight = _quantize_LSQ_non_uniform_weight(module.weight,module.w_neg_scale,  module.w_pos_scale, module.w_Qn, module.w_Qp, module.weight.numel(),module.w_grad_scale_mode)
    elif module.mode == "LSQ_non_uniform_first_layer":
        input  = _quantize_LSQ_non_uniform_weight(input,         module.x_neg_scale,  module.x_pos_scale,  module.x_Qn, module.x_Qp, input.shape[1],"LSQ_grad_scale")
        weight = _quantize_LSQ_non_uniform_weight(module.weight, module.w_neg_scale,  module.w_pos_scale,  module.w_Qn, module.w_Qp, module.weight.numel(),"LSQ_grad_scale")
    elif module.mode == "LSQ_x":
        # input :     uniform quantization
        # weight:     real-valued
        input  = _quantize_LSQ(input,         module.x_scale, module.x_Qn, module.x_Qp, input.shape[1],module.x_grad_scale_mode)
        weight = module.weight
    elif module.mode == "LSQ_x_non_uniform_only_activation":
        # input : non-uniform quantization
        # weight:     real-valued
        input  = _quantize_LSQ_non_uniform_act(input,         module.x_scale, module.num_bits, input.shape[1], module.x_grad_scale_mode)
        weight = module.weight

    # compute forward pass
    if   module.__class__.__name__ in ["QLinear"]:
        out = F.linear(input, weight, module.bias)
    elif module.__class__.__name__ in ["QConv2d"]:
        out = F.conv2d(input, weight, module.bias, module.stride, module.padding, module.dilation, module.groups)

    return out


#----------------------------------------------------------
# initializatoin for QLinear and QConv2d 
#----------------------------------------------------------
def _init_common(module, mode, num_bits):

    module.mode       = mode
    module.num_bits   = num_bits
    

    if mode == "real":
        module.x_Qn    = None
        module.x_Qp    = None
        module.w_Qn    = None
        module.w_Qp    = None
        module.register_buffer('init_state', torch.tensor(True))
        module.register_buffer('init_state_for_nonuniform_weight', torch.tensor(True))
    elif mode == "LSQ_first_layer" or mode == "shiftLSQ_first_layer": 
        module.w_Qn = 2 ** (num_bits-1)
        module.w_Qp = 2 ** (num_bits-1) - 1
        module.x_Qn = 2 ** (num_bits-1)
        module.x_Qp = 2 ** (num_bits-1) - 1
        module.x_scale = nn.Parameter(torch.tensor([0.0]))
        module.w_scale = nn.Parameter(torch.tensor([0.0]))
        module.register_buffer('init_state', torch.tensor(False))   
        module.register_buffer('init_state_for_nonuniform_weight', torch.tensor(True))
    elif module.mode == "LSQ_non_uniform_first_layer":
        module.w_Qn = 2 ** (num_bits-1)
        module.w_Qp = 2 ** (num_bits-1) - 1
        module.x_Qn = 2 ** (num_bits-1) 
        module.x_Qp = 2 ** (num_bits-1) -1
        module.x_pos_scale = nn.Parameter(torch.tensor([0.0] * module.x_Qp))
        module.x_neg_scale = nn.Parameter(torch.tensor([0.0] * module.x_Qp))
        module.w_pos_scale = nn.Parameter(torch.tensor([0.0] * module.w_Qp))
        module.w_neg_scale = nn.Parameter(torch.tensor([0.0] * module.w_Qn))
        module.register_buffer('init_state', torch.tensor(True))
        module.register_buffer('init_state_for_nonuniform_weight', torch.tensor(False))

    elif mode in ["LSQ","floorLSQ", "ceilLSQ", "W_floorLSQ_A_LSQ", "SoftPlus_LSQ", \
                  "Exp_LSQ", "LSQ_x", "LSQ_non_uniform_only_activation", "LSQ_non_uniform_only_activation_fast", \
                   "LSQ_non_uniform_only_activation_auto_grad", \
                    "LSQ_non_uniform_non_local_only_activation", "LSQ_non_uniform_non_local_only_activation_II", \
                        "LSQ_non_uniform_non_local_only_activation_III", "LSQ_non_uniform_non_local_only_activation_IV", \
                            "LSQ_non_uniform_non_local_only_activation_V", "LSQ_x_non_uniform_activation",\
                          "LSQ_non_uniform_only_weight","LSQ_non_uniform_both_activation_weight", \
                            "check_LSQ_non_uniform_non_local_only_activation"]:
        module.x_Qn = 0
        module.x_Qp = 2 ** (num_bits  ) - 1
        module.w_Qn = 2 ** (num_bits-1)
        module.w_Qp = 2 ** (num_bits-1) - 1
        if mode in ["LSQ_non_uniform_only_activation",  "LSQ_non_uniform_only_activation_fast", \
                    "LSQ_non_uniform_only_activation_auto_grad", "LSQ_x_non_uniform_only_activation", \
                    "LSQ_non_uniform_non_local_only_activation","LSQ_non_uniform_non_local_only_activation_II",\
                       "LSQ_non_uniform_non_local_only_activation_III", "LSQ_non_uniform_non_local_only_activation_IV",\
                        "LSQ_non_uniform_non_local_only_activation_V", "check_LSQ_non_uniform_non_local_only_activation"]:
            module.x_scale = nn.Parameter(torch.tensor([0.0] * module.x_Qp))
            module.w_scale = nn.Parameter(torch.tensor([0.0]))
            module.register_buffer('Ns_x',torch.tensor([0.0] * module.x_Qp))
            module.register_buffer('init_state', torch.tensor(False))
            module.register_buffer('init_state_for_nonuniform_weight', torch.tensor(True))
        elif mode in ["LSQ", "floorLSQ", "ceilLSQ", "W_floorLSQ_A_LSQ", "LSQ_x", "SoftPlus_LSQ", "Exp_LSQ",]:
            #w_scale = float((2 * module.weight.detach().abs().mean()) / (module.w_Qp**0.5))
            module.x_scale = nn.Parameter(torch.tensor([0.0]))
            module.w_scale = nn.Parameter(torch.tensor([0.0]))
            module.register_buffer('init_state', torch.tensor(False))   
            module.register_buffer('init_state_for_nonuniform_weight', torch.tensor(True))
        elif mode in ["LSQ_non_uniform_only_weight"]:
            module.x_scale = nn.Parameter(torch.tensor([0.0]))
            module.w_pos_scale = nn.Parameter(torch.tensor([0.0] * module.w_Qp))
            module.w_neg_scale = nn.Parameter(torch.tensor([0.0] * module.w_Qn))
            module.register_buffer('init_state', torch.tensor(True))
            module.register_buffer('init_state_for_nonuniform_weight', torch.tensor(False))            
        elif mode in ["LSQ_non_uniform_both_activation_weight"]:
            module.x_scale = nn.Parameter(torch.tensor([0.0] * module.x_Qp))
            module.w_pos_scale = nn.Parameter(torch.tensor([0.0] * module.w_Qp))
            module.w_neg_scale = nn.Parameter(torch.tensor([0.0] * module.w_Qn))
            module.register_buffer('init_state', torch.tensor(True))
            module.register_buffer('init_state_for_nonuniform_weight', torch.tensor(False))
            


#----------------------------------------------------------
# Fully connected layer 
#----------------------------------------------------------
class QLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, quant_mode="real", num_bits=None, w_grad_scale_mode = "LSQ_grad_scale", x_grad_scale_mode = "LSQ_grad_scale"):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        _init_common(self, quant_mode, num_bits)
        self.x_grad_scale_mode = x_grad_scale_mode
        self.w_grad_scale_mode = w_grad_scale_mode

    def forward(self, input):
        return _forward_common(self, input)


#----------------------------------------------------------
# convolutional layer
#----------------------------------------------------------
class QConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, quant_mode='real', num_bits=None, w_grad_scale_mode = "LSQ_grad_scale", x_grad_scale_mode = "LSQ_grad_scale"):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        _init_common(self, quant_mode, num_bits)
        self.x_grad_scale_mode = x_grad_scale_mode
        self.w_grad_scale_mode = w_grad_scale_mode

    def forward(self, input):
        return _forward_common(self, input)


