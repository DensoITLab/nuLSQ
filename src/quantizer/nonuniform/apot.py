import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#----------------------------------------------------------
# APoT (https://github.com/yhhhli/APoT_Quantization)
# Only B=2,3,4,5,6 are supported in original source code
#----------------------------------------------------------
class APoT_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(APoT_quantizer, self).__init__()
        self.params_set(module, num_bits, mode)
        self.first_layer = module.first_layer

    def params_set(self, module, num_bits, mode):
        self.bit = num_bits
        if mode == "activation":
            if module.first_layer:
                module.x_Qn = 2 ** (num_bits-1) - 1
                module.x_Qp = 2 ** (num_bits-1) - 1
                if self.bit > 2:
                    self.proj_set = build_power_value(B=self.bit - 1)
                    assert torch.isnan(self.proj_set) == False
                else:
                    self.proj_set = None
                self.is_weight = True
            else:
                module.x_Qn = 0
                module.x_Qp = 2 ** (num_bits  ) - 1
                self.proj_set = build_power_value(B=self.bit)
                self.is_weight = False
            module.x_threshold = nn.Parameter(torch.tensor([0.0]))
            module.x_Qparms['threshold'] = module.x_threshold

        elif mode == "weight":
            module.w_Qn = 2 ** (num_bits-1) - 1 # 1-bit for sign according to APoT definition
            module.w_Qp = 2 ** (num_bits-1) - 1
            module.w_threshold = nn.Parameter(torch.tensor([0.0]))
            module.w_Qparms['threshold'] = module.w_threshold
            if self.bit > 2:
                self.proj_set = build_power_value(B=self.bit - 1)
            else:
                self.proj_set = None
            self.is_weight = True
    
    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode = "none"):
        alpha = Qparms['threshold']
        if self.is_weight and self.bit == 2:
            return _uq.apply(x, alpha, Qn, Qp, num_elements, grad_scale_mode  )
        else:
            return _APoT_quantizer(x, alpha, self.proj_set,  Qn, Qp, num_elements, grad_scale_mode, self.is_weight)

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        dev = Qparms["init_scale"].device
        Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(dev)
        Qparms["threshold"].data = Qparms["init_scale"].clone().detach()*Qp_on_device


def _APoT_quantizer(x, alpha, proj_set,  Qn, Qp, num_elements, grad_scale_mode, is_weight):
    Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
    Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
    assert alpha > 0, 'threhold = {}, {}, {}'.format(alpha, Qn_on_device, Qp_on_device)

    def power_quant(x, value_s):
        if is_weight:
            shape = x.shape
            xhard = x.view(-1)
            sign = x.sign()
            value_s = value_s.type_as(x)
            xhard = xhard.abs()
            idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
            xhard = value_s[idxs].view(shape).mul(sign)
            xhard = xhard
        else:
            shape = x.shape
            xhard = x.view(-1)
            value_s = value_s.type_as(x)
            xhard = xhard
            idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
            xhard = value_s[idxs].view(shape)
            xhard = xhard
        xout = (xhard - x).detach() + x
        return xout

    if num_elements > 0:
        if grad_scale_mode == "wo_grad_scale":
            grad_scale = torch.tensor(1.0).to(x.device)
        elif grad_scale_mode == "10_fac":
            grad_scale = torch.tensor(10.0).to(x.device)
        elif grad_scale_mode == "LSQ_grad_scale":
            grad_scale = torch.sqrt(Qp_on_device / num_elements ) 
        else:
            grad_scale = torch.tensor(1.0).to(x.device)
        bw_alpha   = alpha * grad_scale

    alpha = (alpha - bw_alpha).detach() + bw_alpha
    data = x / alpha
    if is_weight:
        data = data.clamp(-1, 1)
        data_q = power_quant(data, proj_set)
        data_q = data_q * alpha
    else:
        data = data.clamp(0, 1)
        data_q = power_quant(data, proj_set)
        data_q = data_q * alpha

    return data_q
    
class _uq(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, alpha, Qn, Qp, num_elements, grad_scale_mode):
        assert alpha > 0, 'threhold = {}, {}, {}'.format(alpha, Qn, Qp)
        Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(input.device)
        input_div = input/alpha
        input_c = input_div.clamp(min=-1, max=1)
        input_q = input_c.round()
        if num_elements > 0:
            if grad_scale_mode == "10_fac":
                grad_scale = torch.tensor(10.0).to(input.device)
            elif grad_scale_mode == "LSQ_grad_scale":
                grad_scale = torch.sqrt(Qp_on_device / num_elements ) 
            else:
                grad_scale = torch.tensor(1.0).to(input.device)

        ctx.save_for_backward(input_div, input_q, grad_scale, alpha)
        input_q = input_q.mul(alpha)  # rescale to the original range

        return input_q

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()  # calibration: grad for weights will not be clipped
        input_div, input_q, grad_scale, alpha = ctx.saved_tensors
        i = (input_div.abs() > 1.).float()
        sign = input_div.sign()
        grad_alpha = (grad_output * (sign * i + (input_q - input_div) * (1 - i))).sum()
        grad_alpha = grad_alpha * grad_scale
        return grad_input, grad_alpha.unsqueeze(0), None, None, None, None


def build_power_value(B=2, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 1):
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    return values