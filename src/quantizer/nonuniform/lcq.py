import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#----------------------------------------------------------
# LCQ (CVPR2021) 
# Learnable companding quantization for accurate low-bit neural networks
#----------------------------------------------------------
class LCQ_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode,  **kwargs):
        super(LCQ_quantizer, self).__init__()
        self.params_set(module, num_bits, mode)
        self.softmax = nn.Softmax(dim=0)
        self.bit = num_bits

    def params_set(self, module, num_bits, mode):
        self.K = 16
        self.delta = 1/self.K
        if mode == "activation":
            module.x_Qn = 0
            module.x_Qp = 2 ** (num_bits) - 1
            module.x_threshold = nn.Parameter(torch.tensor([0.0] ))
            module.x_theta = nn.Parameter(torch.tensor([0.0] * self.K ))
            module.x_dst = nn.Parameter(self.delta * torch.arange(self.K), requires_grad=False )
            module.x_Qparms['threshold'] = module.x_threshold
            module.x_Qparms['companding_params'] = module.x_theta
            module.x_Qparms["dst"] = module.x_dst
            self.is_weight = False
        elif mode == "weight":
            module.w_Qn = 2 ** (num_bits-1) - 1
            module.w_Qp = 2 ** (num_bits-1) - 1
            module.w_threshold = nn.Parameter(torch.tensor([0.0] ))
            module.w_Qparms['threshold'] = module.w_threshold
            if num_bits > 2:
                module.w_dst = nn.Parameter(self.delta * torch.arange(self.K), requires_grad=False )
                module.w_theta = nn.Parameter(torch.tensor([0.0] * self.K ))
                module.w_Qparms['companding_params'] = module.w_theta
                module.w_Qparms["dst"] = module.w_dst
            self.is_weight = True

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        alpha = Qparms['threshold']
        if self.is_weight and self.bit ==2:
            y = _uq.apply(x, alpha, Qn, Qp, num_elements, grad_scale_mode  )            
        
        else:
            alpha = Qparms['threshold']
            theta = Qparms['companding_params']
            dst =Qparms["dst"]

            theta_tmp = self.softmax(theta)
            gamma = theta_tmp/self.delta
            beta = torch.cumsum(theta_tmp, dim=0)[:self.K-1]
            beta = torch.cat([torch.tensor([0], device=x.device), beta])

            y = LCQ_fast_core.apply(x, alpha, gamma, beta,  dst, Qn, Qp, num_elements, grad_scale_mode)


        return y
    
    def scale_to_Qparms(self, Qparms, Qn, Qp):
        dev = Qparms["init_scale"].device
        Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(dev)
        Qparms["threshold"].data = Qparms["init_scale"].clone().detach()*Qp_on_device


class LCQ_fast_core(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha, gamma, beta,  dst, Qn, Qp, num_elements, grad_scale_mode,  box_size=1024*256, counting_num = False):
        device = x.device
        Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
        Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
        assert alpha > 0, 'alpha = {}, gamma = {}, beta={}: alpha_grad={}, gamma_grad ={}, beta_grad={}'.format(alpha, gamma, beta, alpha.grad, gamma.grad, beta.grad)
        for gamma_i, beta_i in zip(gamma, beta):
            assert (gamma_i >= 0) , 'alpha = {}, gamma = {}, beta={}: alpha_grad={}, gamma_grad ={}, beta_grad={}'.format(alpha, gamma, beta, alpha.grad, gamma.grad, beta.grad)
            assert (beta_i >= 0) and (beta_i <= 1.01), 'alpha = {}, gamma = {}, beta={}: alpha_grad={}, gamma_grad ={}, beta_grad={}'.format(alpha, gamma, beta, alpha.grad, gamma.grad, beta.grad)

        s = Qp_on_device

        if num_elements > 0:
            if grad_scale_mode in ["LSQ_grad_scale"] :
                grad_scale = torch.sqrt(Qp_on_device / num_elements ) 
            else:
                grad_scale= torch.tensor(1.0).to(device)
        else:
            grad_scale = torch.tensor(1.0).to(device)

        flag_middle = (torch.abs(x) < alpha).float()
        flag_high = 1 - flag_middle

        x_tmp = torch.abs(x)/alpha
        idx = torch.searchsorted(dst, x_tmp , right=True)
        y = compressing_func(x_tmp, idx, dst, gamma, beta)
        y_q = torch.round(y * s)/s 

        idx_q = torch.searchsorted(beta, y_q, right=True)
        z = expanding_func(y_q, idx_q, dst, gamma, beta)
        # z = y_q
        z = torch.sign(x)*alpha * (z * flag_middle + flag_high )
        ctx.save_for_backward(x, y_q, z, alpha, gamma, beta, dst, idx, idx_q, grad_scale, torch.tensor(box_size))

        return z
    
    @staticmethod
    def backward(ctx, dLdz):

        x, y_q, z, alpha, gamma, beta, dst, idx, idx_q, grad_scale, box_size = ctx.saved_tensors
        device = x.device
        
        x_tmp = torch.abs(x)/alpha
        flag_middle = (torch.abs(x) < alpha)
        flag_high = (~flag_middle)
        dLdx         = torch.where(flag_high, torch.tensor(0, dtype=x.dtype, device = device), dLdz)

        dzdalpha = z/alpha - x/alpha * flag_middle
        dLdalpha = torch.sum(dLdz * dzdalpha).view(-1) * grad_scale

        dst_idx = dst[idx-1]
        beta_idx_q = beta[idx_q-1]
        gamma_idx_q = gamma[idx_q-1]
        # dy_qdgamma  corresponds to dgdgamma in the LCQ paper
        # dz_qdgamma  corresponds to dQgdgamma in the LCQ paper
        dy_qdgamma_x = (x_tmp - dst_idx)/gamma_idx_q 
        dy_qdgamma_y_q =  (y_q - beta_idx_q)/gamma_idx_q**2 
        dzdgamma_x = torch.sign(x) * alpha * dy_qdgamma_x * flag_middle
        dzdgamma_y_q = torch.sign(x) * alpha * dy_qdgamma_y_q * flag_middle

        dy_qdbeta_x = 1/gamma_idx_q * flag_middle
        # dy_qdbeta_y_q = 1/gamma[idx_q-1] * flag_middle
        dzdbeta_x = torch.sign(x) * alpha * dy_qdbeta_x
        # dzdbeta_y_q = torch.sign(x) * alpha * dy_qdbeta_y_q


        box_size = int(box_size) # magic parameter for parallelize scatter_add_

        num_intervals = gamma.numel() 
        N          = int(box_size / num_intervals)
        M_residual = int(x.numel() % box_size)
        M_parallel = x.numel() - M_residual

        dLdgamma = scatter_split_sum(\
            dLdz, dzdgamma_x, idx, M_parallel, M_residual, num_intervals, N, device)

        dLdbeta = scatter_split_sum(\
            dLdz, dzdbeta_x, idx, M_parallel, M_residual, num_intervals, N, device)

        dLdgamma = dLdgamma - scatter_split_sum(\
            dLdz, dzdgamma_y_q, idx_q, M_parallel, M_residual, num_intervals, N, device)

        dLdbeta = dLdbeta - scatter_split_sum(\
            dLdz, dzdbeta_x, idx_q, M_parallel, M_residual, num_intervals, N, device)

        dLdgamma = dLdgamma * grad_scale
        dLdbeta = dLdbeta * grad_scale

        assert torch.isnan(dLdx).sum() == 0, \
            "dLdx is nan, dLdz = {}, dLdx = {},x={}, y_q={}, z={}, idx={}, idx_q={}, flag_high={}, alpha={}, beta={}, gamma ={}".\
                format(dLdz, dLdx, x, y_q, z, idx, idx_q, flag_high, alpha,beta, gamma)
        assert torch.isnan(dLdalpha).sum() == 0, \
            "dLdalpha is nan, dLdz = {},dLdx = {},x={}, y_q={}, z={}, idx={}, idx_q={}, dzdalpha = {}, alpha={}, beta={}, gamma ={}".\
                format(dLdz,  dLdx, x, y_q, z, idx, idx_q, dzdalpha, alpha,beta, gamma)
        assert torch.isnan(dLdgamma).sum() == 0, \
            "dLdgamma is nan, dLdz = {}, dLdx = {},x={}, y_q={}, z={}, idx={}, idx_q={}, dzdgamma_x = {}, dy_qdgamma_y_q={}, alpha={}, beta={}, gamma ={}"\
                .format(dLdz,  dLdx, x, y_q, z, idx, idx_q, dzdgamma_x, dy_qdgamma_y_q, alpha,beta, gamma)
        assert torch.isnan(dLdbeta).sum() == 0, \
            "dLdbeta is nan, dLdz = {}, dLdx = {},x={}, y_q={}, z={}, idx={}, idx_q={}, dzdbeta_x = {}, alpha={}, beta={}, gamma ={}"\
                .format(dLdz, dLdx, x, y_q, z, idx, idx_q, dzdbeta_x, alpha,beta, gamma)
        return dLdx, dLdalpha, dLdgamma, dLdbeta, None, None, None, None, None

class LCQ_core(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha, gamma, beta,  dst, Qn, Qp, num_elements, grad_scale_mode,  box_size=1024*256, counting_num = False):
        device = x.device
        Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
        Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)
        assert alpha > 0, 'alpha = {}, gamma = {}, beta={}: alpha_grad={}, gamma_grad ={}, beta_grad={}'.format(alpha, gamma, beta, alpha.grad, gamma.grad, beta.grad)
        for gamma_i, beta_i in zip(gamma, beta):
            assert (gamma_i >= 0) , 'alpha = {}, gamma = {}, beta={}: alpha_grad={}, gamma_grad ={}, beta_grad={}'.format(alpha, gamma, beta, alpha.grad, gamma.grad, beta.grad)
            assert (beta_i >= 0) and (beta_i <= 1.01), 'alpha = {}, gamma = {}, beta={}: alpha_grad={}, gamma_grad ={}, beta_grad={}'.format(alpha, gamma, beta, alpha.grad, gamma.grad, beta.grad)

        s = Qp_on_device

        if num_elements > 0:
            if grad_scale_mode in ["LSQ_grad_scale"] :
                grad_scale = torch.sqrt(Qp_on_device / num_elements ) 
            else:
                grad_scale= torch.tensor(1.0).to(device)
        else:
            grad_scale = torch.tensor(1.0).to(device)

        flag_middle = (torch.abs(x) < alpha).float()
        flag_high = 1 - flag_middle

        x_tmp = torch.abs(x)/alpha
        idx = torch.searchsorted(dst, x_tmp , right=True)
        y = compressing_func(x_tmp, idx, dst, gamma, beta)
        y_q = torch.round(y * s)/s 

        idx_q = torch.searchsorted(beta, y_q, right=True)
        z = expanding_func(y_q, idx_q, dst, gamma, beta)
        # z = y_q
        z = torch.sign(x)*alpha * (z * flag_middle + flag_high )
        ctx.save_for_backward(x, y_q, z, alpha, gamma, beta, dst, idx, idx_q, grad_scale, torch.tensor(box_size))
        Ns_x = torch.zeros_like(alpha)

        return z, Ns_x
    
    @staticmethod
    def backward(ctx, dLdz, Ns_x_tmp):

        x, y_q, z, alpha, gamma, beta, dst, idx, idx_q, grad_scale, box_size = ctx.saved_tensors
        device = x.device

        flag_middle = (torch.abs(x) < alpha)
        flag_high = (~flag_middle)
        dLdx         = torch.where(flag_high, torch.tensor(0, dtype=x.dtype, device = device), dLdz)

        dzdalpha = z/alpha - x/alpha * flag_middle
        dLdalpha = torch.sum(dLdz * dzdalpha).view(-1) * grad_scale
        # dy_qdgamma  corresponds to dgdgamma in the LCQ paper
        # dz_qdgamma  corresponds to dQgdgamma in the LCQ paper
        dy_qdgamma_x = ((torch.abs(x)/alpha-dst[idx-1])/gamma[idx_q-1] )* flag_middle
        dy_qdgamma_y_q =  (y_q - beta[idx_q-1])/gamma[idx_q-1]**2 * flag_middle
        dzdgamma_x = torch.sign(x) * alpha * dy_qdgamma_x
        dzdgamma_y_q = torch.sign(x) * alpha * dy_qdgamma_y_q

        dy_qdbeta_x = 1/gamma[idx_q-1] * flag_middle
        dzdbeta_x = torch.sign(x) * alpha * dy_qdbeta_x


        box_size = int(box_size) # magic parameter for parallelize scatter_add_

        num_intervals = gamma.numel() 
        N          = int(box_size / num_intervals)
        M_residual = int(x.numel() % box_size)
        M_parallel = x.numel() - M_residual

        dLdgamma = scatter_split_sum(\
            dLdz, dzdgamma_x, idx, M_parallel, M_residual, num_intervals, N, device)

        dLdbeta = scatter_split_sum(\
            dLdz, dzdbeta_x, idx, M_parallel, M_residual, num_intervals, N, device)

        dLdgamma = dLdgamma - scatter_split_sum(\
            dLdz, dzdgamma_y_q, idx_q, M_parallel, M_residual, num_intervals, N, device)

        dLdbeta = dLdbeta - scatter_split_sum(\
            dLdz, dzdbeta_x, idx_q, M_parallel, M_residual, num_intervals, N, device)

        dLdgamma = dLdgamma * grad_scale
        dLdbeta = dLdbeta * grad_scale

        assert torch.isnan(dLdx).sum() == 0, \
            "dLdx is nan, dLdz = {}, dLdx = {},x={}, y_q={}, z={}, idx={}, idx_q={}, flag_high={}, alpha={}, beta={}, gamma ={}".\
                format(dLdz, dLdx, x, y_q, z, idx, idx_q, flag_high, alpha,beta, gamma)
        assert torch.isnan(dLdalpha).sum() == 0, \
            "dLdalpha is nan, dLdz = {},dLdx = {},x={}, y_q={}, z={}, idx={}, idx_q={}, dzdalpha = {}, alpha={}, beta={}, gamma ={}".\
                format(dLdz,  dLdx, x, y_q, z, idx, idx_q, dzdalpha, alpha,beta, gamma)
        assert torch.isnan(dLdgamma).sum() == 0, \
            "dLdgamma is nan, dLdz = {}, dLdx = {},x={}, y_q={}, z={}, idx={}, idx_q={}, dzdgamma_x = {}, dy_qdgamma_y_q={}, alpha={}, beta={}, gamma ={}"\
                .format(dLdz,  dLdx, x, y_q, z, idx, idx_q, dzdgamma_x, dy_qdgamma_y_q, alpha,beta, gamma)
        assert torch.isnan(dLdbeta).sum() == 0, \
            "dLdbeta is nan, dLdz = {}, dLdx = {},x={}, y_q={}, z={}, idx={}, idx_q={}, dzdbeta_x = {}, alpha={}, beta={}, gamma ={}"\
                .format(dLdz, dLdx, x, y_q, z, idx, idx_q, dzdbeta_x, alpha,beta, gamma)
        # timestamp = stop_watch(timestamp, "dLdgamma_second term")
        return dLdx, dLdalpha, dLdgamma, dLdbeta, None, None, None, None, None        

def scatter_split_sum(dLdz, dzdp, idx, M_parallel, M_residual, num_intervals, N, device):
    idx1, idx2 = torch.split(               idx.view(-1), (M_parallel, M_residual))
    val1, val2 = torch.split((dLdz * dzdp).view(-1), (M_parallel, M_residual))

    dLdp1 = torch.zeros((num_intervals, N)).to(device)
    
    idx1  = idx1.view(-1, N)
    val1  = val1.view(-1, N)
    dLdp1.scatter_add_(0, idx1-1, val1)
    dLdp1 = dLdp1.sum(1).view(-1)

    dLdp2 = torch.zeros(num_intervals).to(device)
    if M_residual > 0:
        dLdp2.scatter_add_(0, idx2-1, val2)
    dLdp  = (dLdp1 + dLdp2)

    return dLdp


def compressing_func (x, idx, d, gamma, beta):
    y = gamma[idx-1] * (x - d[idx-1]) + beta[idx-1]
    return y

def expanding_func (x, idx, d, gamma, beta):
    y =  (x - beta[idx-1])/gamma[idx-1] + d[idx-1]
    return y

class _uq(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha, Qn, Qp, num_elements, grad_scale_mode):
        assert alpha > 0, 'threhold = {}, {}, {}'.format(alpha, Qn, Qp)
        device = x.device
        Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(device)

        if num_elements > 0:
            if grad_scale_mode == "10_fac":
                grad_scale = torch.tensor(10.0).to(device)
            elif grad_scale_mode == "LSQ_grad_scale":
                grad_scale = torch.sqrt(Qp_on_device / num_elements ) 
            else:
                grad_scale = torch.tensor(1.0).to(device)

        flag_middle = (torch.abs(x) < alpha).float()
        flag_high = 1 - flag_middle

        x_tmp = torch.abs(x)/alpha
        y_q = torch.round(x_tmp)

        z = torch.sign(x)*alpha * (y_q * flag_middle + flag_high )
        ctx.save_for_backward(x, z, grad_scale, alpha)

        return z

    @staticmethod
    def backward(ctx, dLdz):
        x, z, grad_scale, alpha = ctx.saved_tensors
        x_tmp = torch.abs(x)/alpha
        device = x.device
        flag_middle = (torch.abs(x_tmp ) < 1)
        flag_high = (~flag_middle)
        dLdx         = torch.where(flag_high, torch.tensor(0, dtype=x.dtype, device = device), dLdz)

        dzdalpha = z/alpha - x/alpha * flag_middle
        dLdalpha = torch.sum(dLdz * dzdalpha).view(-1) * grad_scale

        return dLdx, dLdalpha, None, None, None, None

