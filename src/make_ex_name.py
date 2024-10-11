from src.quantizer.nonuniform import *
def make_ex_name(args):
    if args.w_quantizer == None:
        name_w_quantizer = "FP"
    else:  
        name_w_quantizer = args.w_quantizer.__name__
    if args.x_quantizer == None:
        name_x_quantizer = "FP"
    else:
        name_x_quantizer = args.x_quantizer.__name__

    name_quant_mode = "w_" + name_w_quantizer + "__x_" + name_x_quantizer

    if args.w_initializer == None:
        name_w_initializer = "FP"
    else:  
        name_w_initializer = args.w_initializer.__name__
    if args.x_initializer == None:
        name_x_initializer = "FP"
    else:
        name_x_initializer = args.x_initializer.__name__

    name_initializer = "w_" + name_w_initializer + "__x_" + name_x_initializer

    name_wd_for_shift = "wo_shift"
    if args.action == 'progressive_quantization':
        name_action = "from_{}bits_progressive_quantion".format(args.progressive_bits)
    elif args.action == 'load':
        name_action = "from_fp_model"
    elif args.action == 'resume':
        name_action = "from_fp_model"

    if args.different_optimizer_mode:
        name_optimizer ="{}_for_s_{}_for_others".format(args.step_size_optimizer, args.optimizer)
    else:
        name_optimizer ="{}_for_all_parms".format(args.optimizer)

    if args.lr_scheduler == "CosineAnnealing":
        name_lr_sch = "Cos"
    else:
        name_lr_sch = args.lr_scheduler

    if args.l1_reg:
        name_l1_reg = "w_{}_l1_reg_for_s_{}".format(args.l1_coeff,  args.l1_reg_scheduler)
    else:
        name_l1_reg = "wo_l1_reg_for_s"

    if args.osc_damp_scheduler== "CosineAnnealing":
        name_od_sch = "Cos"
    elif args.osc_damp_scheduler == "ConstWarmupCosineGrowth":
        name_od_sch = "CnstWmup_CosG"
    elif args.osc_damp_scheduler == "ConstWarmupCosineAnnealing":
        name_od_sch = "CnstWmup_Cos"
    elif args.osc_damp_scheduler == "Constant":
        name_od_sch = "Cnst"
    else:
        name_od_sch = args.lr_scheduler

    if args.osc_damp:
        name_osc_damp = "w_{}_od_{}".format(args.osc_damp_coeff,  name_od_sch)
    else:
        name_osc_damp = "wo_od"

    if args.weight_norm:
        name_wn = "w_wn"
    else:
        name_wn = "wo_wn"

    if args.nesterov:
        name_nesterov = "w_nv"
    else:
        name_nesterov = "wo_nv"

    if args.x_grad_scale_mode == "LSQ_grad_scale":
        name_x_grad_scale = "LSQ_gs"
    else:
        name_x_grad_scale = "wo_gs"

    if args.w_grad_scale_mode == "LSQ_grad_scale":
        name_w_grad_scale = "LSQ_gs"
    else:
        name_w_grad_scale = "wo_gs"

    ex_name = '{0}/{1}/{2}/{3}bit/{4}_{5}_lr{6}_wd{7}_xslr{8}_wslr{9}_xswd{10}_wswd{11}_w_{12}_x_{13}_{14}_init_num{15}_{16}_{17}_{18}_{19}_{20}_{21}epochs-{22}' \
    .format(args.model, name_initializer, name_quant_mode, str(args.num_bits), name_optimizer, name_lr_sch, \
    str(args.lr), str(args.weight_decay), str(args.x_step_size_lr ), str(args.w_step_size_lr ),
    str(args.x_step_size_wd),str(args.w_step_size_wd), \
            name_w_grad_scale,  name_x_grad_scale, name_wd_for_shift,  args.init_num, \
            name_action, name_l1_reg, name_osc_damp, name_wn, name_nesterov, args.nepochs, str(args.train_id))
    
 
    return ex_name