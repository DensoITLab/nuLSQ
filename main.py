#========================================================================
# import modules 
#========================================================================


import os
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

# torchvision
import torchvision
import module_lsq as Q
from models.model import create_model
from configs import *

import copy
import numpy as np
from utils import *
from functools import partial

import math
import time
import warnings
import lr_scheduler

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def run_one_epoch(net, dataloader, optimizer, soptimizer, criterion, epoch, mode, args):
    global best_acc
    if mode == "train":
        net.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    total_loss        = 0.0
    total_num_sample  = 0
    total_num_correct = 0
    # print(args.nepochs)
    if args.calculate_A:
        os.makedirs(args.save_path, exist_ok=True)
        write_experiment_params(args, os.path.join(args.save_path, args.model+'.txt'))
        writer = SummaryWriter(os.path.join(args.save_path, 'tf_log'))
        Ax_list = []
        if args.quant_mode == 'LSQ_non_uniform_both_activation_weight':
            Aw_pos_list = []        
            Aw_neg_list = []        
        else:
            Aw_list = []

    
    # setup progress bar
    with tqdm(total=len(dataloader[mode]), disable=True) as pbar:
        pbar.set_description(f"Epoch[{epoch}/{args.nepochs}]({mode})")

        # loop for each epoch
        for i, (inputs, labels) in enumerate(dataloader[mode]):

            inputs, labels = inputs.to(args.device), labels.to(args.device)
            # if i == 0 and mode == "train" and epoch == 0:
            #     os.makedirs(args.save_path, exist_ok=True)
            #     writer = SummaryWriter(os.path.join(args.save_path, 'tf_log'))
            #     print("writing histogram:", args.save_path)
            #     writer.add_histogram("histogram/input_data",    inputs)
            #     writer.close()



            # run train/val/test
            if mode == "train":
                optimizer.zero_grad()
                if args.different_optimizer_mode:
                    soptimizer.zero_grad()
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                if args.calculate_A:
                    if args.dataset_name == 'imagenet':
                        if i > 500 :
                            break
                    temp_x = []
                    temp_w = []
                    if args.quant_mode == 'LSQ_non_uniform_both_activation_weight':
                        temp_wpos = []
                        temp_wneg = []
                    else:
                        temp_w = []
                    for (name, param) in net.named_parameters():
                        if "x_scale" in name:
                            each_Ax = torch.abs(param.grad)/param
                            temp_x.append(torch.unsqueeze(each_Ax, dim=0))
                        elif "w_scale" in name:
                            each_Aw = torch.abs(param.grad)/param
                            temp_w.append(torch.unsqueeze(each_Aw, dim=0))
                        elif "w_pos_scale" in name:
                            each_Aw = torch.abs(param.grad)/param
                            temp_wpos.append(torch.unsqueeze(each_Aw, dim=0))
                        elif "w_neg_scale" in name:
                            each_Aw = torch.abs(param.grad)/param
                            temp_wneg.append(torch.unsqueeze(each_Aw, dim=0))
                    temp_x = torch.cat(temp_x, dim=0)
                    Ax_list.append(torch.unsqueeze(temp_x, dim=0))
                    if args.quant_mode == 'LSQ_non_uniform_both_activation_weight':
                        temp_wpos = torch.cat(temp_wpos, dim=0)
                        temp_wneg = torch.cat(temp_wneg, dim=0)
                        Aw_pos_list.append(torch.unsqueeze(temp_wpos, dim=0))
                        Aw_neg_list.append(torch.unsqueeze(temp_wneg, dim=0))
                    else:
                        temp_w = torch.cat(temp_w, dim=0)
                        Aw_list.append(torch.unsqueeze(temp_w, dim=0))

                optimizer.step()
                if args.different_optimizer_mode:
                    soptimizer.step()

            else:
                outputs = net(inputs).cuda()
                loss = criterion(outputs, labels)
                # loss = net.module.compute_loss(outputs, labels)

            # statistics
            _, predicted       = torch.max(outputs.detach(), 1)
            num_sample         = labels.size(0)
            num_correct        = int((predicted == labels).sum())    
            total_loss        += loss.detach() * num_sample
            total_num_sample  += num_sample
            total_num_correct += num_correct

            # update progress bar
            pbar.update(1)

    avg_loss = total_loss        / total_num_sample
    accuracy = total_num_correct / total_num_sample
    if args.calculate_A and mode == "train":
        grad_LSQ_w = []
        grad_LSQ_x = []
        for (name, param) in net.named_parameters():
            if name in ['features.init_block', 'output', 'features.init_block.conv.weight']:
                continue
            if  "conv.weight" in name:
                grad_LSQ_w.append(torch.unsqueeze(1.0 /torch.sqrt(torch.tensor(param.numel())), dim=0))
                grad_LSQ_x.append(torch.unsqueeze(1.0 /torch.sqrt(torch.tensor(param.size(0))), dim=0))
        grad_LSQ_w = torch.cat(grad_LSQ_w, dim=0)
        grad_LSQ_x = torch.cat(grad_LSQ_x, dim=0)
        print("grad_LSQ_w:", grad_LSQ_w.size())
        print(grad_LSQ_w)
        print("grad_LSQ_x:", grad_LSQ_w.size())
        print(grad_LSQ_x)



        os.makedirs(args.save_path, exist_ok=True)
        Ax_list = torch.cat(Ax_list, dim=0)
        Axmean = torch.mean(Ax_list, dim=0)
        Axstd = torch.std(Ax_list, dim=0)
        torch.save(Axmean, os.path.join(args.save_path, 'Ax_mean.pt'))
        torch.save(Axstd, os.path.join(args.save_path, 'Ax_std.pt'))
        if args.quant_mode == 'LSQ_non_uniform_both_activation_weight':
            Aw_pos_list = torch.cat(Aw_pos_list, dim=0)
            Aw_neg_list = torch.cat(Aw_neg_list, dim=0)
            Awpos_mean = torch.mean(Aw_pos_list, dim=0)
            Awneg_mean = torch.mean(Aw_neg_list, dim=0)
            Awpos_std = torch.std(Aw_pos_list, dim=0)
            Awneg_std = torch.std(Aw_neg_list, dim=0)
            torch.save(Awpos_mean, os.path.join(args.save_path, 'Awpos_mean.pt'))
            torch.save(Awneg_mean, os.path.join(args.save_path, 'Awneg_mean.pt'))
            torch.save(Awpos_std, os.path.join(args.save_path, 'Awpos_std.pt'))
            torch.save(Awneg_std, os.path.join(args.save_path, 'Awneg_std.pt'))
            torch.save(grad_LSQ_w, os.path.join(args.save_path, 'grad_LSQ_w.pt'))
            torch.save(grad_LSQ_x, os.path.join(args.save_path, 'grad_LSQ_x.pt'))
        else:
            Aw_list = torch.cat(Aw_list, dim=0)
            Awmean = torch.mean(Aw_list, dim=0)
            Awstd = torch.std(Aw_list, dim=0)
            torch.save(Awmean, os.path.join(args.save_path, 'Aw_mean.pt'))
            torch.save(Awstd, os.path.join(args.save_path, 'Aw_std.pt'))
            torch.save(grad_LSQ_w, os.path.join(args.save_path, 'grad_LSQ_w.pt'))
            torch.save(grad_LSQ_x, os.path.join(args.save_path, 'grad_LSQ_x.pt'))
    if args.write_log and accuracy > best_acc and mode != 'train':
        if args.info_outputs:
            print('Saving..')
        save_ckp(net, None, None, best_acc, epoch, best_acc, filename=os.path.join(args.save_path, 'best.pth'))
        best_acc = accuracy
    return accuracy, avg_loss

def run_test(args):
    print('==> Preparing testing for {0}..'.format(args.dataset_name))
    assert args.dataset_name in ["cifar", "imagenet"]
    if args.dataset_name == 'cifar':
        cifar_config(args)
        if args.calculate_A:
            args.nepochs = 1
            args.write_log = False

        dataloader = cifar_dataloaders(args.batch_size, args.nworkers)
        net = create_model(args)
    elif args.dataset_name == 'imagenet':
        imagenet_config(args)
        dataloader = imagenet_dataloaders(args.batch_size, args.nworkers)
        net = create_model(args)
        
    assert net != None
    
    net = net.to(args.device)
    if args.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        
    criterion = nn.CrossEntropyLoss()
        
    val_accuracy, val_loss = run_one_epoch(net, dataloader, None, criterion, 0, "test", args)
    print('test_Loss: %.3f, test_Acc: %.3f' % (val_loss.item(), val_accuracy))
    
    

def run_train(args):
    global best_acc
    start_epoch = 0
    if args.info_outputs:      
        print('==> Preparing training for {0}..'.format(args.dataset_name))
    assert args.dataset_name == "cifar" or args.dataset_name == "imagenet"
    if args.dataset_name == 'cifar':
        cifar_config(args)
        if args.calculate_A:
            args.nepochs = 1
            args.write_log = False
        dataloader = cifar_dataloaders(args.batch_size, args.nworkers)
        net = create_model(args)
    elif args.dataset_name == 'imagenet':
        imagenet_config(args)
        if args.calculate_A:
            args.nepochs = 1
            args.write_log = False
        dataloader = imagenet_dataloaders(args.batch_size, args.nworkers)
        net = create_model(args)

    if args.weight_decay_for_scale:
        name_wd_for_scale = "w_wd_for_scale"
    else:
        name_wd_for_scale = "wo_wd_for_scale"
    if args.action == 'progressive_quantization':
        name_action = "from_{}bits_progressive_quantion".format(args.progressive_bits)
    elif args.action == 'load':
        name_action = "from_float_model"

    if args.different_optimizer_mode:
        name_optimizer ="{}_for_stepsize_{}_for_other_parms".format(args.step_size_optimizer, args.optimizer)
    else:
        name_optimizer ="{}_for_all_parms".format(args.optimizer)

    if args.calculate_A:
        args.ex_name = 'calculate_A_{0}_{1}bit_{2}_{3}_lr{4}_{5}_w_{6}_x_{7}_{8}-{9}'.format(args.model, str(args.num_bits), args.optimizer, args.lr_scheduler, str(args.lr), name_wd_for_scale, args.w_grad_scale_mode,args.x_grad_scale_mode, name_action, str(args.train_id))
    else:

        # args.ex_name = '{0}/{0}_{1}bit_{2}_{3}_lr{4}_wd_{5}_{6}_w_{7}_x_{8}_{9}_w_{10}bit_first_symmetric_{11}_Logcosh_init_last_{12}_-{13}'\
        args.ex_name = 'speed_check/{0}/first_{12}_last_{13}/{0}_{1}bit_{2}_{3}_lr{4}_wd{5}_slr{6}_{7}_w_{8}_x_{9}_{10}_w_{11}bit_first_symmetric_{12}_last_{13}_epoch{14}-{15}'\
            .format(args.model, str(args.num_bits), name_optimizer, args.lr_scheduler, \
                    str(args.lr), str(args.weight_decay), \
                        str(args.step_size_lr ), name_wd_for_scale, \
                        args.w_grad_scale_mode,  args.x_grad_scale_mode, name_action, \
                            str(args.first_last_num_bits), args.quant_mode_at_first_layer, \
                                args.quant_mode_at_last_layer, args.nepochs, str(args.train_id))
        # args.ex_name = '{0}_{1}bit_{2}_{3}_lr{4}_{5}_w_{6}_x_{7}_{8}_wo_correction_factor-{9}'.format(args.model, str(args.num_bits), args.optimizer, args.lr_scheduler, str(args.lr),name_wd_for_scale, args.w_grad_scale_mode,  args.x_grad_scale_mode, name_action, str(args.train_id))
    print(args.quant_mode)
    args.save_path = os.path.join(args.save_path, args.dataset_name, args.quant_mode, args.ex_name)
    if args.write_log:
        os.makedirs(args.save_path, exist_ok=True)
        write_experiment_params(args, os.path.join(args.save_path, args.model+'.txt'))
        writer = SummaryWriter(os.path.join(args.save_path, 'tf_log'))
    assert net != None
    
    if args.info_outputs:
        print (net)
        print ("Number of learnable parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6, "M")   
        time.sleep(5)
    
    if args.quant_mode != 'real':
        if args.info_outputs:
            print("==> Replacing model parameters..")
        replacement_dict = {
                            nn.Conv2d : partial(Q.QConv2d, quant_mode=args.quant_mode, num_bits=args.num_bits, w_grad_scale_mode = args.w_grad_scale_mode, x_grad_scale_mode = args.x_grad_scale_mode), 
                            nn.Linear: partial(Q.QLinear, quant_mode=args.quant_mode, num_bits=args.num_bits, w_grad_scale_mode = args.w_grad_scale_mode, x_grad_scale_mode = args.x_grad_scale_mode)}
        exception_dict = {
                '__first__': partial(Q.QConv2d,\
                                      quant_mode=args.quant_mode_at_first_layer, x_grad_scale_mode = args.x_grad_scale_mode,\
                                          w_grad_scale_mode =  args.w_grad_scale_mode, num_bits=args.first_last_num_bits),
                '__last__': partial(Q.QLinear, \
                                    quant_mode=args.quant_mode_at_last_layer, x_grad_scale_mode = args.x_grad_scale_mode, \
                                        w_grad_scale_mode =  args.w_grad_scale_mode, num_bits=args.first_last_num_bits),
                '__last_for_mobilenet__': partial(Q.QConv2d, quant_mode=args.quant_mode_at_first_layer, num_bits=args.first_last_num_bits),
                # '__first__': partial(Q.QConv2d, quant_mode='real', num_bits=8),
                # '__last__': partial(Q.QLinear, quant_mode='real', num_bits=8),
                # '__last_for_mobilenet__': partial(Q.QConv2d, quant_mode='real', num_bits=8),
            }        
        net = replace_module(net, replacement_dict=replacement_dict, exception_dict=exception_dict, arch=args.model)

    net = net.to(args.device)

    if args.info_outputs:              
        print('==> Performing action: ', args.action)
    if args.action == 'load':
        if args.init_from and os.path.isfile(args.init_from):
            if args.info_outputs:
                print('==> Initializing from checkpoint: ', args.init_from)
            checkpoint = torch.load(args.init_from)
            loaded_params = {}
            for k,v in checkpoint.items():
                loaded_params[k] = v
            net_state_dict = net.state_dict()
            net_state_dict.update(loaded_params)
            net.load_state_dict(net_state_dict)
            stepsize_init(net, dataloader, 'train', args.device)
        else:
            warnings.warn("No checkpoint file is provided !!!")
            exit(-1)
    elif args.action == 'progressive_quantization':
        print('==> load_model: ', args.init_from)
        if args.quant_mode == "LSQ" or args.quant_mode == "floorLSQ" or args.quant_mode == "ceilLSQ" or args.quant_mode == "W_floorLSQ_A_LSQ":
            load_epoch, net, _, _, best_load_acc, load_acc = load_ckp( args.init_from, net, None, None)
            # net_state_dict = net.state_dict()
            print("load epoch, load acc, best load acc:", load_epoch, load_acc, best_load_acc)

            for idx, (name, m) in enumerate(net.named_modules()):
                # print(name, type(m))
                print(name, type(m))
                if type(m) in (Q.QLinear, Q.QConv2d):
                    if m.mode == "LSQ" :
                        print(m.x_scale)
                        print(m.w_scale)
                        m.x_scale.data =  m.x_scale * (2 ** args.progressive_bits - 1)/(2**args.num_bits -1)
                        m.w_scale.data =  m.w_scale * (2 ** (args.progressive_bits -1) - 1)/(2** (args.num_bits-1) -1)
                        print(m.x_scale)
                        print(m.w_scale)
                    # print(m.mode)
                    # print(m.init_state)
                        # m.init_state.data = torch.tensor(False).to(args.device).clone()
                    elif m.mode == "floorLSQ" or m.mode == "ceilLSQ" or m.mode == "W_floorLSQ_A_LSQ":
                        print("before")
                        print(m.x_scale)
                        print(m.w_scale)
                        m.x_scale.data =  m.x_scale * (2 ** args.progressive_bits )/(2** args.num_bits )
                        m.w_scale.data =  m.w_scale * (2 ** (args.progressive_bits -1) )/(2 ** (args.num_bits-1) )
                        print("after")
                        print(m.x_scale)
                        print(m.w_scale)
            # stepsize_init(net, dataloader, 'train', args.device)
        elif args.quant_mode == "LSQ_non_uniform_only_activation" or args.quant_mode ==  "LSQ_non_uniform_only_activation_fast" \
                or args.quant_mode ==  "LSQ_non_uniform_only_activation_auto_grad" or args.quant_mode == "LSQ_non_uniform_non_local_only_activation" \
                or args.quant_mode == "LSQ_non_uniform_non_local_only_activation_II" or args.quant_mode == "LSQ_non_uniform_non_local_only_activation_III" \
                or args.quant_mode == "LSQ_non_uniform_non_local_only_activation_IV" or args.quant_mode == "LSQ_non_uniform_non_local_only_activation_V"  \
                    or args.quant_mode == "check_LSQ_non_uniform_non_local_only_activation":
            checkpoint = torch.load(args.init_from)
            loaded_params = {}
            for k,v in checkpoint['state_dict'].items():
                if "x_scale" in k:
                    temp = v.tolist()
                    x_scale = [temp[i-1] + temp[i] for i in range(1, len(temp)-1, 2)]
                    loaded_params[k] = torch.tensor(x_scale).to(args.device)
                else:
                    loaded_params[k] = v
            net_state_dict = net.state_dict()
            net_state_dict.update(loaded_params)
            net.load_state_dict(net_state_dict)
            # for idx, (name, m) in enumerate(net.named_modules()):
            #     if type(m) in (Q.QLinear, Q.QConv2d):
            #         if m.mode == "LSQ_non_uniform_only_activation" :
            #             m.init_state.data = torch.tensor(False).to(args.device).clone()
            # stepsize_init(net, dataloader, 'train', args.device)


    elif args.quant_mode == 'real':
        if args.init_from and os.path.isfile(args.init_from):
            net = load_model(args.init_from, net)
#     return

    # setup optimizer
    assert args.optimizer in ["SGD", "Adam"]
    if args.weight_decay_for_scale:
        params = net.parameters()
    else:
        if args.different_optimizer_mode:
            sparams, params = split_params(net, weight_decay=args.weight_decay, skip_keys=['scale'])
            # print(sparams)
            # print("====")
            # print(params)
        else:
            params = add_weight_decay(net, weight_decay=args.weight_decay, skip_keys=['scale'])        

    if args.optimizer == "SGD":
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(params, lr = args.lr, weight_decay=args.weight_decay)
    else:
        warnings.warn("No optimizer file is provided !!!")
    if args.different_optimizer_mode:
        if args.step_size_optimizer == "SGD":
            soptimizer = optim.SGD(sparams, lr=args.step_size_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.step_size_optimizer == "Adam":
            soptimizer = optim.Adam(sparams, lr = args.step_size_lr, weight_decay=args.weight_decay)
        else:
            warnings.warn("No optimizer file is provided !!!")
    else:
        soptimizer = None

    # scheduler
    assert args.lr_scheduler in ['StepLR', 'CosineAnnealing'], print(args.lr_scheduler)
    if args.lr_scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.milestones, args.gamma)
        if args.different_optimizer_mode:
            sscheduler = optim.lr_scheduler.StepLR(soptimizer, args.milestones, args.gamma)
    elif args.lr_scheduler == "CosineAnnealing":
        if args.enable_warmup:
            print("warmup start@lr=", args.warmup_lr)
            scheduler = lr_scheduler.ConstantWarmupScheduler(optimizer=optimizer, min_lr=args.warmup_lr,  total_epoch=args.warmup_epochs, after_lr=args.lr, 
                                        after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepochs - args.warmup_epochs))
            if args.different_optimizer_mode:
                sscheduler = lr_scheduler.ConstantWarmupScheduler(optimizer=soptimizer, min_lr=args.warmup_lr,  total_epoch=args.warmup_epochs, after_lr=args.lr, 
                                        after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepochs - args.warmup_epochs))
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepochs )        
            if args.different_optimizer_mode:
                sscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(soptimizer, T_max=args.nepochs )        
    else:
        warnings.warn("No scheduler file is provided !!!")
    
    if args.action == 'resume':
        if args.init_from and os.path.isfile(args.init_from):
            start_epoch, net, optimizer, scheduler, best_acc = load_ckp(args.init_from, net, optimizer, scheduler)
        else:
            warnings.warn("No checkpoint file is provided !!!")

    criterion = nn.CrossEntropyLoss()
        
   
    for epoch in range(start_epoch, args.nepochs):
        train_accuracy, train_loss = run_one_epoch(net, dataloader, optimizer, soptimizer, criterion, epoch, "train", args)
        # val_accuracy,   val_loss   = run_one_epoch(net, dataloader, optimizer, soptimizer, criterion, epoch, "test",   args)
        val_accuracy,   val_loss   = run_one_epoch(net, dataloader, optimizer, soptimizer, criterion, epoch, "val",   args)
        print("[Train] Epoch=", epoch, 'train_Loss: %.3f, train_Acc: %.3f' % (train_loss.item(), train_accuracy), '  |  ', 'val_Loss: %.3f, val_Acc: %.3f' % (val_loss.item(), val_accuracy))
        # update learning rate by scheduler
        scheduler.step()
        if args.different_optimizer_mode:
            sscheduler.step()
        
        if args.write_log:
            
            writer.add_scalar("loss/1.train",     train_loss,     epoch)
            writer.add_scalar("loss/2.val",       val_loss,       epoch)
            writer.add_scalar("accuracy/1.train", train_accuracy, epoch)
            writer.add_scalar("accuracy/2.val",   val_accuracy,   epoch)

            save_ckp(net, scheduler, optimizer, best_acc, epoch, val_accuracy, filename=os.path.join(args.save_path, 'ckpt.pth'))
            for idx, (name, m) in enumerate(net.named_modules()):
                # print(name, type(m))
                # if name in ['features.init_block', 'output', 'features.init_block.conv']:
                #     continue
                if type(m) in (Q.QLinear, Q.QConv2d):
                    if m.mode != "real":
                        if args.quant_mode == 'LSQ' or args.quant_mode == 'floorLSQ'  or args.quant_mode == 'ceilLSQ' or args.quant_mode == 'W_floorLSQ_A_LSQ':
                            if m.mode != "LSQ_non_uniform_first_layer":
                                w_scale = {'s[0]': m.w_scale}
                                x_scale = {'s[0]': m.x_scale}
                            else:
                                w_pos_scale = {}
                                x_pos_scale = {}
                                w_neg_scale = {}
                                x_neg_scale = {}
                                Ns_x = {}
                                # print(m.mode)
                                for idx in range(torch.numel(m.x_pos_scale)):
                                    x_pos_scale["s[{:d}]".format(idx)] = m.x_pos_scale[idx]
                                for idx in range(torch.numel(m.x_neg_scale)):
                                    x_neg_scale["s[{:d}]".format(idx)] = m.x_neg_scale[idx]
                                for idx in range(torch.numel(m.w_pos_scale)):
                                    w_pos_scale["s[{:d}]".format(idx)] = m.w_pos_scale[idx]
                                for idx in range(torch.numel(m.w_neg_scale)):
                                    w_neg_scale["s[{:d}]".format(idx)] = m.w_neg_scale[idx]

                        elif args.quant_mode == 'SoftPlus_LSQ':
                            w_scale = {'s[0]': np.log(1 + np.exp(m.w_scale.cpu()))}
                            x_scale = {'s[0]': np.log(1 + np.exp(m.x_scale.cpu()))}
                        elif args.quant_mode == 'LSQ_non_uniform_only_activation' or args.quant_mode ==  "LSQ_non_uniform_only_activation_fast" \
                            or args.quant_mode == 'LSQ_non_uniform_only_activation_auto_grad' \
                            or args.quant_mode == 'LSQ_non_uniform_non_local_only_activation' \
                            or args.quant_mode == 'LSQ_non_uniform_non_local_only_activation_II' \
                            or args.quant_mode == 'LSQ_non_uniform_non_local_only_activation_III' \
                            or args.quant_mode == 'LSQ_non_uniform_non_local_only_activation_IV' \
                            or args.quant_mode == 'LSQ_non_uniform_non_local_only_activation_V' \
                            or args.quant_mode == 'check_LSQ_non_uniform_non_local_only_activation':
                            w_scale = {'s[0]': m.w_scale}
                            x_scale = {}
                            Ns_x = {}
                            # print(m.mode)
                            for idx in range(torch.numel(m.x_scale)):
                                x_scale["s[{:d}]".format(idx)] = m.x_scale[idx]
                                if m.mode != "LSQ_first_layer" and m.mode != "shiftLSQ_first_layer":
                                    Ns_x["s[{:d}]".format(idx)] = m.Ns_x[idx]
                            writer.add_scalars('Ns_x/{:s}'.format(name), Ns_x, epoch)
                            # reset Ns_x                        
                            for idx in range(torch.numel(m.x_scale)):
                                if m.mode != "LSQ_first_layer" and m.mode != "shiftLSQ_first_layer":
                                    m.Ns_x.data[idx] = 0
                        if m.mode != "LSQ_non_uniform_first_layer":
                            writer.add_scalars('x_step_size/{:s}'.format(name), x_scale, epoch)
                            writer.add_scalars('w_step_size/{:s}'.format(name), w_scale, epoch)
                        else:
                            writer.add_scalars('x_pos_step_size/{:s}'.format(name), x_pos_scale, epoch)
                            writer.add_scalars('x_neg_step_size/{:s}'.format(name), x_neg_scale, epoch)
                            writer.add_scalars('w_pos_step_size/{:s}'.format(name), w_pos_scale, epoch)
                            writer.add_scalars('w_neg_step_size/{:s}'.format(name), w_neg_scale, epoch)
        
    # if args.dataset_name == 'cifar':
    #     test_accuracy, test_loss = run_one_epoch(net, dataloader, None, criterion, 0, "test", args)
    #     print('test_Loss: %.3f, test_Acc: %.3f' % (val_loss.item(), test_accuracy))



# if __name__ == '__main__':
if True :    
    time_sta = time.time()
    global args, best_acc
    # print(args.quant_mode)
    parser = define_args()
    args = parser.parse_args()
    # args = parser.parse_args(args=[])    
    best_acc = 0
    # dataloaders = imagenet_dataloaders(128, None, None, None, 8)
    # setup device
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.info_outputs:
        print("Device:", args.device)
    
    args.train_id = 2
    # args.enable_warmup = True
    args.enable_warmup = False # for 8bits

    # StepLR or CosineAnnealing
    args.lr_scheduler = 'CosineAnnealing'
    # cifar or imagenet
    args.dataset_name = 'cifar'
    # args.dataset_name = 'imagenet'
    
    args.lr = 0.01
    # args.lr = 0.001 # for 8bits
    # args.lr = 0.001 # for DEBUG MODE
    args.first_last_num_bits = 2
    args.quant_mode_at_first_layer = "real"
    # args.quant_mode_at_first_layer = "LSQ_first_layer"
    # args.quant_mode_at_first_layer = "shiftLSQ_first_layer"
    # args.quant_mode_at_first_layer = "LSQ_non_uniform_first_layer"
    # args.quant_mode_at_last_layer = "LSQ"
    args.quant_mode_at_last_layer = "real"
    args.weight_decay = 5e-4 #cifar wd
    # args.weight_decay = 1e-4 #cifar wd
    # args.weight_decay = 0.25e-4
    args.weight_decay_for_scale = False
    # args.x_grad_scale_mode = "10fac_LSQ_grad_scale"    
    args.x_grad_scale_mode = "LSQ_grad_scale"    
    args.w_grad_scale_mode = "LSQ_grad_scale"    
    # args.w_grad_scale_mode = "wo_grad_scale"    

    args.different_optimizer_mode = True
    args.step_size_lr = 0.001
    args.step_size_optimizer = "Adam"
    
    # Path to the pre-train model
    args.action = 'load'
    # args.action = 'progressive_quantization'
    # args.progressive_bits = 3
    if args.dataset_name == 'imagenet' and args.action == 'load':
        imagenet_config(args)
        if args.model == 'preresnet18':
            args.init_from = 'model_zoo/preresnet18-0972-5651bc2d.pth'
        elif args.model == 'mobilenetv2':
            args.init_from = 'model_zoo/mobilenetv2_w1-0887-13a021bc.pth'
    elif args.dataset_name == 'imagenet' and args.action == 'progressive_quantization':
        args.init_from = 'xxxx'        
    elif args.dataset_name == 'cifar' and args.action == 'load':
        cifar_config(args)
        args.init_from = 'model_zoo/preresnet20_cifar100-3022-3dbfa6a2.pth'
    elif args.dataset_name == 'cifar' and args.action == 'progressive_quantization':
        load_name = 'preresnet20_cifar100_{}bit_SGD_CosineAnnealing_lr0.001_wo_wd_for_scale_w_LSQ_grad_scale_x_LSQ_grad_scale_from_4bits_progressive_quantion-2'.format(args.progressive_bits)
        load_quant_mode = 'LSQ_non_uniform_only_activation'
        args.init_from = 'work_dir/{0}/{1}/{2}/ckpt.pth'.format(args.dataset_name,load_quant_mode,load_name)        
    
    # args.action = 'resume'
    # args.init_from = 'work_dir/cifar/SoftPlus_LSQ/preresnet18_3bit-2/ckpt.pth'
#     args.init_from = 'work_dir/imagenet/LSQ_non_uniform_only_activation/preresnet18_3bit-1/ckpt.pth'
    args.evaluate = False
    args.pre_trained = True
    args.write_log = True
    args.calculate_A = False

    if args.evaluate:
        run_test(args)
    else:
        run_train(args)
    time_end = time.time()
    tim = time_end-time_sta
    print("time=", tim)
    print('finished')

        # run test for the pre-trained model
