#========================================================================
# import modules 
#========================================================================

import os
# torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# torchvision
import src.module_quantization as Q
from src.models.model import create_model, prepare_pretrained_model

from src.utils import *
from src.scheduler_optimizer_class import *
from src.make_ex_name import *
from functools import partial

import time
import warnings
import src.scheduler.Qparm_scheduler as Qparm_scheduler

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from src.quantizer.uniform import *
from src.quantizer.nonuniform import *
from src.initializer import *

def run_one_epoch(net, dataloader, optimizers, criterion, epoch, mode, best_acc, args):
    if mode == "train":
        net.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    total_loss        = 0.0
    total_loss_dict = {}
    total_num_sample  = 0
    total_num_correct = 0
    total_num_correct5 = 0

    
    # setup progress bar
    with tqdm(total=len(dataloader[mode]), disable = args.invisible_pgb) as pbar:
        pbar.set_description(f"Epoch[{epoch}/{args.nepochs}]({mode})")

        # loop for each epoch
        for i, (inputs, labels) in enumerate(dataloader[mode]):
            loss_dict = {}

            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            # run train/val/test
            if mode == "train":
                optimizers.zero_grad()
                outputs = net(inputs)
                loss, loss_dict = criterion(outputs, labels)
                loss.backward()
                optimizers.step()

            else:
                outputs = net(inputs).cuda()
                loss, loss_dict = criterion(outputs, labels)

            # statistics
            _, predicted       = torch.max(outputs.detach(), 1)
            _, predicted5       = outputs.topk(5, 1, True, True)
            predicted5 = predicted5.t()
            num_sample         = labels.size(0)
            num_correct        = int((predicted == labels).sum())  
            targets = labels.expand_as(predicted5)
            num_correct5       = predicted5.eq(targets).reshape(-1).float().sum(0)
            total_loss        += loss.detach() * num_sample
            total_loss_dict = {k: total_loss_dict.get(k, 0) + loss_dict.get(k, 0).detach()* num_sample for k in set(total_loss_dict) | set(loss_dict)}
            total_num_sample  += num_sample
            total_num_correct += num_correct
            total_num_correct5 += num_correct5

            # update progress bar
            pbar.update(1)

    if args.ddp:
        total_loss, total_num_correct, total_num_correct5 = parallel_reduce(total_loss, total_num_correct, total_num_correct5)
        total_loss_dict = parallel_reduce_for_dict(total_loss_dict)
        total_num_sample = args.world_size * total_num_sample
    avg_loss = total_loss        / total_num_sample
    avg_loss_dict = {k: total_loss_dict.get(k, 0)/total_num_sample for k in set(total_loss_dict)}
    accuracy = total_num_correct / total_num_sample
    top5_accuracy = total_num_correct5 / total_num_sample
    if accuracy > best_acc and mode != 'train':
        best_acc = accuracy
    return accuracy, top5_accuracy, avg_loss, best_acc, avg_loss_dict

def run_test(args):
    print('==> Preparing testing for {0}..'.format(args.dataset_name))
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, DDP_mode = False, model = args.model)
    net = create_model(args)        
    assert net != None
    
    net = net.to(args.device)

    if args.init_from and os.path.isfile(args.init_from):
        print('==> Loading from checkpoint: ', args.init_from)
        net = load_from_FP32_model(args.init_from, net)

    cudnn.benchmark = True
        
    task_loss_fn = nn.CrossEntropyLoss()
    criterion_val = Multiple_Loss( {"task_loss": task_loss_fn})       
        
    val_accuracy, val_top5_accuracy,  val_loss, best_acc, val_loss_dict   = run_one_epoch(net, dataloader, None, criterion_val, 0, "val", 0, args)
    print('[FP32 model] val_Loss: %.5f, val_top1_Acc: %.5f, val_top5_Acc: %.5f' % (val_loss_dict["task_loss"].item(), val_accuracy, val_top5_accuracy))
    
def run_load_model(args):

    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, DDP_mode = False, model = args.model)
    net = create_model(args)

    assert net != None

    print("==> Replacing model parameters..")
    replacement_dict = {
                        nn.Conv2d : partial(Q.QConv2d, \
                        num_bits=args.num_bits, w_grad_scale_mode = args.w_grad_scale_mode, \
                        x_grad_scale_mode = args.x_grad_scale_mode, \
                        weight_norm = args.weight_norm, w_quantizer = args.w_quantizer, x_quantizer = args.x_quantizer, \
                        w_initializer = args.w_initializer, x_initializer = args.x_initializer), 
                        nn.Linear: partial(Q.QLinear,  \
                        num_bits=args.num_bits, w_grad_scale_mode = args.w_grad_scale_mode, \
                        x_grad_scale_mode = args.x_grad_scale_mode, \
                        weight_norm = args.weight_norm, w_quantizer = args.w_quantizer, x_quantizer = args.x_quantizer, \
                        w_initializer = args.w_initializer, x_initializer = args.x_initializer)}
    exception_dict = {
            '__first__': partial(Q.QConv2d, num_bits=args.first_bits, w_quantizer =args.w_first_last_quantizer, x_quantizer = args.x_first_last_quantizer, \
                                  w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, \
                                  w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                  x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                 first_layer = True),
            '__last__': partial(Q.QLinear,  num_bits=args.last_bits, w_quantizer =args.w_first_last_quantizer,x_quantizer = args.x_first_last_quantizer, \
                                  w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, \
                                  w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                  x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                    first_layer = False),
            '__last_for_mobilenet__': partial(Q.QConv2d, num_bits=args.last_bits, w_quantizer =args.w_first_last_quantizer, x_quantizer = args.x_first_last_quantizer, \
                                  w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, 
                                  w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                  x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                  first_layer = False),
        }        
    net = replace_module(net, replacement_dict=replacement_dict, exception_dict=exception_dict, arch=args.model)
    net = net.to(args.device)
    
    print('==> Performing action: ', args.action)
    if args.action == 'load':
        if args.init_from and os.path.isfile(args.init_from):
            print('==> Initializing from checkpoint: ', args.init_from)
            net = load_from_FP32_model(args.init_from, net)
            stepsize_init(net, dataloader["train"], args.device, args.init_num)
        else:
            warnings.warn("No checkpoint file is provided !!!")

    return net



def run_train(rank, args, net):
    start_epoch = 0
    world_size = args.world_size
    # setup the process groups
    if args.ddp:
        ddp_setup(rank, world_size)
        data_mode = True
    else:
        data_mode = False
    best_acc = 0.0

    print('==> Preparing training for {0}..'.format(args.dataset_name))
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, DDP_mode = data_mode, model = args.model)

    net = net.to(args.device)

    args.ex_name = make_ex_name(args)


    save_path = os.path.join(args.save_path, args.dataset_name, args.ex_name)
    if args.write_log:
        os.makedirs(save_path, exist_ok=True)
        write_experiment_params(args, os.path.join(save_path, args.model+'.txt'))
        writer = SummaryWriter(os.path.join(save_path, 'tf_log'))

        if rank == 0:
            if args.first_run:
                print (net)
                print ("Number of learnable parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6, "M")   
    time.sleep(5)
    if torch.cuda.device_count() >= 1:
        if args.ddp:
            if rank == 0:                  
                print("Let's use", torch.cuda.device_count(), "GPUs!")
            # Convert BatchNorm to SyncBatchNorm. 
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = DDP(net, device_ids=[rank])
            cudnn.benchmark = True


    # split parameters for different optimizers
    if args.different_optimizer_mode:
        sparams, params = split_params(\
            net, weight_decay=args.weight_decay, lr = args.lr, x_lr= args.x_step_size_lr, \
                w_lr= args.w_step_size_lr, x_wd = args.x_step_size_wd, w_wd = args.w_step_size_wd)
        if rank == 0:
            print(sparams)
            print("========================================================================================================================")
    else:
        params = net.parameters()

    # setup optimizer & scheduler        
    optimizer, scheduler = scheduler_optimizer_class(args, params, args.optimizer)
    optimizer_dict = {"optimizer": optimizer}
    scheduler_dict = {"scheduler": scheduler}
    if args.different_optimizer_mode:
        soptimizer, sscheduler = scheduler_optimizer_class(args, sparams, args.step_size_optimizer)
        optimizer_dict["step_size_optimizer"] = soptimizer
        scheduler_dict["step_size_scheduler"] = sscheduler
    
    if args.action == 'resume':
        if args.init_from and os.path.isfile(args.init_from):
            start_epoch, net, optimizer, scheduler, _, acc= load_ckp(args.init_from, net, optimizer, scheduler)
            print("acc=", acc)
            best_acc = acc.to(rank)
            print("best_acc=", best_acc)
        else:
            warnings.warn("No checkpoint file is provided !!!")

    task_loss_fn = nn.CrossEntropyLoss()
    loss_dict = {"task_loss": task_loss_fn}


    criterion = Multiple_Loss(loss_dict)
    all_optimizers = Multiple_optimizer_scheduler(optimizer_dict)
    all_schedulers = Multiple_optimizer_scheduler(scheduler_dict)
   
    val_accuracy, val_top5_accuracy,  total_val_loss, best_acc, val_loss_dict   = run_one_epoch(net, dataloader, all_optimizers, criterion, 0, "val", best_acc, args)
    if rank == 0:
        print('before learning val_Loss: %.4f, val_Acc: %.4f' % (val_loss_dict["task_loss"].item(), val_accuracy))
    for epoch in range(start_epoch, args.nepochs):
        if args.ddp:
            dataloader["train"].sampler.set_epoch(epoch)
        train_accuracy, train_top5_accuracy, total_train_loss, _, train_loss_dict = run_one_epoch(net, dataloader, all_optimizers, criterion, epoch, "train", best_acc, args)
        val_accuracy, val_top5_accuracy,  total_val_loss, best_acc, val_loss_dict   = run_one_epoch(net, dataloader, all_optimizers, criterion, epoch, "val", best_acc, args)
        if rank == 0:
            print("[Train] Epoch=", epoch, 'train_total_loss: %.3f, train_task_loss: %.3f, train_top1_Acc: %.3f, train_top5_Acc: %.3f' \
                  % (total_train_loss.item(),train_loss_dict["task_loss"].item(), train_accuracy, train_top5_accuracy))
            print("[Val] Epoch=", epoch, 'val_Loss: %.3f, val_top1_Acc: %.3f, val_top5_Acc: %.3f' % (val_loss_dict["task_loss"].item(), val_accuracy, val_top5_accuracy))
        # update coefficients by scheduler
        all_schedulers.step()
        
        if args.write_log:

            writer.add_scalar("loss/1.train_total",     total_train_loss,     epoch)
            writer.add_scalar("loss/2.train_cross_entropy",     train_loss_dict["task_loss"],     epoch)
            writer.add_scalar("loss/3.val_cross_entropy",       val_loss_dict["task_loss"],       epoch)
            writer.add_scalar("top1_accuracy/1.train", train_accuracy, epoch)
            writer.add_scalar("top1_accuracy/2.val",   val_accuracy,   epoch)
            writer.add_scalar("top5_accuracy/1.train", train_top5_accuracy, epoch)
            writer.add_scalar("top5_accuracy/2.val",   val_top5_accuracy,   epoch)

            if args.save_mode == "all_checkpoints":
                save_ckp(net, scheduler, optimizer, best_acc, epoch, val_accuracy, args.ddp, filename=os.path.join(save_path, 'ckpt_{}epoch.pth'.format(epoch)))
            else:
                save_ckp(net, scheduler, optimizer, best_acc, epoch, val_accuracy, args.ddp, filename=os.path.join(save_path, 'ckpt.pth'))
            if best_acc == val_accuracy:
                print('Saving best acc model ..')
                save_ckp(net, None, None, best_acc, epoch, best_acc, args.ddp, filename=os.path.join(save_path, 'best.pth'))


def main(args):    
    # if args.action == 'load':
    #     prepare_pretrained_model(args)
    
    print ("Arguments:\n", args)
    
    if args.evaluation_mode:
        args.ddp = False
        run_test(args)
    else:
        net = run_load_model(args)
        if args.ddp == True:
            print("DDP mode")
            mp.spawn( \
                run_train, \
                nprocs= args.world_size, \
                args= (args,net) \
            )
        else:
            torch.cuda.synchronize()
            start = time.time()
            print("no DDP mode")
            gpu = 0
            run_train(gpu, args, net)
            torch.cuda.synchronize()
            end = time.time()
            print("Elapsed time: ", end - start)

    print('finished')

