import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import tqdm
import torchvision
import src.scheduler.lr_scheduler as lr_scheduler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import src.module_quantization as Q
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

#========================================================================
# create dataloaders
#========================================================================
def imagenet_dataloaders(batch_size, num_workers, DDP_mode = True, model = None):
    
    traindir = './data/imagenet/train'
    valdir = './data/imagenet/val'
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]))
    val_dataset = ImageFolder(valdir, transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,]))

    dataloaders = {}
    if DDP_mode == True:
        train_sampler = DistributedSampler(train_dataset, shuffle=True) 
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    dataloaders["train"] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    dataloaders["val"] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=val_sampler)


    return dataloaders

def cifar100_dataloaders(batch_size, num_workers, num_train=50000, num_val=10000, num_test=10000, DDP_mode = True, model = None):

    # setup tarnsform object
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                            std=(0.2023, 0.1994, 0.2010)),])
    transform_test  = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                            std=(0.2023, 0.1994, 0.2010)),])

    # download CIFAR100 (if necessary) and get dataset object
    org_trainset = torchvision.datasets.CIFAR100(root='./data', train=True,  download=True, transform=transform_train)
    org_testset  = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # extract subset of the dataset
    trainset = torch.utils.data.Subset(org_trainset, list(range(0, num_train)))
    valset   = torch.utils.data.Subset(org_testset, list(range(0,  num_val)))
    testset  = torch.utils.data.Subset(org_testset,  list(range(0, num_test)))    

    # setup dataloader
    dataloaders = {}
    if DDP_mode == True:
        train_sampler = DistributedSampler(trainset, shuffle=True) 
        val_sampler = DistributedSampler(valset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    dataloaders["train"] = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None),  num_workers=num_workers, sampler=train_sampler)
    dataloaders["val"]   = torch.utils.data.DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=val_sampler)
    dataloaders["test"]  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=val_sampler)

    return dataloaders

def cifar10_dataloaders(batch_size, num_workers, num_train=50000, num_val=10000, num_test=10000, DDP_mode = True, model = None):
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

    # setup tarnsform object
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,])
    transform_test  = transforms.Compose([transforms.ToTensor(),
                                        normalize,])

    # download CIFAR10 (if necessary) and get dataset object
    org_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
    org_testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # extract subset of the dataset
    trainset = torch.utils.data.Subset(org_trainset, list(range(0, num_train)))
    valset   = torch.utils.data.Subset(org_testset, list(range(0,  num_val)))
    testset  = torch.utils.data.Subset(org_testset,  list(range(0, num_test)))    

    # setup dataloader
    dataloaders = {}
    if DDP_mode == True:
        train_sampler = DistributedSampler(trainset, shuffle=True) 
        val_sampler = DistributedSampler(valset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    dataloaders["train"] = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None),  num_workers=num_workers, sampler=train_sampler)
    dataloaders["val"]   = torch.utils.data.DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=val_sampler)
    dataloaders["test"]  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=val_sampler)

    return dataloaders

def setup_dataloader(name, batch_size, nworkers, DDP_mode, model):
    if name == "cifar100":
        dataloader = cifar100_dataloaders(batch_size, nworkers, DDP_mode=DDP_mode, model = model)
    elif name == "cifar10":
        dataloader = cifar10_dataloaders(batch_size, nworkers, DDP_mode=DDP_mode,  model = model)
    elif name == "imagenet":
        dataloader = imagenet_dataloaders(batch_size, nworkers, DDP_mode=DDP_mode,  model = model)
    else:
        raise NotImplementedError
    return dataloader
    

def write_experiment_params(args, file_path):
    with open(file_path, 'w') as f:
        for name in args.keys():
            f.write(f"{name}: {args[name]}\n")
    
def load_ckp(checkpoint_fpath, model, optimizer = None, scheduler = None):
    checkpoint = torch.load(checkpoint_fpath)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None: optimizer.load_state_dict(checkpoint['optimizer'])
    if type(scheduler) == lr_scheduler.ConstantWarmupScheduler:
        scheduler.load_state_dict(checkpoint['scheduler'], epoch) 
    elif scheduler is not None: 
        scheduler.load_state_dict(checkpoint['scheduler'])
    best_acc = checkpoint['best_acc']
    acc = checkpoint['acc']



    return epoch, model, optimizer, scheduler, best_acc, acc


def load_from_FP32_model(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath)
    loaded_params = {}
    if 'state_dict' in checkpoint.keys():
        st_dct = checkpoint['state_dict']
    else:
        st_dct = checkpoint
    for k,v in st_dct.items():
        if 'module' in k: 
            k = k[7:]
        loaded_params[k] = v
    model_state_dict = model.state_dict()
    model_state_dict.update(loaded_params)
    model.load_state_dict(model_state_dict)

    return model



def save_ckp(net, lr_scheduler, optimizer, best_acc, epoch, acc, ddp, filename='ckpt_best.pth'):
    if ddp:
        saved_net = net.module
    else:
        saved_net = net
    state = {
            'epoch': epoch + 1,
            'state_dict': saved_net.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler' : lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'best_acc' : best_acc,
            'acc' : acc,
        }
    torch.save(state, filename)

def stepsize_init(net, dataloader, device, num_batches=1):

    net.train()
    torch.set_grad_enabled(False)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            if i + 1 == num_batches:
                break

        for idx, (name, m) in enumerate(net.named_modules()):
            if type(m) in (Q.QLinear, Q.QConv2d):
                m.init_state.data = torch.tensor(True).to(device)
    return

def replace_all(model, replacement_dict={}):
    """
    Replace all layers in the original model with new layers corresponding to `replacement_dict`.
    E.g input example:
    replacement_dict={ nn.Conv2d : partial(NIPS2019_QConv2d, bit=args.bit) }
    """ 

    def __replace_module(model):
        for module_name in model._modules:
            m = model._modules[module_name]             
            if type(m) in replacement_dict.keys():
                if isinstance(m, nn.Conv2d):
                    new_module = replacement_dict[type(m)]
                    model._modules[module_name] = new_module(in_channels=m.in_channels, 
                            out_channels=m.out_channels, kernel_size=m.kernel_size, 
                            stride=m.stride, padding=m.padding, dilation=m.dilation, 
                            groups=m.groups, bias=(m.bias!=None))
                
                elif isinstance(m, nn.Linear):
                    new_module = replacement_dict[type(m)]
                    model._modules[module_name] = new_module(in_features=m.in_features, 
                            out_features=m.out_features,
                            bias=(m.bias!=None))

            elif len(model._modules[module_name]._modules) > 0:
                __replace_module(model._modules[module_name])

    __replace_module(model)

    return model


def replace_single_module(new_cls, current_module):
    m = current_module
    if isinstance(m, Q.QConv2d):
        return new_cls(in_channels=m.in_channels, 
                out_channels=m.out_channels, kernel_size=m.kernel_size, 
                stride=m.stride, padding=m.padding, dilation=m.dilation, 
                groups=m.groups, bias=(m.bias!=None))
    
    elif isinstance(m, Q.QLinear):
        return new_cls(in_features=m.in_features, out_features=m.out_features, bias=(m.bias != None))        

    return None

def replace_module(model, replacement_dict={}, exception_dict={}, arch="pytorchcv_preresnet18"):
    """
    Replace all layers in the original model with new layers corresponding to `replacement_dict`.
    E.g input example:
    replacement_dict={ nn.Conv2d : partial(Q.QConv2d, \
                        num_bits=args.num_bits, w_grad_scale_mode = args.w_grad_scale_mode, \
                        x_grad_scale_mode = args.x_grad_scale_mode, \
                        weight_norm = args.weight_norm, w_quantizer = args.w_quantizer, x_quantizer = args.x_quantizer, \
                        w_initializer = args.w_initializer, x_initializer = args.x_initializer), }
    exception_dict={
            '__first__': partial(Q.QConv2d, num_bits=args.first_bits, w_quantizer =args.w_first_last_quantizer, x_quantizer = args.x_first_last_quantizer, \
                                  w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, first_layer = True),
            '__last__': partial(Q.QLinear,  num_bits=args.last_bits, w_quantizer =args.w_first_last_quantizer,x_quantizer = args.x_first_last_quantizer, \
                                  w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, first_layer = False),
    }
    """ 
    assert arch in ["pytorchcv_preresnet20_cifar10","pytorchcv_preresnet20_cifar100", "pytorchcv_preresnet18", "pytorchcv_preresnet34", 'pytorchcv_mobilenetv2' ],\
            ("Not support this type of architecture !")

    model = replace_all(model, replacement_dict=replacement_dict)
    if arch in ["pytorchcv_preresnet18", "pytorchcv_preresnet34"]:
        model.features.init_block.conv = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.features.init_block.conv)
        model.output = replace_single_module(new_cls=exception_dict['__last__'], current_module=model.output)
    elif arch in ["pytorchcv_preresnet20_cifar100", "pytorchcv_preresnet20_cifar10"]:
        model.features.init_block = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.features.init_block)
        model.output = replace_single_module(new_cls=exception_dict['__last__'], current_module=model.output)
    elif arch == "pytorchcv_mobilenetv2":
        model.features.init_block.conv = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.features.init_block.conv)
        print("fin init")
        model.output = replace_single_module(new_cls=exception_dict['__last_for_mobilenet__'], current_module=model.output)
        
    return model

def split_params(model, weight_decay, lr, x_lr, w_lr, x_wd, w_wd):
    decay, no_decay_x, no_decay_w = [], [], []

    for name, param in model.named_parameters():
        # print(name, type(param))
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in ['x_scale', 'x_threshold', 'x_theta', 'x_bitscale']:
            if skip_key in name:
                # print ("Skip weight decay & x_lr for: ", name)
                no_decay_x.append(param)
                added = True
                break
        for skip_key in ['w_scale','w_zero_scale', 'w_shift_scale', 'w_threshold', 'w_theta', 'w_bitscale']:
            if skip_key in name:
                # print ("Skip weight decay & w_lr for: ", name)
                no_decay_w.append(param)
                added = True
                break
        if not added:
            decay.append(param)
    return [{'group_name': "x_Qparms", 'params': no_decay_x, 'lr': x_lr,'after_lr': x_lr, 'weight_decay': x_wd},\
    {'group_name': "w_Qparms", 'params': no_decay_w, 'lr': w_lr, 'after_lr': w_lr,'weight_decay': w_wd}],\
    [ {'group_name': "others", 'params': decay, 'weight_decay': weight_decay, 'lr': lr, 'after_lr': lr}]


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def parallel_reduce(*argv):
    tensor = torch.FloatTensor(argv).cuda()
    torch.distributed.all_reduce(tensor)
    ret = tensor
    return ret

def parallel_reduce_for_dict(loss_dict):
    for loss_name in loss_dict.keys():
        loss = loss_dict[loss_name]
        torch.distributed.all_reduce(loss)
        loss_dict[loss_name] = loss
    return loss_dict


class Multiple_Loss:
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.loss_value_dict ={}

    def __call__(self, prediction, target, *args, **kwargs):
        total_loss = 0
        for loss_name in set(self.loss_dict):
            self.loss_value_dict[loss_name] = self.loss_dict[loss_name](prediction, target, *args, **kwargs)
            total_loss += self.loss_value_dict[loss_name] 
            
        return total_loss, self.loss_value_dict



class Multiple_optimizer_scheduler:
    def __init__(self, opt_or_sche_dict):
        self.opt_or_sche_dict = opt_or_sche_dict

    def step(self):
        for opt_or_sche_name in set(self.opt_or_sche_dict):
            self.opt_or_sche_dict[opt_or_sche_name].step()
            
    def zero_grad(self):
        for opt_or_sche_name in set(self.opt_or_sche_dict):
            self.opt_or_sche_dict[opt_or_sche_name].zero_grad()

def set_Qparm_value(model, key, val):
    for idx, (name, m) in enumerate(model.named_modules()):
        if name in ['module.features.init_block', 'module.output', 'module.features.init_block.conv', 'features.init_block', 'output', 'features.init_block.conv']:
            continue
        if type(m) in (Q.QLinear, Q.QConv2d):
            for name_qparm in m.x_Qparms.keys():
                if name_qparm == key:
                    m.x_Qparms[name_qparm].data = torch.tensor([val])
                    m.w_Qparms[name_qparm].data = torch.tensor([val])

#ref: https://github.com/aparo/pyes/issues/183            
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)
        
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

