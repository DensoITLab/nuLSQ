import os
import sys

import torch
import torch.nn as nn
import shutil
import tqdm
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import module_lsq as Q
from cmath import inf


#========================================================================
# create dataloaders
#========================================================================
def imagenet_dataloaders(batch_size, num_workers):
    
    # traindir = '../../../../../gs/hs0/GSIC/ILSVRC2012/ILSVRC2012_img_train'
    # valdir = '../../../../../gs/hs0/GSIC/ILSVRC2012/ILSVRC2012_img_val_classified'
    traindir = './data/imagenet/train'
    valdir = './data/imagenet/val'
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]))
    val_dataset = ImageFolder(valdir, transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,]))

    dataloaders = {}
    dataloaders["train"] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, sampler=None)
    dataloaders["val"] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, sampler=None)
    dataloaders["test"]  = torch.utils.data.DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return dataloaders

def cifar_dataloaders(batch_size, num_workers, num_train=50000, num_val=10000, num_test=10000):

    # setup tarnsform object
    # - ToTensor() maps [0, 255] to [0, 1]: y = x / 255.
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                        #                     std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),])
                                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                            std=(0.2023, 0.1994, 0.2010)),])
    transform_test  = transforms.Compose([transforms.ToTensor(),
                                        # transforms.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                        #                     std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),])
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
    dataloaders["train"] = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dataloaders["val"]   = torch.utils.data.DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloaders["test"]  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloaders

def write_experiment_params(args, file_path):
    """
    Writes all experiment parameters in `args` to a text file at `file_path`.
    """
    with open(file_path, 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None: optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None: scheduler.load_state_dict(checkpoint['scheduler'])
    best_acc = checkpoint['best_acc']
    acc = checkpoint['acc']
    
    return epoch, model, optimizer, scheduler, best_acc, acc

def load_model(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint)
    return model

def save_ckp(net, lr_scheduler, optimizer, best_acc, epoch, acc, filename='ckpt_best.pth'):
    state = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler' : lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'best_acc' : best_acc,
            'acc' : acc,
        }
    torch.save(state, filename)

def stepsize_init(net, dataloader, mode, device):

    net.eval()
    torch.set_grad_enabled(False)
    
    with torch.no_grad():
        data = iter(dataloader[mode])
        (inputs, labels) = next(data)

        inputs, labels = inputs.to(device), labels.to(device)

        net(inputs)
    return

def replace_all(model, replacement_dict={}):
    """
    Replace all layers in the original model with new layers corresponding to `replacement_dict`.
    E.g input example:
    replacement_dict={ nn.Conv2d : partial(NIPS2019_QConv2d, bit=args.bit) }
    """ 
    # for (name, param) in model.named_parameters():
        # print(name, type(param))

    def __replace_module(model):
        for module_name in model._modules:
            m = model._modules[module_name]             
            # print(module_name, type(m))
            if type(m) in replacement_dict.keys():
                # print(type(m))
                if isinstance(m, nn.Conv2d):
                    new_module = replacement_dict[type(m)]
                    # print(new_module)
                    # print(isinstance(m, nn.Conv2d))
                    model._modules[module_name] = new_module(in_channels=m.in_channels, 
                            out_channels=m.out_channels, kernel_size=m.kernel_size, 
                            stride=m.stride, padding=m.padding, dilation=m.dilation, 
                            groups=m.groups, bias=(m.bias!=None))
                    # print(isinstance(m, Q.QConv2d))
                
                elif isinstance(m, nn.Linear):
                    new_module = replacement_dict[type(m)]
                    model._modules[module_name] = new_module(in_features=m.in_features, 
                            out_features=m.out_features,
                            bias=(m.bias!=None))
                    # print(type(m))

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

def replace_module(model, replacement_dict={}, exception_dict={}, arch="preresnet18"):
    """
    Replace all layers in the original model with new layers corresponding to `replacement_dict`.
    E.g input example:
    replacement_dict={ nn.Conv2d : partial(NIPS2019_QConv2d, bit=args.bit) }
    exception_dict={
        'conv1': partial(NIPS2019_QConv2d, bit=8)
        'fc': partial(NIPS2019_QLinear, bit=8)
    }
    """ 
    assert arch in ["preresnet20_cifar100", "preresnet18", 'mobilenetv2' ],\
            ("Not support this type of architecture !")

    model = replace_all(model, replacement_dict=replacement_dict)
    if arch == "preresnet18":
        model.features.init_block.conv = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.features.init_block.conv)
        model.output = replace_single_module(new_cls=exception_dict['__last__'], current_module=model.output)
    elif arch == "preresnet20_cifar100":
        model.features.init_block = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.features.init_block)
        model.output = replace_single_module(new_cls=exception_dict['__last__'], current_module=model.output)
    elif arch == "mobilenetv2":
        model.features.init_block.conv = replace_single_module(new_cls=exception_dict['__first__'], current_module=model.features.init_block.conv)
        print("fin init")
        model.output = replace_single_module(new_cls=exception_dict['__last_for_mobilenet__'], current_module=model.output)
    # for idx, (name, m) in enumerate(model.named_modules()):
    #     # print(name, type(m))
    #     print(name, type(m))
    #     if type(m) in (Q.QLinear, Q.QConv2d):
    #         print(m)
    #         print(m.init_state)
        
    return model


def add_weight_decay(model, weight_decay, skip_keys):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        # print(name, type(param))
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in skip_keys:
            if skip_key in name:
                print ("Skip weight decay for: ", name)
                no_decay.append(param)
                added = True
                break
        if not added:
            decay.append(param)
    # print(no_decay)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]

def split_params(model, weight_decay, skip_keys):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        # print(name, type(param))
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in skip_keys:
            if skip_key in name:
                # print ("Skip weight decay for: ", name)
                no_decay.append(param)
                added = True
                break
        if not added:
            decay.append(param)
    # print(no_decay)
    return [{'params': no_decay, 'weight_decay': 0.}],[ {'params': decay, 'weight_decay': weight_decay}]


#----------------------------------------------------------
# MinRE (minimize reconstruction error)
#----------------------------------------------------------
def _find_step_size_by_minimizing_quantization_error(x, scale, Qn, Qp, cost_mode):

    Qn_on_device = torch.tensor([Qn], dtype=torch.float).to(x.device)
    Qp_on_device = torch.tensor([Qp], dtype=torch.float).to(x.device)

    # vectorize input data
    x_1d = torch.flatten(x)
    
    #solve by newton method 
    if cost_mode == "LogCosh":
        s_rec    = scale
        delta    = +inf
        alpha_lc = 100
        while delta > 1e-6:
            x_1d_tmp  = x_1d / s_rec
            x_1d_tmp  = torch.min(torch.max(x_1d_tmp, -Qn_on_device), Qp_on_device)
            B         = torch.round(x_1d_tmp)
            dlds      = torch.sum(torch.tanh(alpha_lc*(x_1d - B*s_rec))*(-alpha_lc*B))
            d2lds2    = torch.sum(torch.div(alpha_lc*B,torch.cosh(alpha_lc*(x_1d - B*s_rec)))**2)
            delta     = torch.abs(dlds/d2lds2)
            s_rec     = s_rec - dlds/d2lds2
            scale_new = s_rec
            # cost = torch.sum(torch.log(torch.cosh(x_1d - B*s_rec)))/x_1d.size()[0];
            # print("LogCosh: scale=", scale_new, "  cost =", cost);
    elif cost_mode == "L2":
        # s_rec = 0.01
        s_rec = scale
        delta = +inf
        while delta > 1e-8:
            x_1d_tmp  = x_1d / s_rec
            x_1d_tmp  = torch.min(torch.max(x_1d_tmp, -Qn_on_device), Qp_on_device)
            B         = torch.round(x_1d_tmp)
            s_new     = torch.sum(x_1d * B)/torch.sum(B*B)
            delta     = torch.abs(s_rec - s_new)
            s_rec     = s_new
        cost = torch.sum((x_1d - B*s_rec)**2)/x_1d.size()[0]
        scale_new = s_rec
        #print("1 init result:scale=", scale_new, " cost=", cost)

    return scale_new