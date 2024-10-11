import torch.optim as optim
import src.scheduler.lr_scheduler as lr_scheduler

def scheduler_optimizer_class(args, params, optimizer_name):
    #specify the optimizer
    if optimizer_name == "SGD":
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov =args.nesterov)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
       raise ValueError("Optimizer not supported.")
    
    #specify the scheduler for learning rate
    if args.lr_scheduler == "CosineAnnealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepochs )        
        # if args.warmup_mode == "const":
    elif args.lr_scheduler == "LinearWarmupCosineAnnealing":
        print("warmup start@lr=", args.warmup_lr)
        scheduler = lr_scheduler.LinearWarmupScheduler(optimizer=optimizer, min_lr=args.warmup_lr,  total_epoch=args.warmup_epochs, 
                                    after_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepochs - args.warmup_epochs))
    else:
        raise ValueError("Scheduler not supported.")  

    return optimizer, scheduler  
