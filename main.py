#========================================================================
# import modules 
#========================================================================
# torch
import torch
import time
import yaml
import argparse

from src.utils import *
from src.scheduler_optimizer_class import *
from src.make_ex_name import *

from src.quantizer.uniform import *
from src.quantizer.nonuniform import *
from src.initializer import *
from src.run_DDP import main

if __name__ == '__main__':

    with open('./config/cifar10/nuLSQ_base.yaml') as file:
        config = yaml.safe_load(file.read())
    config = dotdict(config)   
    config.world_size = torch.cuda.device_count()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("number of gpu", torch.cuda.device_count())
    print("Device:", config.device)

    # firstlast quantizer
    config.w_first_last_quantizer = MinMax_quantizer
    config.x_first_last_quantizer = MinMax_quantizer


    parser = argparse.ArgumentParser(description='Cifar10 Training for LCQ')
    parser.add_argument('--lr', default=0.1, type=float, metavar='N', help='learning rate')
    parser.add_argument('--coeff_qparm_lr', default=0.1, type=float, metavar='N', help='qparm learning rate = coeff*lr')
    parser.add_argument('--weight_decay', default=0.1, type=float, metavar='N', help='weight decay')
    parser.add_argument('--qparm_wd', default=0.1, type=float, metavar='N', help='qparm_wd')
    parser.add_argument('--num_bits', type=int, help='quantization bit-width')
    parser.add_argument('--train_id', type=str, help='training id, is used for collect experiment results')
    parser.add_argument('--x_quantizer', type=str, help='x quantizer')
    parser.add_argument('--w_quantizer', type=str, help='w quantizer')
    parser.add_argument('--initializer', type=str, help='initializer')
    parser.add_argument('--first_run', action= 'store_true', help='control logs to reduce the redundancy')
    parser.add_argument('--init_from', type=str, help='init_from')

    QuantizerDict ={
        "MinMax_quantizer": MinMax_quantizer,
        "LSQ_quantizer": LSQ_quantizer,
        "LCQ_quantizer": LCQ_quantizer,
        "APoT_quantizer": APoT_quantizer,
        "Positive_nuLSQ_quantizer": Positive_nuLSQ_quantizer,
        "Symmetric_nuLSQ_quantizer": Symmetric_nuLSQ_quantizer,
        "FP": None
    }

    InitializerDict ={
        "NMSE_initializer": NMSE_initializer,
        "LSQ_initializer": LSQ_initializer,
        "Const_initializer": Const_initializer,
    }

    args = parser.parse_args()
    config.lr = args.lr
    config.x_step_size_lr = round(args.coeff_qparm_lr*config.lr, ndigits=5)
    config.w_step_size_lr = round(args.coeff_qparm_lr*config.lr, ndigits=5)
    config.weight_decay = args.weight_decay
    config.x_step_size_wd = args.qparm_wd
    config.w_step_size_wd = args.qparm_wd
    config.x_quantizer = QuantizerDict[args.x_quantizer]
    config.w_quantizer = QuantizerDict[args.w_quantizer]
    config.x_initializer = InitializerDict[args.initializer]
    config.w_initializer = InitializerDict[args.initializer]
    config.num_bits = args.num_bits
    config.train_id = args.train_id
    config.first_run = args.first_run
    config.init_from = args.init_from if args.init_from != None else None


    if config.different_optimizer_mode == False:
        print("reset Qparm hyper parameters to be same as other parameters' ones")
        config.step_size_optimizer = config.optimizer
        config.x_step_size_lr = config.lr
        config.w_step_size_lr = config.lr
        config.x_step_size_wd = config.weight_decay
        config.w_step_size_wd = config.weight_decay



    torch.cuda.synchronize()
    start = time.time()
    print("Start training at the following setting")
    print("lr:", config.lr, " x_step_size_lr:", config.x_step_size_lr, \
          " w_step_size_lr:", config.w_step_size_lr, " weight_decay:", \
            config.weight_decay, " x_step_size_wd:", config.x_step_size_wd,\
            " w_step_size_wd:", config.w_step_size_wd, "x_quantizer:", config.x_quantizer, \
            " w_quantizer:", config.w_quantizer, " x_initializer:", config.x_initializer, " w_initializer:", config.w_initializer, " num_bits:", config.num_bits, \
                " train_id:", config.train_id)
    main(config)
    print("End of training")


    torch.cuda.synchronize()
    end = time.time()
    print("total time", end - start)




