import argparse
import errno
import os
import sys



def define_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet/CIFAR Training')
    # Paths settings
    parser.add_argument('--action', type=str, help='action that you want to take to perform the experiments')
    parser.add_argument('--train_id', type=str, help='training id, is used for collect experiment results')
    parser.add_argument('--ex_name', type=str, help='training id, is used for collect experiment results')
    parser.add_argument('--dataset_name', type=str, help='the dataset name to be used in saving model names')
    parser.add_argument('--data_dir', type=str, help='The path saving train.json and val.json files')
    parser.add_argument('--dataset_dir', type=str, help='The path saving actual data')
    parser.add_argument('--save_path', type=str, default='work_dir/', help='directory to save output')
    parser.add_argument('--device', type=str, help='device for training')
    # Dataset settings
    # Quantization settings
    parser.add_argument('--quant_mode', type=str, help='quantization configuration mode (LSQ or nuLSQ), real for FP training')
    parser.add_argument('--num_bits', type=int, help='quantization bit-width')
    parser.add_argument('--first_last_num_bits', type=str, help='quantization bit-width at first and last layer')
    # parser.add_argument("--pretrained", type=str2bool, nargs='?', const=True, default=True, help="use pretrained vgg model")
    # General model settings 

    parser.add_argument('--model', default='resnet18', type=str, help='Main learning rate')
    parser.add_argument("--write_log", action="store_true", help="set the flag to write log and save model for trianing")
    parser.add_argument("--pre_trained", action="store_true", help="set the flag to true for loading the pre-trained model")
    parser.add_argument('--init_from', type=str, help='init weights from from checkpoint')
    parser.add_argument('--lr', default=0.01, type=float, help='Main learning rate')
    parser.add_argument('--step_size_lr', default=1e-5, type=float, help='Main learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--nepochs', type=int, default=10, help='total numbers of epochs')
    parser.add_argument('--gpu_ids', action='append', type=int, help='what gpu to use for training')
    parser.add_argument('--nworkers', type=int, default=8, help='num of threads')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout in network')
    parser.add_argument('--channels_in', type=int, default=3, help='num channels of input image')
    parser.add_argument('--test_mode', action='store_true', help='prevents loading latest saved model')
    parser.add_argument('--start_epoch', type=int, default=0, help='prevents loading latest saved model')
    parser.add_argument('--evaluate', action='store_true', help='only perform evaluation')
    parser.add_argument('--info_outputs', action='store_true', help='show network information')
    parser.add_argument('--resume', type=str, default='', help='resume latest saved run')
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='SGD', help='Adam or SGD')
    parser.add_argument('--step_size_optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_init', type=str, default='normal', help='normal, xavier, kaiming, orhtogonal weights initialisation')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='L2 weight decay/regularisation on')
    parser.add_argument('--weight_decay_for_scale',  action='store_true', help='L2 reg. for scale on/off')
    parser.add_argument('--lr_decay', action='store_true', help='decay learning rate with rule')
    parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=400, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr_decay_iters', type=int, default=30, help='multiply by a gamma every lr_decay_iters iterations')
    # Scheduler settings
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='StepLR or CosineAnnealing')
    parser.add_argument('--milestones', default=30, type=float, help='milestones for step-size learning rate scheduler')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma for step-size learning rate scheduler')
    parser.add_argument('--enable_warmup', dest='enable_warmup', action='store_true', help='Enable warm-up learning rate.')
    parser.add_argument('--different_optimizer_mode', dest='different_optimizer_mode', action='store_true', help='different_optimizer_mode bw step size and other params')
    parser.add_argument('--progressive_bits', type=int, default=8, help='bit number in loaded model on progressive quantization')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='number of epochs for warm-up')    
    parser.add_argument('--warmup_lr', default=0.001, type=float, help='Warmup learning rate')    
    # Print settings
    parser.add_argument('--print_freq', type=int, default=500, help='padding')
    parser.add_argument('--save_freq', type=int, default=500, help='padding')

    return parser

def cifar_config(args):
    args.model = 'preresnet20_cifar100'
    # args.batch_size = 100
    # args.nepochs = 30 # for debug
    # args.nepochs = 90
    args.nworkers = 8
    # args.optimizer = 'SGD'

def imagenet_config(args):
    args.model = 'preresnet18'
    # args.model = 'mobilenetv2'
    args.batch_size = 256
    # args.nepochs = 5
    args.nworkers = 8
    # args.optimizer = 'SGD'