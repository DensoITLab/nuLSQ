

from .pytorchcv.preresnet import *
from .pytorchcv.preresnet_cifar import *
from .pytorchcv.mobilenetv2 import *
import torchvision.models as models

def create_model(args):

    model = None
    if args.dataset_name == 'imagenet':
        if args.model == 'pytorchcv_preresnet18':
            model = preresnet18(pretrained=args.pre_trained)
        elif args.model == 'pytorchcv_preresnet34':
            model = preresnet34(pretrained=args.pre_trained)
        elif args.model == 'pytorchcv_mobilenetv2':
            model = mobilenetv2_w1(pretrained=args.pre_trained)
    elif args.dataset_name == 'cifar100':
        if args.model == 'pytorchcv_preresnet20_cifar100':
            model = preresnet20_cifar100(pretrained=args.pre_trained)
        elif args.model == 'pytorchcv_preresnet56_cifar100':
            model = preresnet56_cifar100(pretrained=args.pre_trained)
    elif args.dataset_name == 'cifar10':
        if args.model == 'pytorchcv_preresnet20_cifar10':
            model = preresnet20_cifar10(pretrained=args.pre_trained)
        elif args.model == 'pytorchcv_preresnet56_cifar10':
            model = preresnet56_cifar10(pretrained=args.pre_trained)
    if model == None:
        print('Model architecture `%s` for `%s` dataset is not supported' % (args.model, args.dataset_name))
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset' % (args.model, args.dataset_name)
    msg += '\n          Use pre-trained model = %s' % args.pre_trained
    print(msg)

    return model

def prepare_pretrained_model(args):
    if args.model == 'pytorchcv_preresnet18':
        args.init_from = 'model_zoo/pytorchcv/preresnet18-0972-5651bc2d.pth'
    elif args.model == 'pytorchcv_preresnet34':
        args.init_from = 'model_zoo/pytorchcv/preresnet34-0774-fd5bd1e8.pth'
    elif args.model == 'pytorchcv_mobilenetv2':
        args.init_from = 'model_zoo/pytorchcv/mobilenetv2_w1-0887-13a021bc.pth'
    elif args.model == 'pytorchcv_preresnet20_cifar100':
        args.init_from = 'model_zoo/pytorchcv/preresnet20_cifar100-3022-3dbfa6a2.pth'
    elif args.model == 'pytorchcv_preresnet20_cifar10':
        args.init_from = 'model_zoo/pytorchcv/preresnet20_cifar10-0651-76cec68d.pth'
    print("args.init_from", args.init_from)