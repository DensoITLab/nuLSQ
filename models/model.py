

from .preresnet import *
from .preresnet_cifar import *
from .mobilenetv2 import *

def create_model(args):

    model = None
    if args.dataset_name == 'imagenet':
        if args.model == 'preresnet18':
            model = preresnet18(pretrained=args.pre_trained)
        elif args.model == 'preresnet34':
            model = preresnet34(pretrained=args.pre_trained)
        elif args.model == 'preresnet50':
            model = preresnet50(pretrained=args.pre_trained)
        elif args.model == 'preresnet101':
            model = preresnet101(pretrained=args.pre_trained)
        elif args.model == 'preresnet152':
            model = preresnet152(pretrained=args.pre_trained)
        elif args.model == 'mobilenetv2':
            model = mobilenetv2_w1(pretrained=args.pre_trained)
    elif args.dataset_name == 'cifar':
        if args.model == 'preresnet20_cifar100':
            model = preresnet20_cifar100(pretrained=args.pre_trained)
        elif args.model == 'preresnet56_cifar100':
            model = preresnet56_cifar100(pretrained=args.pre_trained)
        elif args.model == 'preresnet110_cifar100':
            model = preresnet110_cifar100(pretrained=args.pre_trained)

    if model == None:
        print('Model architecture `%s` for `%s` dataset is not supported' % (args.model, args.dataset_name))
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset' % (args.model, args.dataset_name)
    msg += '\n          Use pre-trained model = %s' % args.pre_trained
    print(msg)

    return model