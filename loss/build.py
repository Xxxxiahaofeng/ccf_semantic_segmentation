from .losses import *


def build_loss(args):
    if args.loss == 'CrossEntropyLoss':
        return CrossEntropyLoss(ignore_index=args.ignore_value, weight=args.loss_weight)
