import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utee import misc
from util import fgsm_gt, pgd_gt, ifgsm_gt, grad_gt
from util_trts import model_train, model_test, model_train_admm

from pruning_tools import l0proj, idxproj, layers_nnz, layers_unique
from quantize import quantize_kmeans as kmeans

np.set_printoptions(threshold=np.nan)

import dataset
from caffelenet.caffelenet_dense import CaffeLeNet as CLdense
from caffelenet.caffelenet_abcv2 import CaffeLeNet as CLabcv2

parser = argparse.ArgumentParser(description="Pytorch MNIST bold")
parser.add_argument('--wd', type=float, default=1e-4, help="weight decay factor")
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--gpu', default="0", help="index of GPUs to use")
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--seed', type=int, default=117, help="random seed (default: 117)")
parser.add_argument('--log_interval', type=int, default=20, help="how many batches to wait before logging training status")
parser.add_argument('--test_interval', type=int, default=1, help="how many epochs to wait before another test")
parser.add_argument('--logdir', default='log/default', help='folder to load the log')
parser.add_argument('--savedir', default=None, help="folder to save the log")
parser.add_argument('--data_root', default='/media/hdd/mnist/', help='folder to save the data')
parser.add_argument('--decreasing_lr', default='80,120', help="decreasing strategy")
parser.add_argument('--attack_algo', default='fgsm', help='adversarial algo for attack')
parser.add_argument('--attack_eps', type=float, default=None, help='perturbation radius for attack phase')
parser.add_argument('--defend_algo', default=None, help='adversarial algo for defense')
parser.add_argument('--defend_eps', type=float, default=None, help='perturbation radius for defend phase')

parser.add_argument('--raw_train', type=bool, default=False, help="raw training without pre-train model loading")
parser.add_argument('--model_name', default=None, help="file name of pre-train model")
parser.add_argument('--save_model_name', default=None, help="save model name after training")

parser.add_argument('--prune_algo', default=None, help="pruning projection method")
parser.add_argument('--prune_ratio', type=float, default=0.1, help='sparse ratio or energy budget')
parser.add_argument('--quantize_algo', default=None, help="quantization method")
parser.add_argument('--quantize_bits', type=int, default=8, help="quantization bits")

parser.add_argument('--proj_interval', type=int, default=1, help="pruning interval along interation over batches")

parser.add_argument('--model_type', default="dense", help="select model type")

args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)

if args.savedir is None:
    args.savedir = args.logdir
else:
    args.savedir = os.path.join(os.path.dirname(__file__), args.savedir)

misc.logger.init(args.logdir, 'train_log')
print = misc.logger.info

# select gpu
args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)

# logger
misc.ensure_dir(args.logdir)
misc.ensure_dir(args.savedir)
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader, test_loader = dataset.get(batch_size=args.batch_size, data_root=args.data_root, num_workers=1)

if args.model_type == "dense":
    model = CLdense()
    modelp = CLdense()
    modelq = CLdense()
    weight_name = ["weight"]
elif args.model_type == "abc":
    model0 = CLdense()
    ranks_up = model0.get_ranks()
    model = CLabcv2(ranks_up)
    weight_name = ["weightA", "weightB", "weightC"]
else:
    raise NotImplementedError

if args.raw_train:
    exit(0)
else:
    descripter = args.defend_algo if args.defend_algo is not None else ""
    full_m = misc.model_loader(args.logdir, args.model_name, descripter=descripter)
    prune_m = misc.model_loader(args.logdir, args.model_name, descripter="sparse_" + descripter)
    quant_m = misc.model_loader(args.logdir, args.model_name, descripter="quant_" + descripter)

model.load_state_dict(full_m)
modelp.load_state_dict(prune_m)
modelq.load_state_dict(quant_m)

if args.cuda:
    model.cuda()
    modelp.cuda()
    modelq.cuda()

algo = {'fgsm': fgsm_gt, 'bim': ifgsm_gt, 'pgd': pgd_gt, 'grad': grad_gt}

attack_algo = algo[args.attack_algo] if args.attack_algo is not None else None
defend_algo = algo[args.defend_algo] if args.defend_algo is not None else None

defend_name = "None" if args.defend_algo is None else args.defend_algo

# if args.prune_algo == "l0proj":
#     prune_algo = l0proj
# elif args.prune_algo is None:
#     prune_algo = None
# elif args.prune_algo == "baseline":
#     prune_algo = l0proj
# else:
#     raise NotImplementedError

prune_name = "unify"
quant_name = "unify"

# if args.quantize_algo == "kmeans":
#     quantize_algo = kmeans
# else:
#     quantize_algo = None

# quantize_name = "None" if args.quantize_algo is None else args.quantize_algo

def print_dict(argslist, name):
    print("================={}==================".format(name))
    for k, v in argslist.items():
        print('{}: {}'.format(k, v))
    print("========================================")

try:
    model_test(model, 0, test_loader, 
            atk_algo=attack_algo, atk_eps=args.attack_eps, 
            iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)

    layers = layers_nnz(model, param_name=weight_name)[1]
    print_dict(layers, name="NONZERO SIZE")
    layers = layers_unique(model, normalized=False, param_name=weight_name)[1]
    print_dict(layers, name="UNIQUE SIZE")

    model_test(modelp, 0, test_loader,
            atk_algo=attack_algo, atk_eps=args.attack_eps,
            iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)
    layers = layers_nnz(modelp, param_name=weight_name)[1]
    print_dict(layers, name="NONZERO SIZE")
    layers = layers_unique(modelp, normalized=False, param_name=weight_name)[1]
    print_dict(layers, name="UNIQUE SIZE")

    model_test(modelq, 0, test_loader,
            atk_algo=attack_algo, atk_eps=args.attack_eps,
            iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)
    layers = layers_nnz(modelq, param_name=weight_name)[1]
    print_dict(layers, name="NONZERO SIZE")
    layers = layers_unique(modelq, normalized=False, param_name=weight_name)[1]
    print_dict(layers, name="UNIQUE SIZE")

    # if args.prune_algo == "baseline":
    #     prune_idx, Weight_shapes = prune_algo(model, args.prune_ratio, param_name=weight_name)
    #     prune_lambda = lambda m: idxproj(m, z_idx=prune_idx, W_shapes=Weight_shapes)
    # elif args.prune_algo == "l0proj":
    #     normalized = 0<=args.prune_ratio<=1.0
    #     prune_lambda = lambda m: prune_algo(m, args.prune_ratio, param_name=weight_name, normalized=normalized)
    # else:
    #     prune_lambda = None

    # if args.quantize_algo == "kmeans":
    #     quantize_lambda = lambda m: quantize_algo(m, bit_depth=args.quantize_bits)
    # else:
    #     quantize_lambda = None

    # if args.prune_algo is not None:
    #     prune_lambda(model)

    # model_test(model, 0, test_loader, 
    #         atk_algo=attack_algo, atk_eps=args.attack_eps, 
    #         iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)
    
    # layers = layers_nnz(model, param_name=weight_name)[1]
    # print_dict(layers, name="MODEL SIZE")

    # if args.quantize_algo is not None:
    #     quantize_lambda(model)

    # model_test(model, 0, test_loader, 
    #         atk_algo=attack_algo, atk_eps=args.attack_eps, 
    #         iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)

    # layers = layers_unique(model, normalized=False, param_name=weight_name)[1]
    # print_dict(layers, name="UNIQUE SIZE")

except Exception as e:
    import traceback
    traceback.print_exc()

finally:
    exit(0)