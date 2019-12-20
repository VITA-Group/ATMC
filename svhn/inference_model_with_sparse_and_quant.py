import argparse, pickle
import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'utee'))
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utee import misc
from util import fgsm_gt, pgd_gt, ifgsm_gt, wrm_gt
from util_trts import model_test, model_train_proj_prune_admm_quant
import util_trts

from pruning_tools import l0proj, idxproj, layers_nnz, layers_unique
import pruning_tools as pt
from quantize import quantize_kmeans as kmeans
from quantize import quantize_kmeans_nnz as kmeans_nnz
from quantize import quantize_kmeans_fixed_nnz as kmeans_fixed_nnz
from quantize import quantize_kmeans_nnz_fixed_0_center as kmeans_nnz_fixed_0_center

# np.set_printoptions(threshold=np.nan)

import dataset
# from caffelenet.caffelenet_dense import CaffeLeNet as CLdense
# from caffelenet.caffelenet_abcv2 import CaffeLeNet as CLabcv2
# from caffelenet.caffelenet_lr import CaffeLeNet as CLlr
from wrn.wideresnet_dense import WideResNet as CLdense
from wrn.wideresnet_abcv2 import WideResNet as CLabcv2
from wrn.wideresnet_lr import WideResNet as CLlr


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
parser.add_argument('--loaddir', default='log/default', help='folder to load the log')
parser.add_argument('--savedir', default=None, help="folder to save the log")
parser.add_argument('--data_root', default='/media/hdd/mnist/', help='folder to save the data')
parser.add_argument('--decreasing_lr', default='40,70,90', help="decreasing strategy")
parser.add_argument('--attack_algo', default='fgsm', help='adversarial algo for attack')
parser.add_argument('--attack_eps', type=float, default=None, help='perturbation radius for attack phase')
parser.add_argument('--attack_iter', type=int, default=7, help="attack iteration for the adversarial sample computation")
parser.add_argument('--defend_algo', default=None, help='adversarial algo for defense')
parser.add_argument('--defend_eps', type=float, default=None, help='perturbation radius for defend phase')
parser.add_argument('--defend_iter', type=int, default=7, help="defend iteration for the adversarial sample computation")

# parser.add_argument('--raw_train', type=bool, default=False, help="raw training without pre-train model loading")
parser.add_argument("--raw_train", action="store_true")
parser.add_argument("--abc_special", action="store_true")
parser.add_argument("--abc_initialize", action="store_true")
parser.add_argument("--lr_special", action="store_true")
parser.add_argument("--lr_initialize", action="store_true")
parser.add_argument('--model_name', default=None, help="file name of pre-train model")
parser.add_argument('--prefix_name', default="", help="save model name after training")

parser.add_argument('--prune_algo', default=None, help="pruning projection method")
parser.add_argument('--prune_ratio', type=float, default=0.1, help='sparse ratio or energy budget')
parser.add_argument('--quantize_algo', default=None, help="quantization method")
parser.add_argument('--quantize_bits', type=int, default=8, help="quantization bits")

parser.add_argument('--prune_interval', type=int, default=1, help="pruning interval along iteration over batches")
parser.add_argument("--quant_interval", type=int, default=5, help="quantize interval along iteration over batches")

parser.add_argument("-e", "--exp_logger", default=None, help="exp results stored to")



args = parser.parse_args()
args.loaddir = os.path.join(os.path.dirname(__file__), args.loaddir)

if args.savedir is None:
    args.savedir = args.loaddir
else:
    args.savedir = os.path.join(os.path.dirname(__file__), args.savedir)

misc.logger.init(args.loaddir, 'train_log')
print = misc.logger.info


# select gpu
args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)

# logger
misc.ensure_dir(args.loaddir)
misc.ensure_dir(args.savedir)
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader, test_loader = dataset.get(batch_size=args.batch_size, data_root=args.data_root, num_workers=4)

algo = {'fgsm': fgsm_gt, 'bim': ifgsm_gt, 'pgd': pgd_gt, 'wrm': wrm_gt}
# attack_algo = algo[args.attack_algo]

attack_algo = algo[args.attack_algo] if args.attack_algo is not None else None
defend_algo = algo[args.defend_algo] if args.defend_algo is not None else None

defend_name = "None" if args.defend_algo is None else args.defend_algo

if args.prune_algo == "l0proj":
    prune_algo = l0proj
elif args.prune_algo is None:
    prune_algo = None
elif args.prune_algo == "baseline":
    prune_algo = l0proj
elif args.prune_algo == "model_size_prune":
    prune_algo = pt.prune_admm_ms
elif args.prune_algo == "low_rank":
    prune_algo = None
else:
    raise NotImplementedError

prune_name = "None" if args.prune_algo is None else args.prune_algo

if args.quantize_algo == "kmeans":
    quantize_algo = kmeans
elif args.quantize_algo == "kmeans_nnz":
    quantize_algo = kmeans_nnz
elif args.quantize_algo == "kmeans_fixed_nnz":
    quantize_algo = kmeans_fixed_nnz
elif args.quantize_algo == "kmeans_nnz_fixed_0_center":
    quantize_algo = kmeans_nnz_fixed_0_center
elif args.quantize_algo == "model_size_quant":
    quantize_algo = pt.quantize_admm_ms
else:
    quantize_algo = None

quantize_name = "None" if args.quantize_algo is None else args.quantize_algo

# descripter = "{}_proj_{}_nnz_{}_quant_{}_bits_{}_"\
#     .format(defend_name, prune_name, args.prune_ratio, quantize_name, args.quantize_bits) \
#         if args.prune_algo is not None or args.quantize_algo is not None else ""
descripter = ''


model_base = CLdense(depth=16, num_classes=10, widen_factor=8, dropRate=0.4)
weight_name = ["weight"] if not args.abc_special else ["weightA", "weightB", "weightC"]
weight_name = ["weightA", "weightB"] if args.lr_special else weight_name

if args.raw_train:
    pass
else:
    if args.model_name is None:
        if args.defend_algo is not None:
            model_path = os.path.join(args.loaddir, args.defend_algo + "_densepretrain.pth")
        else:
            model_path = os.path.join(args.loaddir, '_densepretrain.pth')
    else:
        sparse_model_path = os.path.join(args.loaddir, "sparse_" + descripter + args.model_name)
        quant_model_path = os.path.join(args.loaddir, "quant_" + descripter + args.model_name)

if args.abc_special:
    if not args.abc_initialize:
        model_base.load_state_dict(torch.load(model_path))
        ranks_up = model_base.get_ranks()
        model_abc = CLabcv2(depth=16, num_classes=10, widen_factor=8, dropRate=0.4, ranks=ranks_up)
        model_abc.set_weights(model_base.raw_weights(ranks_up))
        model_abc.load_state_dict(model_base.state_dict(), strict=False)
        model = model_abc
    else:
        ranks_up = model_base.get_ranks()
        sparse_model = CLabcv2(depth=16, num_classes=10, widen_factor=8, dropRate=0.4, ranks=ranks_up)
        quant_model = CLabcv2(depth=16, num_classes=10, widen_factor=8, dropRate=0.4, ranks=ranks_up)
        sparse_model.load_state_dict(torch.load(sparse_model_path))
        quant_model.load_state_dict(torch.load(quant_model_path))

    modelu, modelz = CLabcv2(depth=16, num_classes=10, widen_factor=8, dropRate=0.4, ranks=ranks_up), CLabcv2(depth=16, num_classes=10, widen_factor=8, dropRate=0.4, ranks=ranks_up)
elif args.lr_special:
    if not args.lr_initialize:
        model_base.load_state_dict(torch.load(model_path))
        weights_list, ranks_up = model_base.svd_global_lowrank_weights(k=args.prune_ratio)
        print(ranks_up)
        model = CLlr(depth=16, num_classes=10, widen_factor=8, dropRate=0.4, ranks=ranks_up)
        # print([weight.shape for weight in weights_list])
        # print(weights_list)
        model.set_weights(weights_list)
        model.load_state_dict(model_base.state_dict(), strict=False)
        # descripter = "{}_proj_{}_nnz_{}_quant_{}_bits_{}_".format(defend_name, prune_name, args.prune_ratio, quantize_name, args.quantize_bits) 
        with open(os.path.join(args.savedir, args.model_name[0:-4] + ".npy"), "rb") as filestream:
            pickle.dump(ranks_up, filestream, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(args.loaddir, args.model_name[0:-4] + ".npy"), "rb") as filestream:
            ranks_up = pickle.load(filestream)
        sparse_model = CLlr(depth=16, num_classes=10, widen_factor=8, dropRate=0.4, ranks=ranks_up)
        quant_model = CLlr(depth=16, num_classes=10, widen_factor=8, dropRate=0.4, ranks=ranks_up)
        sparse_model.load_state_dict(torch.load(sparse_model_path))
        quant_model.load_state_dict(torch.load(quant_model_path))
    modelu, modelz = CLlr(depth=16, num_classes=10, widen_factor=8, dropRate=0.4, ranks=ranks_up), CLlr(depth=16, num_classes=10, widen_factor=8, dropRate=0.4, ranks=ranks_up)

else:
    model_base.load_state_dict(torch.load(sparse_model_path))
    sparse_model = model_base
    quant_model = CLdense(depth=16, num_classes=10, widen_factor=8, dropRate=0.4)
    quant_model.load_state_dict(torch.load(quant_model_path))
    modelu, modelz = CLdense(depth=16, num_classes=10, widen_factor=8, dropRate=0.4), CLdense(depth=16, num_classes=10, widen_factor=8, dropRate=0.4)

modelu.empty_all()

modelz.load_state_dict(quant_model.state_dict())

# if not args.raw_train:
#     modelz.load_state_dict()
# model_feature = torch.nn.DataParallel(model_feature, device_ids=range(args.ngpu))
if args.cuda:
    sparse_model.cuda()
    quant_model.cuda()
    modelu.cuda()
    modelz.cuda()


# optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
# optim_pred = optim.Adam(list(model_feature.parameters()) + list(model_pred.parameters()), lr=args.lr)
# optim_distortion = optim.Adam(model_distortion.parameters())

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

# crossloss = nn.CrossEntropyLoss
print('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()
best_train_acc = 0.
best_dist = np.inf

f = open(os.path.join(args.loaddir, args.exp_logger), "a+")
f.write('prune_ratio: %f, quantize_bits: %d, attack: %s-%d\n' % (args.prune_ratio, args.quantize_bits, args.attack_algo, args.attack_eps))

try:
    import math
    # ready to go
    if isinstance(quant_model, nn.DataParallel):
        quant_model.module.replace_bias(sparse_model.module, weight_name)
    else:
        quant_model.replace_bias(sparse_model, weight_name)
    acc, acc_adv = model_test(quant_model, 0, test_loader, 
            atk_algo=attack_algo, atk_eps=args.attack_eps, 
            iscuda=args.cuda, adv_iter=args.attack_iter, criterion=F.cross_entropy)
    
    layers = layers_nnz(quant_model, param_name=weight_name)[1]
    misc.print_dict(layers, name="NNZ PER LAYER")
    # print(all_num)
    sparse_factor = sum(layers.values())
    # print(sparse_factor)
    if quantize_algo is not None:
        if args.quantize_algo == "kmeans_nnz_fixed_0_center":
            weight_bits = [math.ceil(math.log2(item-1)) if item > 1 else 0 for item in list(pt.layers_unique(quant_model, weight_name)[1].values())]
            weight_dict_size = [ (item - 1) * 32 for item in list(pt.layers_unique(quant_model, weight_name)[1].values())] 
        elif args.quantize_algo == "kmeans":
            weight_bits = [math.ceil(math.log2(item)) for item in list(pt.layers_unique(quant_model, weight_name)[1].values())]
            weight_dict_size = [item * 32 for item in list(pt.layers_unique(quant_model, weight_name)[1].values())]
        elif args.quantize_algo == "kmeans_fixed_nnz":
            weight_bits = [math.ceil(math.log2(item-1)) if item > 1 else 0 for item in list(pt.layers_unique(quant_model, weight_name)[1].values())]
            weight_dict_size = [ (item - 1) * 32 for item in list(pt.layers_unique(quant_model, weight_name)[1].values())] 
        else:
            raise 'quantize algo error!'
    else:
        weight_bits = [32] * len(layers)
    model_size = 0
    for i in range(len(layers)):
        dict_size = weight_dict_size[i] if quantize_algo is not None else 0
        nnz_cur = list(layers.values())[i]
        model_size += nnz_cur * weight_bits[i] + dict_size if 2 ** weight_bits[i] < nnz_cur or weight_bits[i] == 32 else nnz_cur * 32
        
    print("\t weight bits {}".format(weight_bits))
    print("\t MODEL SIZE {}".format(model_size))
    f.write("TA: %.4f, ATA: %.4f, MODEL SIZE %d\n" % (acc/100, acc_adv/100, model_size))
    layernnz = lambda m: list(pt.layers_nnz(m, param_name=weight_name)[1].values())
    param_list = lambda m: pt.param_list(m, param_name=weight_name)

    # print("\t MODEL SIZE {}".format(model_size))

    if args.prune_algo == "baseline":
        prune_idx, Weight_shapes = prune_algo(quant_model, args.prune_ratio, param_name=weight_name)
        # prune_idx, Weight_shapes = prune_algo(model, sparse_factor, normalized=False, param_name=weight_name)
        prune_lambda = lambda m: idxproj(m, z_idx=prune_idx, W_shapes=Weight_shapes)
    elif args.prune_algo == "l0proj":
        # prune_lambda = lambda m: prune_algo(m, sparse_factor, normalized=False, param_name=weight_name)
        prune_lambda = lambda m: prune_algo(m, args.prune_ratio, normalized=True, param_name=weight_name)
    elif args.prune_algo == "model_size_prune":
        prune_lambda = lambda wl, wb: prune_algo(wl, wb, model_size)
    elif args.prune_algo == 'low_rank':
        prune_lambda = None
    else:
        prune_lambda = None

    if args.quantize_algo == "kmeans":
        quantize_lambda = lambda m: quantize_algo(m, bit_depth=args.quantize_bits)
    elif args.quantize_algo == "kmeans_nnz":
        quantize_lambda = lambda m: quantize_algo(m, sparse_model, bit_depth=args.quantize_bits)
    elif args.quantize_algo == "kmeans_nnz_fixed_0_center":
        quantize_lambda = lambda m: quantize_algo(m, sparse_model, bit_depth=args.quantize_bits)
    elif args.quantize_algo == "kmeans_fixed_nnz":
        quantize_lambda = lambda m: quantize_algo(m, sparse_model, bit_depth=args.quantize_bits)
    elif args.quantize_algo == "model_size_quant":
        quantize_lambda = lambda wl, nnz, wnnz: quantize_algo(wl, nnz, wnnz, model_size)
    else:
        quantize_lambda = None

    if quantize_lambda is not None and args.prune_algo is not None and args.prune_algo not in ["low_rank", "model_size_prune"]:
        f.write("-----------------------------------------------\r\n")
        prune_lambda(quant_model)
        acc, acc_dv = model_test(quant_model, 1, test_loader, 
            atk_algo=attack_algo, atk_eps=args.attack_eps, 
            iscuda=args.cuda, adv_iter=args.attack_iter, criterion=F.cross_entropy)

        layers = layers_nnz(quant_model, param_name=weight_name)[1]
        misc.print_dict(layers, name="NNZ PER LAYER")
        # print(all_num)
        sparse_factor = sum(layers.values())
        # print(sparse_factor)
        if quantize_algo is not None:
            if args.quantize_algo == "kmeans_nnz_fixed_0_center":
                weight_bits = [math.ceil(math.log2(item-1)) if item > 1 else 0 for item in list(pt.layers_unique(quant_model, weight_name)[1].values())]
                weight_dict_size = [ (item - 1) * 32 for item in list(pt.layers_unique(quant_model, weight_name)[1].values())] 
            elif args.quantize_algo == "kmeans":
                weight_bits = [math.ceil(math.log2(item)) for item in list(pt.layers_unique(quant_model, weight_name)[1].values())]
                weight_dict_size = [item * 32 for item in list(pt.layers_unique(quant_model, weight_name)[1].values())]
            elif args.quantize_algo == "kmeans_fixed_nnz":
                weight_bits = [math.ceil(math.log2(item-1)) if item > 1 else 0 for item in list(pt.layers_unique(quant_model, weight_name)[1].values())]
                weight_dict_size = [ (item - 1) * 32 for item in list(pt.layers_unique(quant_model, weight_name)[1].values())] 
            else:
                raise 'quantize algo error!'
        else:
            weight_bits = [32] * len(layers)
        model_size = 0
        for i in range(len(layers)):
            dict_size = weight_dict_size[i] if quantize_algo is not None else 0
            nnz_cur = list(layers.values())[i]
            model_size += nnz_cur * weight_bits[i] + dict_size if 2 ** weight_bits[i] < nnz_cur or weight_bits[i] == 32 else nnz_cur * 32
        
        print("\t weight bits {}".format(weight_bits))
        print("\t MODEL SIZE {}".format(model_size))
        f.write("TA: %.4f, ATA: %.4f, MODEL SIZE %d\n" % (acc/100, acc_adv/100, model_size))

    f.close()

except Exception as e:
    import traceback
    traceback.print_exc()

finally:
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))
    f.close()    