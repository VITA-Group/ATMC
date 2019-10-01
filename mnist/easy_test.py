import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utee import misc
from util import fgsm_gt, pgd_gt, ifgsm_gt
from util_trts import model_train, model_test, model_train_admm, model_train_proj_prune_admm_quant

from pruning_tools import l0proj, idxproj, layers_nnz, layers_unique
from quantize import quantize_kmeans as kmeans
from quantize import quantize_kmeans_nnz as kmeans_nnz
from quantize import quantize_kmeans_fixed_nnz as kmeans_fixed_nnz

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
parser.add_argument('--decreasing_lr', default='40,70,90', help="decreasing strategy")
parser.add_argument('--attack_algo', default='fgsm', help='adversarial algo for attack')
parser.add_argument('--attack_eps', type=float, default=None, help='perturbation radius for attack phase')
parser.add_argument('--defend_algo', default=None, help='adversarial algo for defense')
parser.add_argument('--defend_eps', type=float, default=None, help='perturbation radius for defend phase')
parser.add_argument('--defend_iter', type=int, default=16, help="defend iteration for the adversarial sample computation")

# parser.add_argument('--raw_train', type=bool, default=False, help="raw training without pre-train model loading")
parser.add_argument("--raw_train", action="store_true")
parser.add_argument('--model_name', default=None, help="file name of pre-train model")
parser.add_argument('--save_model_name', default=None, help="save model name after training")

parser.add_argument('--prune_algo', default=None, help="pruning projection method")
parser.add_argument('--prune_ratio', type=float, default=0.1, help='sparse ratio or energy budget')
parser.add_argument('--quantize_algo', default=None, help="quantization method")
parser.add_argument('--quantize_bits', type=int, default=8, help="quantization bits")

parser.add_argument('--prune_interval', type=int, default=1, help="pruning interval along iteration over batches")
parser.add_argument("--quant_interval", type=int, default=5, help="quantize interval along iteration over batches")


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

model = CLdense()
weight_name = ["weight"]

if args.raw_train:
    pass
else:
    if args.model_name is None:
        if args.defend_algo is not None:
            model_path = os.path.join(args.logdir, args.defend_algo + "_densepretrain.pth")
        else:
            model_path = os.path.join(args.logdir, '_densepretrain.pth')
    else:
        model_path = os.path.join(args.logdir, args.model_name)

    model.load_state_dict(torch.load(model_path))
modelu, modelz = CLdense(), CLdense()
modelu.empty_all()

if not args.raw_train:
    modelz.load_state_dict(torch.load(model_path))
# model_feature = torch.nn.DataParallel(model_feature, device_ids=range(args.ngpu))
if args.cuda:
    model.cuda()
    modelu.cuda()
    modelz.cuda()

algo = {'fgsm': fgsm_gt, 'bim': ifgsm_gt, 'pgd': pgd_gt}
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
else:
    raise NotImplementedError

prune_name = "None" if args.prune_algo is None else args.prune_algo

if args.quantize_algo == "kmeans":
    quantize_algo = kmeans
if args.quantize_algo == "kmeans_nnz":
    quantize_algo = kmeans_nnz
if args.quantize_algo == "kmeans_fixed_nnz":
    quantize_algo = kmeans_fixed_nnz
else:
    quantize_algo = None

quantize_name = "None" if args.quantize_algo is None else args.quantize_algo

# optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
# optim_pred = optim.Adam(list(model_feature.parameters()) + list(model_pred.parameters()), lr=args.lr)
# optim_distortion = optim.Adam(model_distortion.parameters())

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

# crossloss = nn.CrossEntropyLoss
print('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()
best_train_acc = 0.

try:
    # ready to go
    model_test(model, 0, test_loader, 
            atk_algo=attack_algo, atk_eps=args.attack_eps, 
            iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)
    
    layers = layers_nnz(model, param_name=weight_name)[1]
    misc.print_dict(layers, name="MODEL SIZE")


    res = {}
    count_res = {}
    normalized = False
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in weight_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            layer_name = name
            # W_nz = torch.nonzero(W.data)
            W_nz = W.data
            # print("{} {}".format(layer_name, W.data.shape))
            if W_nz.dim() > 0:
                if not normalized:
                    res[layer_name] = (W_nz ** 2).sum()
                else:
                    # print("{} layer nnz:{}".format(name, torch.nonzero(W.data)))
                    res[layer_name] = float(W_nz.shape[0]) / torch.numel(W)
                count_res[layer_name] =(W_nz ** 2).sum() / W_nz.shape[0]
            else:
                res[layer_name] = 0
                count_res[layer_name] = 0
    
    misc.print_dict(count_res, name="NORM")

    # if args.prune_algo == "baseline":
    #     prune_idx, Weight_shapes = prune_algo(model, args.prune_ratio, param_name=["weight"])
    #     prune_lambda = lambda m: idxproj(m, z_idx=prune_idx, W_shapes=Weight_shapes)
    # elif args.prune_algo == "l0proj":
    #     prune_lambda = lambda m: prune_algo(m, args.prune_ratio, param_name=["weight"])
    # else:
    #     prune_lambda = None

    # if args.quantize_algo == "kmeans":
    #     quantize_lambda = lambda m: quantize_algo(m, bit_depth=args.quantize_bits)
    # if args.quantize_algo == "kmeans_nnz":
    #     quantize_lambda = lambda m: quantize_algo(m, model, bit_depth=args.quantize_bits)
    # if args.quantize_algo == "kmeans_fixed_nnz":
    #     quantize_lambda = lambda m: quantize_algo(m, model, bit_depth=args.quantize_bits)
    # else:
    #     quantize_lambda = None

    # for epoch in range(args.epochs):

    #     if epoch in decreasing_lr:
    #         optimizer.param_groups[0]['lr'] *= 0.1
        
    #     train_acc = model_train_proj_prune_admm_quant([model, modelz, modelu], epoch, train_loader, optimizer,
    #         dfn_algo=defend_algo, dfn_eps=args.defend_eps, 
    #         log_interval=args.log_interval, iscuda=args.cuda, 
    #         adv_iter=args.defend_iter, criterion=F.cross_entropy, prune_tk=prune_lambda, 
    #         quantize_tk=quantize_lambda, admm_interval=args.quant_interval, proj_interval=args.prune_interval)

    #     elapse_time = time.time() - t_begin
    #     speed_epoch = elapse_time / (epoch + 1)
    #     speed_batch = speed_epoch / len(train_loader)
    #     eta = speed_epoch * args.epochs - elapse_time

    #     print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
    #         elapse_time, speed_epoch, speed_batch, eta))
        
    #     if train_acc > best_train_acc:
    #         descripter = "{}_proj_{}_nnz_{}_quant_{}_bits_{}_".format(defend_name, prune_name, args.prune_ratio, quantize_name, args.quantize_bits)
    #         misc.model_saver(model, args.savedir, args.save_model_name, "sparse_latest_" + descripter)
    #         misc.model_saver(modelz, args.savedir, args.save_model_name, "quant_latest_" + descripter)

    #     if epoch % args.test_interval == 0:

    #         acc = model_test(model, epoch, test_loader, 
    #                 atk_algo=attack_algo, atk_eps=args.attack_eps, 
    #                 iscuda=args.cuda, adv_iter=args.defend_iter, criterion=F.cross_entropy)

    #         # acc = model_uai_test(model_feature, model_pred, epoch, test_loader, attack_algo, args.attack_eps, iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)
    #         layers = layers_nnz(model, param_name=weight_name)[1]
    #         misc.print_dict(layers, name="MODEL SIZE")
    #         layers = layers_unique(modelz, normalized=False, param_name=weight_name)[1]
    #         misc.print_dict(layers, name="UNIQUE SIZE")
    #         if acc > best_acc:
    #             # new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
    #             # descripter = args.defend_algo if args.defend_algo is not None else ""
    #             descripter = "{}_proj_{}_nnz_{}_quant_{}_bits_{}_".format(defend_name, prune_name, args.prune_ratio, quantize_name, args.quantize_bits)
    #             misc.model_saver(model, args.savedir, args.save_model_name, "sparse_" + descripter)
    #             misc.model_saver(modelz, args.savedir, args.save_model_name, "quant_" + descripter)
    #             # if args.save_model_name is None:
    #             #     if args.defend_algo is not None:
    #             #         misc.model_snapshot(model, os.path.join(args.savedir, "sparse_" + args.defend_algo+'_densepretrain.pth'))
    #             #         misc.model_snapshot(modelz, os.path.join(args.savedir, "quant_" + args.defend_algo + "_densepretrain.pth"))
    #             #     else:
    #             #         misc.model_snapshot(model, os.path.join(args.savedir, 'sparse_densepretrain.pth'))
    #             #         misc.model_snapshot(modelz, os.path.join(args.savedir, 'quant_densepretrain.pth'))
    #             # else:
    #             #     misc.model_snapshot(model, os.path.join(args.savedir, "sparse_" + args.save_model_name))
    #             #     misc.model_snapshot(modelz, os.path.join(args.savedir, "quant_" + args.save_model_name))
    #             best_acc = acc
    #             # old_file = new_file

except Exception as e:
    import traceback
    traceback.print_exc()

finally:
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))
        