import argparse
import os
from collections import OrderedDict
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utee import misc
from util import fgsm_gt, pgd_gt, ifgsm_gt
from util_trts import model_train, model_test, model_train_admm, model_train_admm_ms

from pruning_tools import l0proj, idxproj, layers_nnz, prune_admm_ms, quantize_admm_ms
import pruning_tools as pt
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
parser.add_argument('--decreasing_lr', default='40,70,90', help="decreasing strategy")
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

parser.add_argument('--proj_interval', type=int, default=10, help="pruning interval along interation over batches")
parser.add_argument('--sparse_init', action="store_true")

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
misc.print_dict(args.__dict__, "FLAGS")

args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader, test_loader = dataset.get(batch_size=args.batch_size, data_root=args.data_root, num_workers=1)

model = CLdense()

if args.raw_train:
    exit(0)
else:
    if args.model_name is None:
        if args.defend_algo is not None:
            model_path = os.path.join(args.logdir, args.defend_algo + "_densepretrain.pth")
        else:
            model_path = os.path.join(args.logdir, '_densepretrain.pth')
    else:
        model_path = os.path.join(args.logdir, args.model_name)

model.load_state_dict(torch.load(model_path))
modelu, modelv, modelz, modely = CLdense(), CLdense(), CLdense(), CLdense()
modelu.empty_all()
modelv.empty_all()

modelz.load_state_dict(torch.load(model_path))
modely.load_state_dict(torch.load(model_path))

# model_feature = torch.nn.DataParallel(model_feature, device_ids=range(args.ngpu))
model = torch.nn.DataParallel(model, device_ids=range(args.ngpu))
modelz = torch.nn.DataParallel(modelz, device_ids=range(args.ngpu))
modely = torch.nn.DataParallel(modely, device_ids=range(args.ngpu))
modelu = torch.nn.DataParallel(modelu, device_ids=range(args.ngpu))
modelv = torch.nn.DataParallel(modelv, device_ids=range(args.ngpu))

if args.cuda:
    model.cuda()
    modelu.cuda()
    modelv.cuda()
    modelz.cuda()
    modely.cuda()

algo = {'fgsm': fgsm_gt, 'bim': ifgsm_gt, 'pgd': pgd_gt}
# attack_algo = algo[args.attack_algo]

attack_algo = algo[args.attack_algo] if args.attack_algo is not None else None
defend_algo = algo[args.defend_algo] if args.defend_algo is not None else None

defend_name = "None" if args.defend_algo is None else args.defend_algo

# prune_tk, quant_tk = prune_admm_ms, quantize_admm_ms

prune_name = "unify"
quant_name = "unify"

optimizer = optim.Adam(model.parameters(), lr=args.lr)

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

# crossloss = nn.CrossEntropyLoss
print('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()

weight_name_space = ["weight"]

try:
    # ready to go
    model_test(model, 0, test_loader, 
            atk_algo=attack_algo, atk_eps=args.attack_eps, 
            iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)
    
    if args.sparse_init:
        layers = pt.layers_nnz(model, param_name=weight_name_space)[1]
    else:
        layers = pt.layers_n(model, param_name=weight_name_space)[1]
        layers_cp = OrderedDict()
        all_num = sum(layers.values())
        for name, layer in layers.items():
            layers_cp[name] = int(max(layer * args.prune_ratio, 1))
        layers = layers_cp
    
    misc.print_dict(layers, name="NNZ PER LAYER")

    model_size = pt.modelsize_calculator(layers.values(), args.quantize_bits)
    print("\t MODEL SIZE {}".format(model_size))
    prune_lambda = lambda wl, wb: prune_admm_ms(wl, wb, model_size)
    quant_lambda = lambda wl, nnz, wnnz: quantize_admm_ms(wl, nnz, wnnz, model_size)
    weight_bits = [args.quantize_bits for _ in layers]
    layernnz = lambda m: list(pt.layers_nnz(m, param_name=weight_name_space)[1].values())
    param_list = lambda m: pt.param_list(m, param_name=weight_name_space)

    for epoch in range(args.epochs):

        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        
        _, weight_bits, clusters = model_train_admm_ms([model, modelz, modely, modelu, modelv], epoch, train_loader, optimizer,
            dfn_algo=defend_algo, dfn_eps=args.defend_eps, 
            log_interval=args.log_interval, layernnz=layernnz, param_list=param_list, iscuda=args.cuda, 
            adv_iter=16, criterion=F.cross_entropy, prune_tk=prune_lambda, rho=1e-2,
            quantize_tk=quant_lambda, admm_interval=args.proj_interval, weight_bits=weight_bits)

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time

        print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))

        if epoch % args.test_interval == 0:

            acc = model_test(model, epoch, test_loader, 
                    atk_algo=attack_algo, atk_eps=args.attack_eps, 
                    iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)

            layers = layers_nnz(modelz, param_name=weight_name_space)[1]
            misc.print_dict(layers, name="MODEL SIZE")
            if acc > best_acc:
                descripter = args.defend_algo if args.defend_algo is not None else ""
                misc.model_saver(model, args.savedir, args.save_model_name, descripter)
                misc.model_saver(modelz, args.savedir, args.save_model_name, "sparse_" + descripter)
                misc.model_saver(modely, args.savedir, args.save_model_name, "quant_" + descripter)
                import scipy.io as sio
                filepath = os.path.join(args.savedir, "config_" + "".join(args.save_model_name.split(".")[:-1]) + ".mat")
                sio.savemat(filepath, {"bits": weight_bits, "clusters": clusters})
                # with open(os.path.join(args.savedir, "config_" + "".join(args.save_model_name.split(".")[:-1]) + ".json"), "w") as jsfile:
                #     json.dump({"bits": weight_bits, "clusters": clusters}, jsfile)
                best_acc = acc
                # old_file = new_file

except Exception as e:
    import traceback
    traceback.print_exc()

finally:
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))
        