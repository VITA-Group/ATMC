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
from util_unnormalize import fgsm_gt, pgd_gt, ifgsm_gt
from util_trts import model_train, model_test, model_train_admm, model_train_proj_prune_admm_quant
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
from resnet.resnet_dense import ResNet101 as CLdense
from resnet.resnet_abcv2 import ResNet101 as CLabcv2
from resnet.resnet_lr import ResNet101 as CLlr

torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description="Pytorch MNIST bold")
parser.add_argument('--wd', type=float, default=1e-4, help="weight decay factor")
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 256)')
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
parser.add_argument('--save_model_name', default=None, help="save model name after training")

parser.add_argument('--prune_algo', default=None, help="pruning projection method")
parser.add_argument('--prune_ratio', type=float, default=0.1, help='sparse ratio or energy budget')
parser.add_argument('--quantize_algo', default=None, help="quantization method")
parser.add_argument('--quantize_bits', type=int, default=8, help="quantization bits")

parser.add_argument('--prune_interval', type=int, default=1, help="pruning interval along iteration over batches")
parser.add_argument("--quant_interval", type=int, default=5, help="quantize interval along iteration over batches")


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

train_loader, test_loader = dataset.get10(batch_size=args.batch_size, data_root=args.data_root, num_workers=1)

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

model_base = CLdense()
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
        model_path = os.path.join(args.loaddir, args.model_name)

if args.abc_special:
    if not args.abc_initialize:
        model_base.load_state_dict(torch.load(model_path))
        ranks_up = model_base.get_ranks()
        model_abc = CLabcv2(ranks_up)
        model_abc.set_weights(model_base.raw_weights(ranks_up))
        model_abc.load_state_dict(model_base.state_dict(), strict=False)
        model = model_abc
    else:
        ranks_up = model_base.get_ranks()
        model = CLabcv2(ranks_up)
        model.load_state_dict(torch.load(model_path))
    modelu, modelz = CLabcv2(ranks_up), CLabcv2(ranks_up)

elif args.lr_special:
    if not args.lr_initialize:
        model_base.load_state_dict(torch.load(model_path))
        weights_list, ranks_up = model_base.svd_global_lowrank_weights(k=args.prune_ratio)
        print(ranks_up)
        model = CLlr(ranks_up)
        # print([weight.shape for weight in weights_list])
        # print(weights_list)
        model.set_weights(weights_list)
        model.load_state_dict(model_base.state_dict(), strict=False)
        # descripter = "{}_proj_{}_nnz_{}_quant_{}_bits_{}_".format(defend_name, prune_name, args.prune_ratio, quantize_name, args.quantize_bits)
        # with open(os.path.join(args.savedir, "param_" + descripter + "lr.npy"), "wb") as filestream:
        with open(os.path.join(args.savedir, args.save_model_name[0:-4] + ".npy"), "wb") as filestream:
            pickle.dump(ranks_up, filestream, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # with open(os.path.join(args.loaddir, "param_" + descripter + "lr.npy"), "rb") as filestream:
        fname = os.path.join(args.loaddir, 'cifar_alr_' + str(args.prune_ratio) + ".npy")
        print('+++++++++++++++++++')
        print(fname)
        print('+++++++++++++++++++')
        with open(fname, "rb") as filestream:
            ranks_up = pickle.load(filestream)
        model = CLlr(ranks_up)
        model.load_state_dict(torch.load(model_path))
    modelu, modelz = CLlr(ranks_up), CLlr(ranks_up)

else:
    if not args.raw_train:
        model_base.load_state_dict(torch.load(model_path))
    model = model_base
    modelu, modelz = CLdense(), CLdense()

modelu.empty_all()

modelz.load_state_dict(model.state_dict())

# if not args.raw_train:
#     modelz.load_state_dict()
# model_feature = torch.nn.DataParallel(model_feature, device_ids=range(args.ngpu))

if args.cuda:
    model = torch.nn.DataParallel(model, device_ids=range(args.ngpu))
    modelu = torch.nn.DataParallel(modelu, device_ids=range(args.ngpu))
    modelz = torch.nn.DataParallel(modelz, device_ids=range(args.ngpu))
    model.cuda()
    modelu.cuda()
    modelz.cuda()

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
best_dist = np.inf

try:
    # ready to go
    model_test(model, 0, test_loader, 
            atk_algo=attack_algo, atk_eps=args.attack_eps, 
            iscuda=args.cuda, adv_iter=3, criterion=F.cross_entropy)
    
    layers = layers_nnz(model, param_name=weight_name)[1]
    # misc.print_dict(layers, name="NNZ PER LAYER")

    layers_n = pt.layers_n(model_base, param_name=["weight"])[1]
    all_num = sum(layers_n.values())
    # print(all_num)
    sparse_factor = int(all_num * args.prune_ratio)
    # print(sparse_factor)
    model_size = sparse_factor * args.quantize_bits
    print("\t MODEL SIZE {}".format(model_size))
    weight_bits = [args.quantize_bits for _ in layers] if args.quantize_algo == "model_size_quant" else None
    print("\t weight bits {}".format(weight_bits))
    layernnz = lambda m: list(pt.layers_nnz(m, param_name=weight_name)[1].values())
    param_list = lambda m: pt.param_list(m, param_name=weight_name)

    # print("\t MODEL SIZE {}".format(model_size))

    print(args.prune_algo)

    if args.prune_algo == "baseline":
        # prune_idx, Weight_shapes = prune_algo(model, args.prune_ratio, param_name=weight_name)
        prune_idx, Weight_shapes = prune_algo(model, sparse_factor, normalized=False, param_name=weight_name)
        prune_lambda = lambda m: idxproj(m, z_idx=prune_idx, W_shapes=Weight_shapes)
    elif args.prune_algo == "l0proj":
        prune_lambda = lambda m: prune_algo(m, sparse_factor, normalized=False, param_name=weight_name)
    elif args.prune_algo == "model_size_prune":
        prune_lambda = lambda wl, wb: prune_algo(wl, wb, model_size)
    elif args.prune_algo == 'low_rank':
        prune_lambda = None
    else:
        prune_lambda = None

    if args.quantize_algo == "kmeans":
        quantize_lambda = lambda m: quantize_algo(m, bit_depth=args.quantize_bits)
    elif args.quantize_algo == "kmeans_nnz":
        quantize_lambda = lambda m: quantize_algo(m, model, bit_depth=args.quantize_bits)
    elif args.quantize_algo == "kmeans_nnz_fixed_0_center":
        quantize_lambda = lambda m: quantize_algo(m, model, bit_depth=args.quantize_bits)
    elif args.quantize_algo == "kmeans_fixed_nnz":
        quantize_lambda = lambda m: quantize_algo(m, model, bit_depth=args.quantize_bits)
    elif args.quantize_algo == "model_size_quant":
        quantize_lambda = lambda wl, nnz, wnnz: quantize_algo(wl, nnz, wnnz, model_size)
    else:
        quantize_lambda = None

    # descripter = "{}_proj_{}_nnz_{}_quant_{}_bits_{}_".format(defend_name, prune_name, args.prune_ratio, quantize_name, args.quantize_bits)
    descripter = ''
    for epoch in range(args.epochs):

        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        
        # print("quant_interval {}".format(args.quant_interval))
        message = model_train_proj_prune_admm_quant([model, modelz, modelu], epoch, train_loader, optimizer,
            dfn_algo=defend_algo, dfn_eps=args.defend_eps, 
            log_interval=args.log_interval, iscuda=args.cuda, nnzfix=args.prune_algo=="baseline",
            param_list=param_list, layernnz=layernnz,
            adv_iter=args.defend_iter, criterion=F.cross_entropy, prune_tk=prune_lambda, 
            quantize_tk=quantize_lambda, rho=0.1, admm_interval=args.quant_interval, proj_interval=args.prune_interval, weight_bits=weight_bits)

        if weight_bits is not None:
            train_acc, weight_bits = message
        else:
            train_acc = message[0]

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time

        print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))
        
        # if train_acc > best_train_acc:
        misc.model_saver(model, args.savedir, args.save_model_name, "sparse_latest_" + descripter)
        if quantize_algo is not None:
            misc.model_saver(modelz, args.savedir, args.save_model_name, "quant_latest_" + descripter)
            # best_train_acc = train_acc

        if epoch % args.test_interval == 0:

            acc, acc_adv = model_test(model, epoch, test_loader, 
                    atk_algo=attack_algo, atk_eps=args.attack_eps, 
                    iscuda=args.cuda, adv_iter=args.defend_iter, criterion=F.cross_entropy)

            # acc = model_uai_test(model_feature, model_pred, epoch, test_loader, attack_algo, args.attack_eps, iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)
            layers = layers_nnz(model, param_name=weight_name)[1]
            # misc.print_dict(layers, name="MODEL SIZE")
            if quantize_lambda is not None:
                if isinstance(modelz, nn.DataParallel):
                    modelz.module.replace_bias(model.module, weight_name)
                else:
                    modelz.replace_bias(model, weight_name)
                acc, acc_adv = model_test(modelz, epoch, test_loader,
                        atk_algo=attack_algo, atk_eps=args.attack_eps, 
                        iscuda=args.cuda, adv_iter=args.defend_iter, criterion=F.cross_entropy)

                layers = layers_unique(modelz, normalized=False, param_name=weight_name)[1]
                # misc.print_dict(layers, name="UNIQUE SIZE")
            if quantize_lambda is not None:
                dist, dist_layer, relative_layer = util_trts.model_distance(model, modelz, weight_name=weight_name)
                print("\t model dist: {}, layer-wise dist :{}, relative: {}".format(dist, dist_layer, relative_layer))
                if dist < best_dist:
                    best_dist = dist
                    misc.model_saver(model, args.savedir, args.save_model_name, "sparse_closest_" + descripter)
                    misc.model_saver(modelz, args.savedir, args.save_model_name, "quant_closest_" + descripter)
                    print("\t\t closest model saved")
                
            if acc_adv > best_acc:
                # new_file = os.path.join(args.loaddir, 'best-{}.pth'.format(epoch))
                # descripter = args.defend_algo if args.defend_algo is not None else ""
                misc.model_saver(model, args.savedir, args.save_model_name, "sparse_" + descripter)
                if quantize_algo is not None:
                    misc.model_saver(modelz, args.savedir, args.save_model_name, "quant_" + descripter)
                # if args.save_model_name is None:
                #     if args.defend_algo is not None:
                #         misc.model_snapshot(model, os.path.join(args.savedir, "sparse_" + args.defend_algo+'_densepretrain.pth'))
                #         misc.model_snapshot(modelz, os.path.join(args.savedir, "quant_" + args.defend_algo + "_densepretrain.pth"))
                #     else:
                #         misc.model_snapshot(model, os.path.join(args.savedir, 'sparse_densepretrain.pth'))
                #         misc.model_snapshot(modelz, os.path.join(args.savedir, 'quant_densepretrain.pth'))
                # else:
                #     misc.model_snapshot(model, os.path.join(args.savedir, "sparse_" + args.save_model_name))
                #     misc.model_snapshot(modelz, os.path.join(args.savedir, "quant_" + args.save_model_name))
                best_acc = acc_adv
                # old_file = new_file

except Exception as e:
    import traceback
    traceback.print_exc()

finally:
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))
        