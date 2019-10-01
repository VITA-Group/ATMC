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
from util_trts import model_train, model_test, model_train_admm

from pruning_tools import l0proj, idxproj, layers_nnz, layers_n
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
parser.add_argument('--decreasing_lr', default="20,40,60,80", help="decreasing strategy")
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

if args.raw_train:
    exit(0)

model_basic = CLdense()
if args.prune_ratio < 1.:
    layers = layers_n(model_basic, param_name=['weight'])[1]
    num_param = sum(layers.values())
    args.prune_ratio = int(num_param * args.prune_ratio)
    print("total num of param {}".format(args.prune_ratio))

if args.model_name is None:
    if args.defend_algo is not None:
        model_path = os.path.join(args.logdir, args.defend_algo + "_densepretrain.pth")
    else:
        model_path = os.path.join(args.logdir, '_densepretrain.pth')
else:
    model_path = os.path.join(args.logdir, args.model_name)

model_basic.load_state_dict(torch.load(model_path))
ranks_up = model_basic.get_ranks()

weights_list = model_basic.raw_weights(ranks_up)

models = [CLabcv2(ranks_up), CLabcv2(ranks_up), CLabcv2(ranks_up)]
# model = CLabcv2(ranks_up)
# model.set_weights(weights_list)
for model in models:
    model.set_weights(weights_list)
    model.load_state_dict(model_basic.state_dict(), strict=False)

model, modelz, modely = models

modelu = CLabcv2(ranks_up)
modelu.empty_all()
modelv = CLabcv2(ranks_up)
modelv.empty_all()

if args.cuda:
    # model = torch.nn.DataParallel(model, device_ids=range(args.ngpu))
    # modelz = torch.nn.DataParallel(modelz, device_ids=range(args.ngpu))
    # modely = torch.nn.DataParallel(modely, device_ids=range(args.ngpu))
    # modelu = torch.nn.DataParallel(modelu, device_ids=range(args.ngpu))
    # modelv = torch.nn.DataParallel(modelv, device_ids=range(args.ngpu))

    model.cuda()
    modelz.cuda()
    modely.cuda()
    modelu.cuda()
    modelv.cuda()

algo = {'fgsm': fgsm_gt, 'bim': ifgsm_gt, 'pgd': pgd_gt}
# attack_algo = algo[args.attack_algo]

if args.attack_algo is not None:
    attack_algo = algo[args.attack_algo]
else:
    attack_algo = None

if args.defend_algo is not None:
    defend_algo = algo[args.defend_algo]
else:
    defend_algo = None

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
else:
    quantize_algo = None

quantize_name = "None" if args.quantize_algo is None else args.quantize_algo

optimizer = optim.Adam(model.parameters(), lr=args.lr)

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

# crossloss = nn.CrossEntropyLoss
print('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()

def print_dict(argslist, name):
    print("================={}==================".format(name))
    for k, v in argslist.items():
        print('{}: {}'.format(k, v))
    print("========================================")

try:
    # ready to go
    model_test(model, 0, test_loader, 
            atk_algo=attack_algo, atk_eps=args.attack_eps, 
            iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)
    
    layers = layers_nnz(model, param_name=['weightA', "weightB", "weightC"])[1]
    print_dict(layers, name="MODEL SIZE")

    if args.prune_algo == "baseline":
        prune_idx, Weight_shapes = prune_algo(model, args.prune_ratio, param_name=['weightA', "weightB", "weightC"])
        prune_lambda = lambda m: idxproj(m, z_idx=prune_idx, W_shapes=Weight_shapes)
    elif args.prune_algo == "l0proj":
        normalized = 0<=args.prune_ratio<=1.0
        prune_lambda = lambda m: prune_algo(m, args.prune_ratio, param_name=['weightA', "weightB", "weightC"], normalized=normalized)
    else:
        prune_lambda = None

    if args.quantize_algo == "kmeans":
        quantize_lambda = lambda m: quantize_algo(m, bit_depth=args.quantize_bits)
    else:
        quantize_lambda = None

    for epoch in range(args.epochs):

        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        
        model_train_admm([model, modelz, modely, modelu, modelv], epoch, train_loader, optimizer,
            dfn_algo=defend_algo, dfn_eps=args.defend_eps, 
            log_interval=args.log_interval, iscuda=args.cuda, 
            adv_iter=16, criterion=F.cross_entropy, prune_tk=prune_lambda, 
            quantize_tk=quantize_lambda, admm_interval=args.proj_interval)

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

            layers = layers_nnz(modelz, param_name=['weightA', "weightB", "weightC"])[1]
            print_dict(layers, name="MODEL SIZE")
            if acc > best_acc:
                if args.save_model_name is None:
                    if args.defend_algo is not None:
                        misc.model_snapshot(model, os.path.join(args.savedir, args.defend_algo+'_densepretrain.pth'))
                        misc.model_snapshot(modelz, os.path.join(args.savedir, "sparse_" + args.defend_algo + "_densepretrain.pth"))
                        misc.model_snapshot(modely, os.path.join(args.savedir, "quant_" + args.defend_algo + "_densepretrain.pth"))
                    else:
                        misc.model_snapshot(model, os.path.join(args.savedir, '_densepretrain.pth'))
                        misc.model_snapshot(modelz, os.path.join(args.savedir, 'sparse_densepretrain_sparse.pth'))
                        misc.model_snapshot(modely, os.path.join(args.savedir, 'quant_densepretrain_quant.pth'))
                else:
                    misc.model_snapshot(model, os.path.join(args.savedir, args.save_model_name))
                    misc.model_snapshot(modelz, os.path.join(args.savedir, "sparse_" + args.save_model_name))
                    misc.model_snapshot(modely, os.path.join(args.savedir, "quant_" + args.save_model_name))
                best_acc = acc

except Exception as e:
    import traceback
    traceback.print_exc()

finally:
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))
        