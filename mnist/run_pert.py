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
from util_trts import model_train, model_test, model_uai_train, model_uai_test

# np.set_printoptions(threshold=np.nan)

import dataset
from caffelenet.caffelenet_dense import CaffeLeNet

parser = argparse.ArgumentParser(description="Pytorch MNIST bold")
parser.add_argument('--wd', type=float, default=1e-4, help="weight decay factor")
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--gpu', default="0", help="index of GPUs to use")
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--seed', type=int, default=117, help="random seed (default: 117)")
parser.add_argument('--log_interval', type=int, default=20, help="how many batches to wait before logging training status")
parser.add_argument('--test_interval', type=int, default=1, help="how many epochs to wait before another test")
parser.add_argument('--logdir', default='log/default', help='folder to save the log')
parser.add_argument('--data_root', default='/media/hdd/mnist/', help='folder to save the data')
parser.add_argument('--decreasing_lr', default='80,120', help="decreasing strategy")
parser.add_argument('--attack_algo', default='fgsm', help='adversarial algo for attack')
parser.add_argument('--attack_eps', type=float, default=None, help='perturbation radius for attack phase')
parser.add_argument('--defend_algo', default=None, help='adversarial algo for defense')
parser.add_argument('--defend_eps', type=float, default=None, help='perturbation radius for defend phase')

parser.add_argument('--save_model_name', default=None, help="save model name after training")

args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
misc.logger.init(args.logdir, 'train_log')
print = misc.logger.info

# select gpu
args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)

# logger
misc.ensure_dir(args.logdir)
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader, test_loader = dataset.get(batch_size=args.batch_size, data_root=args.data_root, num_workers=1)

# model
# model = model.mnist(input_dims=784, n_hiddens=[256, 256], n_class=10)

# model_feature = model.CNN(n_class=10)
# model_pred = model.UAI_PRED(n_class=10)
# model_distortion = model.UAI_DISTORTION()

model = CaffeLeNet()

model = torch.nn.DataParallel(model, device_ids=range(args.ngpu))
# model_feature = torch.nn.DataParallel(model_feature, device_ids=range(args.ngpu))
if args.cuda:
    model.cuda()

algo = {'fgsm': fgsm_gt, 'bim': ifgsm_gt, 'pgd': pgd_gt, 'grad': grad_gt}
attack_algo = algo[args.attack_algo]

if args.attack_algo is not None:
    attack_algo = algo[args.attack_algo]
else:
    attack_algo = None

if args.defend_algo is not None:
    defend_algo = algo[args.defend_algo]
else:
    defend_algo = None

# exit(0)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optim_pred = optim.Adam(list(model_feature.parameters()) + list(model_pred.parameters()), lr=args.lr)
# optim_distortion = optim.Adam(model_distortion.parameters())

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

# crossloss = nn.CrossEntropyLoss
print('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()

try:
    # ready to go
    for epoch in range(args.epochs):

        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        
        model_train(model, epoch, train_loader, optimizer,
            dfn_algo=defend_algo, dfn_eps=args.defend_eps, 
            log_interval=args.log_interval, iscuda=args.cuda, 
            adv_iter=6, criterion=F.cross_entropy)

        # model_uai_train(model_fea=model_feature, model_pred=model_pred, model_distortion=model_distortion, epoch=epoch, data_loader=train_loader, optims=[optim_pred, optim_distortion], dfn_algo=defend_algo, dfn_eps=args.defend_eps, log_interval=args.log_interval, iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time

        print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))
        misc.model_snapshot(model, os.path.join(args.logdir, 'latest.pth'))

        if epoch % args.test_interval == 0:

            acc = model_test(model, epoch, test_loader, 
                    atk_algo=attack_algo, atk_eps=args.attack_eps, 
                    iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)

            # acc = model_uai_test(model_feature, model_pred, epoch, test_loader, attack_algo, args.attack_eps, iscuda=args.cuda, adv_iter=16, criterion=F.cross_entropy)

            if acc > best_acc:
                # new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                if args.save_model_name is None:
                    if args.defend_algo is not None:
                        misc.model_snapshot(model, os.path.join(args.logdir, args.defend_algo+'_densepretrain.pth'))
                    else:
                        misc.model_snapshot(model, os.path.join(args.logdir, '_densepretrain.pth'))
                else:
                    misc.model_snapshot(model, os.path.join(args.logdir, args.save_model_name))
                best_acc = acc
                # old_file = new_file

except Exception as e:
    import traceback
    traceback.print_exc()

finally:
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))
        