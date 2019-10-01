import torch, dataset, os, time, argparse, pickle, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'utee'))
import numpy as np

from advertorch.attacks import CarliniWagnerL2Attack, LinfPGDAttack, MomentumIterativeAttack
from advertorch.utils import clamp

from pruning_tools import l0proj, idxproj, layers_nnz, layers_unique
import pruning_tools as pt

from resnet.resnet_dense import ResNet34 as CLdense
from resnet.resnet_abcv2 import ResNet34 as CLabcv2
from resnet.resnet_lr import ResNet34 as CLlr

mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])

max_v = np.max( list((1-mean)/std) )
min_v = np.min( list((0-mean)/std) )

# eps = 8/255 * (max_v-min_v)
eps = np.max(8/255 / std)
print('eps:', eps, 'max_v:', max_v, 'min_v:', min_v)
# eps = 0.0157 (4/255)

def model_test(model, data_loader, output_file_path, attack='mia', eps=8/255, nb_iter=3):
    model.eval()
    
    test_loss, adv_loss, correct, correct_adv, nb_data, adv_l2dist, adv_linfdist = \
    0, 0, 0, 0, 0, 0.0, 0.0

    start_time = time.time()
    for i, (data, target) in enumerate(data_loader):
        print('i:', i)

        indx_target = target.clone()
        data_length = data.shape[0]
        nb_data += data_length
        
        data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            output = model(data)
        
        # print('data max:', torch.max(data))
        # print('data min:', torch.min(data))
        if attack == 'cw':
            if i >= 5:
                break
            adversary = CarliniWagnerL2Attack(predict=model, num_classes=10, targeted=True, 
                clip_min=min_v, clip_max=max_v, max_iterations=50)
        elif attack == 'mia':
            adversary = MomentumIterativeAttack(predict=model, targeted=True, eps=eps, nb_iter=40, eps_iter=0.01*(max_v-min_v), 
                clip_min=min_v, clip_max=max_v )
        elif attack == 'pgd':
            adversary = LinfPGDAttack(predict=model, targeted=True, eps=eps, nb_iter=nb_iter, eps_iter=eps*1.25/nb_iter,
                clip_min=min_v, clip_max=max_v )
        else:
            raise 'unimplemented error'
        pred = model(data) # torch.Size([128, 10])
        print('pred:', type(pred), pred.shape)
        print('target:', type(target), target.shape, target[0:20])
        # pred_argmax = torch.argmax(pred, dim=1)
        # print('pred_argmax:', type(pred_argmax), pred_argmax.shape, pred_argmax[0:10])
        # for i in range(list(pred.shape)[0]):
        #     pred[i,pred_argmax[i]] = -1
        for i in range(list(pred.shape)[0]):
            pred[i,target[i]] = -1
        # target_adv = torch.argmax(pred, dim=1)
        target_adv = (target + 5) % 10
        print('target_adv:', type(target_adv), target_adv.shape, target_adv[0:20])
        data_adv = adversary.perturb(data, target_adv)

        print('data_adv max:', torch.max(data_adv))
        print('data_adv min:', torch.min(data_adv))
        print('linf:', torch.max(torch.abs(data_adv-data)) )

        adv_l2dist += torch.norm((data-data_adv).view(data.size(0), -1), p=2, dim=-1).sum().item()
        adv_linfdist += torch.max((data-data_adv).view(data.size(0), -1).abs(), dim=-1)[0].sum().item()

        with torch.no_grad():
            output_adv = model(data_adv)

        pred_adv = output_adv.data.max(1)[1]
        correct_adv += pred_adv.cpu().eq(indx_target).sum()
        
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()
        
        time_consume = time.time() - start_time
        print('time_consume:', time_consume)

        acc = float(100. * correct) / nb_data
        print('\tTest set: Accuracy: {}/{}({:.2f}%)'.format(
            correct, nb_data, acc))

        acc_adv = float(100. * correct_adv) / nb_data
        print('\tAdv set: Accuracy : {}/{}({:.2f}%)'.format(
            correct_adv, nb_data, acc_adv
        ))

    adv_l2dist /= nb_data
    adv_linfdist /= nb_data
    print('\tAdv dist: L2: {:.8f} , Linf: {:.8f}'.format(adv_l2dist, adv_linfdist))

    with open(output_file_path, "a+") as output_file:
        output_file.write(args.model_name + '\n')
        info_string = 'attack: %s:\n acc: %.2f, acc_adv: %.2f, adv_l2dist: %.2f, adv_linfdist: %.2f, time_consume: %.2f' % (
            attack, acc, acc_adv, adv_l2dist, adv_linfdist, time_consume) 
        output_file.write(info_string)

    return acc, acc_adv


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Pytorch MNIST bold")
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--gpu', default="0", help="index of GPUs to use")

    parser.add_argument("--abc_special", action="store_true")
    parser.add_argument("--abc_initialize", action="store_true")
    parser.add_argument("--lr_special", action="store_true")
    parser.add_argument("--lr_initialize", action="store_true")

    parser.add_argument('--prune_algo', default=None, help="pruning projection method")
    parser.add_argument('--prune_ratio', type=float, default=0.1, help='sparse ratio or energy budget')

    parser.add_argument('--loaddir', default='log/default', help='folder to load the log')
    parser.add_argument('--savedir', default=None, help="folder to save the log")
    parser.add_argument('--data_root', default='/media/hdd/mnist/', help='folder to save the data')
    parser.add_argument('--model_name', default=None, help="file name of pre-train model")
    parser.add_argument('--prefix_name', default="", help="save model name after training")

    parser.add_argument("-e", "--exp_logger", default=None, help="exp results stored to")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    # model:
    model_base = CLdense()
    weight_name = ["weight"] if not args.abc_special else ["weightA", "weightB", "weightC"]
    weight_name = ["weightA", "weightB"] if args.lr_special else weight_name

    model_path = os.path.join(args.loaddir, args.prefix_name + args.model_name)

    if args.abc_special:
        ranks_up = model_base.get_ranks()
        model = CLabcv2(ranks_up)
    elif args.lr_special:        
        with open(os.path.join(args.loaddir, args.model_name[0:-4] + ".npy"), "rb") as filestream:
            ranks_up = pickle.load(filestream)
        model = CLlr(ranks_up)
    else:
        model = model_base
    model.load_state_dict(torch.load(model_path))

    model.cuda()

    #
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

    if args.prune_algo == "baseline":
        prune_idx, Weight_shapes = prune_algo(model, args.prune_ratio, param_name=weight_name)
        # prune_idx, Weight_shapes = prune_algo(model, sparse_factor, normalized=False, param_name=weight_name)
        prune_lambda = lambda m: idxproj(m, z_idx=prune_idx, W_shapes=Weight_shapes)
    elif args.prune_algo == "l0proj":
        # prune_lambda = lambda m: prune_algo(m, sparse_factor, normalized=False, param_name=weight_name)
        prune_lambda = lambda m: prune_algo(m, args.prune_ratio, normalized=True, param_name=weight_name)
    elif args.prune_algo == 'low_rank':
        prune_lambda = None
    else:
        prune_lambda = None

    prune_lambda(model)

    # data loader:
    _, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)

    # run testing:
    output_file_path = os.path.join(args.loaddir, args.exp_logger)
    model_test(model=model, data_loader=test_loader, output_file_path=output_file_path, eps=eps)