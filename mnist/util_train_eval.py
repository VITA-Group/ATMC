import argparse
import os
import time

from utee import misc
import torch
import torch.nn as nn
from util import cross_entropy, wrm, fgm, ifgm, pgm
import torch.optim as optim
import numpy as np
import scipy.misc as smp
import torch.nn.functional as F
from scipy import optimize
from cycler import cycler

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

import matplotlib.pyplot as plt
from PIL import Image

import dataset
import model
from pruning_tools import MyCaffeLeNet

def model_train(model, loss_fn, data_loader, optimizer, epoch, log_interval, 
    dfn_gamma=None, dfn_eps=None, defense=None, prune_tk=None, cuda_flag=False, 
    reg_defense=None, adv_iter=16, dfn_randinit=False):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
            indx_target = target.clone()
            target_ = torch.unsqueeze(target, 1)

            one_hot = torch.FloatTensor(target.size(0), 10).zero_()
            one_hot.scatter_(1, target_, 1)
            
            if cuda_flag:
                data, target = data.cuda(), one_hot.cuda()
            else:
                target = one_hot

            optimizer.zero_grad()
            if defense is not None:
                data_adv = defense(data, None, 
                    y=target, 
                    eps=dfn_eps, 
                    model=model, 
                    label_smoothing=0.1, 
                    gamma=dfn_gamma,
                    randinit=dfn_randinit).data
                output_adv = model(data_adv)
                output = model(data)
                L1 = loss_fn(output_adv, target, label_smoothing=0.1)
                L2 = loss_fn(output, target, label_smoothing=0.1)

                loss = 0.5 * L1 + 0.5 * L2
            else:
                data_adv = data
                output = model(data)
                loss = loss_fn(output, target, label_smoothing=0.1)

            # if reg_defense is not None:
            #     regular = reg_defense(data, None, y=target, eps=dfn_eps, model=model, label_smoothing=0.1).data
            # else:
            #     regular = 0.0
            # output = model(data_adv)
            # loss = loss_fn(output, target, label_smoothing=0.1) #+ regular
            loss.backward()
            optimizer.step()
            if prune_tk is not None:
                prune_tk(model)

            if batch_idx % log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]
                correct = pred.cpu().eq(indx_target).sum()
                acc = float(correct) / len(data)
                print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    loss.data.item(), acc, optimizer.param_groups[0]['lr']))

    return model

def threshplus(x):
    y = x.copy()
    y[y<0]=0
    return y

def loss_map_chi_factory(loss_values, eps):
    return lambda x: np.sqrt(2)*(1.0/eps-1.0)*np.sqrt(np.mean(threshplus(loss_values-x)**2.0)) + x

def loss_map_chi_factory_torch(loss_values, eps):
    return lambda x: np.sqrt(2) * (1.0/ eps - 1.0) * torch.sqrt(torch.mean(F.relu(loss_values - x).pow(2))) + x

def model_search_train(model, loss_fn, data_loader, optimizer, epoch, log_interval, 
    dfn_gamma=None, dfn_eps=None, defense=None, prune_tk=None, cuda_flag=False, 
    reg_defense=None, adv_iter=16, dfn_randinit=False):
    model.train()
    model_eta = 0
    for batch_idx, (data, target) in enumerate(data_loader):
            indx_target = target.clone()
            target_ = torch.unsqueeze(target, 1)

            one_hot = torch.FloatTensor(target.size(0), 10).zero_()
            one_hot.scatter_(1, target_, 1)
            
            if cuda_flag:
                data, target = data.cuda(), one_hot.cuda()
            else:
                target = one_hot

            optimizer.zero_grad()
            # if defense is not None:
            #     data_adv = defense(data, None, y=target, eps=dfn_eps, model=model, label_smoothing=0.1).data
            # else:
            #     data_adv = data

            output = model(data)
            losses = loss_fn(output, target, label_smoothing=0.1, size_average=False)
            loss = torch.sqrt(
                torch.mean(
                    F.relu(
                        losses - model_eta
                        ).pow(2)
                    )
                )
            loss.backward()
            optimizer.step()
            
            epsin = 0.2
            with torch.no_grad():
                chi_loss = loss_map_chi_factory(losses.data.cpu().numpy(), epsin)
                # print(torch.min(losses).data.cpu().numpy() - 1000.0)
                # print(torch.max(losses).data.cpu().numpy())
                model_eta = optimize.fminbound(chi_loss, float(torch.min(losses).data.cpu().numpy() - 1000.0), float(torch.max(losses).data.cpu().numpy()))
                print("model_eta", model_eta)

            if prune_tk is not None:
                prune_tk(model)

            if batch_idx % log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]
                correct = pred.cpu().eq(indx_target).sum()
                acc = float(correct) / len(data)
                print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    loss.data.item(), acc, optimizer.param_groups[0]['lr']))

    return model


def model_eval(model, data_loader, attack_algo, loss_fn, 
    attack_gamma=None, cuda_flag=False, attack_eps=None, adv_iter=16, attack_randinit=False):
    model.eval()
    test_loss, adv_loss, correct, correct_adv, nb_data, adv_l2dist, adv_linfdist = \
        0, 0, 0, 0, 0, 0.0, 0.0

    for data, target in data_loader:
        indx_target = target.clone()
        data_length = data.shape[0]
        nb_data += data_length

        target_ = torch.unsqueeze(target, 1)
        one_hot = torch.FloatTensor(target.size()[0], 10).zero_()
        one_hot.scatter_(1, target_, 1)
        
        if cuda_flag:
            data, target = data.cuda(), one_hot.cuda()
        else:
            target = one_hot
        with torch.no_grad():
            output = model(data)
        
        # img_clamp = lambda x: torch.clamp(x, min=0, max=1)
        # temp = (img_clamp(data).data.cpu().numpy()[0, :, :, :] * 255).astype(np.uint8)
        # print(temp.shape)
        # img_s = smp.toimage(np.squeeze(np.transpose(temp, (1, 2, 0))))
        # img_s.show()

        data_adv = attack_algo(data, output, eps=attack_eps, y=target, model=model, label_smoothing=0.0, randinit=attack_algo).data
        # temp = (img_clamp(data_adv).data.cpu().numpy()[0, :, :, :] * 255).astype(np.uint8)
        # # print(temp.shape)
        # img_d = smp.toimage(np.squeeze(np.transpose(temp, (1, 2, 0))))
        # img_d.show()
        adv_l2dist += torch.norm((data-data_adv).view(data.size(0), -1), p=2, dim=-1).sum().item()
        adv_linfdist += torch.max((data-data_adv).view(data.size(0), -1).abs(), dim=-1)[0].sum().item()
        with torch.no_grad():
            output_adv = model(data_adv)
        adv_loss += loss_fn(output_adv, target, 0.0, size_average=False).data.item()
        pred_adv = output_adv.data.max(1)[1]
        correct_adv += pred_adv.cpu().eq(indx_target).sum()
        test_loss += loss_fn(output, target, 0.0, size_average=False).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()

    test_loss = test_loss / nb_data  # average over number of mini-batch
    acc = float(100. * correct) / len(data_loader.dataset)
    print('\tTest set: Average loss: {:.4f} , Accuracy: {}/{}({:.0f}%)'.format(
        test_loss, correct, len(data_loader.dataset), acc))

    adv_loss = adv_loss / nb_data
    acc_adv = float(100. * correct_adv) / len(data_loader.dataset)
    print('\tAdv set: Average loss: {:.4f} , Accuracy : {}/{}({:.0f}%)'.format(
                adv_loss, correct_adv, len(data_loader.dataset), acc_adv
    ))
    adv_l2dist /= len(data_loader.dataset)
    adv_linfdist /= len(data_loader.dataset)
    print('\tAdv dist: L2: {:.8f} , Linf: {:.8f}'.format(adv_l2dist, adv_linfdist))

    return model, acc_adv

def model_test(model, data_loader, attack_algo, loss_fn, output_file,
    attack_gamma=None, cuda_flag=False, attack_eps=None, adv_iter=16, attack_randinit=False, test_round=3):
    model.eval()
    
    pacc, pacc_adv = [], []
    for _ in range(test_round):
        test_loss, adv_loss, correct, correct_adv, nb_data, adv_l2dist, adv_linfdist = \
        0, 0, 0, 0, 0, 0.0, 0.0
        for data, target in data_loader:
            indx_target = target.clone()
            data_length = data.shape[0]
            nb_data += data_length

            target_ = torch.unsqueeze(target, 1)
            one_hot = torch.FloatTensor(target.size()[0], 10).zero_()
            one_hot.scatter_(1, target_, 1)
            
            if cuda_flag:
                data, target = data.cuda(), one_hot.cuda()
            else:
                target = one_hot
            with torch.no_grad():
                output = model(data)
            
            # img_clamp = lambda x: torch.clamp(x, min=0, max=1)
            # temp = (img_clamp(data).data.cpu().numpy()[0, :, :, :] * 255).astype(np.uint8)
            # print(temp.shape)
            # img_s = smp.toimage(np.squeeze(np.transpose(temp, (1, 2, 0))))
            # img_s.show()
            # (x, preds, y=None, eps=None, model=None, steps=16, alpha=None, label_smoothing=0.0, randinit=False, **kwargs):
            data_adv = attack_algo(data, None, eps=attack_eps, y=target, model=model, label_smoothing=0.0, randinit=attack_randinit).data
            # data_adv = attack_algo(data, None, 
                    # loss_fn=loss_fn, 
                    # eps=attack_eps, 
                    # y=target, 
                    # model=model, 
                    # steps=adv_iter, 
                    # gamma=attack_gamma,
                    # randinit=attack_randinit).data
            # temp = (img_clamp(data_adv).data.cpu().numpy()[0, :, :, :] * 255).astype(np.uint8)
            # # print(temp.shape)
            # img_d = smp.toimage(np.squeeze(np.transpose(temp, (1, 2, 0))))
            # img_d.show()
            adv_l2dist += torch.norm((data-data_adv).view(data.size(0), -1), p=2, dim=-1).sum().item()
            adv_linfdist += torch.max((data-data_adv).view(data.size(0), -1).abs(), dim=-1)[0].sum().item()
            with torch.no_grad():
                output_adv = model(data_adv)
            adv_loss += loss_fn(output_adv, target, 0.0, size_average=False).data.item()
            pred_adv = output_adv.data.max(1)[1]
            correct_adv += pred_adv.cpu().eq(indx_target).sum()
            test_loss += loss_fn(output, target, 0.0, size_average=False).data.item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.cpu().eq(indx_target).sum()

        test_loss = test_loss / nb_data  # average over number of mini-batch
        acc = float(100. * correct) / len(data_loader.dataset)
        pacc.append( acc / 100. )
        print('\tTest set: Average loss: {:.4f} , Accuracy: {}/{}({:.0f}%)'.format(
            test_loss, correct, len(data_loader.dataset), acc))

        adv_loss = adv_loss / nb_data
        acc_adv = float(100. * correct_adv) / len(data_loader.dataset)
        pacc_adv.append( acc_adv / 100. )
        print('\tAdv set: Average loss: {:.4f} , Accuracy : {}/{}({:.0f}%)'.format(
                    adv_loss, correct_adv, len(data_loader.dataset), acc_adv
        ))
        adv_l2dist /= len(data_loader.dataset)
        adv_linfdist /= len(data_loader.dataset)
        print('\tAdv dist: L2: {:.8f} , Linf: {:.8f}'.format(adv_l2dist, adv_linfdist))
    with open(output_file, "r") as stream:
        lines = stream.readlines()
    with open(output_file, "w+") as stream:
        pacc_mean, pacc_var = np.mean(pacc), np.std(pacc)
        pacc_adv_mean, pacc_adv_var = np.mean(pacc_adv), np.std(pacc_adv)
        # lines[1] = lines[1].strip() + " {}\n".format(pacc/(test_round * 100.)) if lines[1] != "\n" else "{}\n".format(pacc/(test_round * 100.))
        # lines[2] = lines[2].strip() + " {}\n".format(pacc_adv/(test_round * 100.)) if lines[2] != "\n" else "{}\n".format(pacc_adv/(test_round * 100.))
        lines[1] = lines[1].strip() + " {}\n".format(pacc_mean) if lines[1] != "\n" else "{}\n".format(pacc_mean)
        lines[2] = lines[2].strip() + " {}\n".format(pacc_var) if lines[2] != "\n" else "{}\n".format(pacc_var)
        lines[3] = lines[3].strip() + " {}\n".format(pacc_adv_mean) if lines[3] != "\n" else "{}\n".format(pacc_adv_mean)
        lines[4] = lines[4].strip() + " {}\n".format(pacc_adv_var) if lines[4] != "\n" else "{}\n".format(pacc_adv_var)
        stream.writelines(lines)

    return model, acc_adv

import pickle as pkl

def model_show(model, data_loader, attack_algo, loss_fn,
    attack_gamma=None, cuda_flag=False, attack_eps=None, adv_iter=16, attack_randinit=False, test_round=3):
    
    dense_model, nonadv_sp_model, adv_sp_model, lowrank_model, amc_model = model
    dense_model.eval()
    nonadv_sp_model.eval()
    adv_sp_model.eval()
    lowrank_model.eval()
    amc_model.eval()
    pacc, pacc_adv = [], []
    for _ in range(test_round):
        test_loss, adv_loss, correct, correct_adv, nb_data, adv_l2dist, adv_linfdist = \
        0, 0, 0, 0, 0, 0.0, 0.0
        for data, target in data_loader:
            indx_target = target.clone()
            data_length = data.shape[0]
            nb_data += data_length

            target_ = torch.unsqueeze(target, 1)
            one_hot = torch.FloatTensor(target.size()[0], 10).zero_()
            one_hot.scatter_(1, target_, 1)
            
            if cuda_flag:
                data, target = data.cuda(), one_hot.cuda()
            else:
                target = one_hot
            
            data_dense_model = attack_algo(
                data, None, eps=attack_eps, y=target, model=dense_model, label_smoothing=0.0, randinit=attack_randinit
            )
            data_nonadv_sp_model = attack_algo(
                data, None, eps=attack_eps, y=target, model=nonadv_sp_model, label_smoothing=0.0, randinit=attack_randinit
            )
            data_adv_sp_model = attack_algo(
                data, None, eps=attack_eps, y=target, model=adv_sp_model, label_smoothing=0.0, randinit=attack_randinit
            )
            data_lowrank_model = attack_algo(
                data, None, eps=attack_eps, y=target, model=lowrank_model, label_smoothing=0.0, randinit=attack_randinit
            )
            data_amc_model = attack_algo(
                data, None, eps=attack_eps, y=target, model=amc_model, label_smoothing=0.0, randinit=attack_randinit
            )
            with torch.no_grad():
                output_dense = dense_model(data_dense_model)
                output_nonadv_sp = nonadv_sp_model(data_nonadv_sp_model)
                output_adv_sp = adv_sp_model(data_adv_sp_model)
                output_lowrank = lowrank_model(data_lowrank_model)
                output_amc = amc_model(data_amc_model)
            
            pred_dense = output_dense.data.max(1)[1].cpu()
            pred_nonadv_sp = output_nonadv_sp.max(1)[1].cpu()
            pred_adv_sp = output_adv_sp.max(1)[1].cpu()
            pred_lowrank = output_lowrank.max(1)[1].cpu()
            pred_amc = output_amc.max(1)[1].cpu()

            iter_data = len(data)
            for i in range(iter_data):
                base_tar = indx_target[i]
                flag = (pred_amc[i] == base_tar ) and pred_dense[i] == base_tar and pred_nonadv_sp[i] != base_tar and pred_adv_sp[i] != base_tar and pred_lowrank[i] != base_tar
                if flag:
                    # plt.figure(1)
                    # plt.axis('off')
                    f, axarr = plt.subplots(2,3)
                    # plt.subplot(231)
                    print(data[i].shape)
                    axarr[0,0].imshow(data[i,0], cmap='gray')
                    axarr[0,0].set_title("Original: {}".format(base_tar))
                    axarr[0,0].axis('off')
                    # plt.show()
                    # plt.subplot(232)
                    axarr[0,1].imshow(data_dense_model.data.clamp(min=0, max=1).cpu().numpy()[i,0], cmap='gray')
                    axarr[0,1].set_title("DA: {}".format(pred_dense[i]))
                    axarr[0,1].axis('off')
                    # plt.show()
                    # plt.subplot(233)
                    axarr[0,2].imshow(data_nonadv_sp_model.clamp(min=0, max=1).data.cpu().numpy()[i,0], cmap='gray')
                    axarr[0,2].set_title("NAP: {}".format(pred_nonadv_sp[i]))
                    axarr[0,2].axis('off')
                    # plt.subplot(234)
                    axarr[1,0].imshow(data_adv_sp_model.clamp(min=0, max=1).data.cpu().numpy()[i,0], cmap='gray')
                    axarr[1,0].set_title("AP: {}".format(pred_adv_sp[i]))
                    axarr[1,0].axis('off')
                    # plt.subplot(235)
                    axarr[1,1].imshow(data_lowrank_model.clamp(min=0, max=1).data.cpu().numpy()[i,0], cmap='gray')
                    axarr[1,1].set_title("ALR: {}".format(pred_lowrank[i]))
                    axarr[1,1].axis('off')
                    # plt.subplot(236)
                    axarr[1,2].imshow(data_amc_model.clamp(min=0, max=1).data.cpu().numpy()[i,0], cmap='gray')
                    axarr[1,2].set_title("ATMC: {}".format(pred_amc[i]))
                    axarr[1,2].axis('off')
                    
                    # plt.show()
                    f.savefig("mnist/mnist-example-{}".format(base_tar), bbox_inches='tight')
                    with open('mnist/mnist-example-{}-figure.pickle'.format(base_tar), 'wb') as stream:
                        pkl.dump(f, stream)
                    plt.show()
                    # return None

    return None


def pick_one_pic(data):
    return data.clamp(min=0, max=1).data.cpu().numpy()[0,0]
    # unorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    # temp = unorm(data)
    # # temp =  torch.clamp(data.data + 1 /2, min=0, max=1)
    # temp = temp.data.cpu().numpy()[0, :, :, :]
    
    # temp = np.transpose(temp, (1, 2, 0))
    # return temp


def model_show_increasing(model, data_loader, attack_algo, loss_fn,
    attack_gamma=None, cuda_flag=False, attack_eps=None, adv_iter=16, attack_randinit=False, test_round=3):
    
    nonadv_dense_model, dense_model, nonadv_sp_model, adv_sp_model, lowrank_model, amc_model = model
    nonadv_dense_model.eval()
    dense_model.eval()
    nonadv_sp_model.eval()
    adv_sp_model.eval()
    lowrank_model.eval()
    amc_model.eval()
    pacc, pacc_adv = [], []
    for _ in range(test_round):
        test_loss, adv_loss, correct, correct_adv, nb_data, adv_l2dist, adv_linfdist = \
        0, 0, 0, 0, 0, 0.0, 0.0
        for data, target in data_loader:
            indx_target = target.clone()
            data_length = data.shape[0]
            nb_data += data_length

            target_ = torch.unsqueeze(target, 1)
            one_hot = torch.FloatTensor(target.size()[0], 10).zero_()
            one_hot.scatter_(1, target_, 1)
            
            if cuda_flag:
                data, target = data.cuda(), one_hot.cuda()
            else:
                target = one_hot
            
            def find_the_error(model, errors, da, bar):
                for error in errors:
                    data_adv = attack_algo(
                                da, None, eps=error, y=target, model=model, label_smoothing=0.0, randinit=attack_randinit
                            )
                    with torch.no_grad():
                        output_adv = model(data_adv)
                    pred = output_adv.data.max(1)[1].cpu()
                    if pred != bar:
                        return data_adv, error, pred.item()
                return data_adv, error+1, bar.item()

            base_tar = indx_target[0]
            errors = [ i/20.0 * 0.3 for i in range(1, 41, 2)]
            print(errors)
            pick = []
            pick_eps = []
            pick_pred = []
            for m in [dense_model, nonadv_dense_model, nonadv_sp_model, adv_sp_model, lowrank_model, amc_model]:
                data_adv, eps, pred = find_the_error(model=m, errors=errors, da=data, bar=base_tar)
                if pred == base_tar:
                    break
                else:
                    pick.append(data_adv)
                    pick_eps.append(eps)
                    pick_pred.append(pred)
            if len(pick_eps) == 0 or len(pick) == 0 or len(pick_eps) != len(model):
                continue
            if not (pick_eps[0] > pick_eps[1] and (pick_eps[0] > pick_eps[-1] >= pick_eps[-2] > pick_eps[2] \
                or pick_eps[0] > pick_eps[-1] >= pick_eps[-3] > pick_eps[2])
            ):
                continue 
            print(len(pick))
            # plt.figure(1)
            # plt.axis('off')
            f, axarr = plt.subplots(1,7, figsize=(12, 2))
            # plt.subplot(231)
            print(data[0].shape)
            name = [str(i) for i in range(10)]
            axarr[0].imshow(data[0,0], cmap='gray')
            axarr[0].set_title("Original: {}".format(name[base_tar]))
            axarr[0].axis('off')
            # plt.show()
            # plt.subplot(232)
            
            pick_pred = [name[term] for term in pick_pred]
            # plt.show()
            # plt.subplot(233)
            idx_picture = 1
            axarr[idx_picture].imshow(pick_one_pic(pick[idx_picture]), cmap='gray')
            axarr[idx_picture].set_title("Attack:{} ({:.2f})".format(pick_pred[idx_picture], pick_eps[idx_picture]))
            axarr[idx_picture].axis('off')

            idx_picture += 1
            axarr[idx_picture].imshow(pick_one_pic(pick[idx_picture]), cmap='gray')
            axarr[idx_picture].set_title("NAP:{} ({:.2f})".format(pick_pred[idx_picture], pick_eps[idx_picture]))
            axarr[idx_picture].axis('off')
            # plt.subplot(234)
            idx_picture += 1
            axarr[idx_picture].imshow(pick_one_pic(pick[idx_picture]), cmap='gray')
            axarr[idx_picture].set_title("AP:{} ({:.2f})".format(pick_pred[idx_picture], pick_eps[idx_picture]))
            axarr[idx_picture].axis('off')
            # plt.subplot(235)
            idx_picture += 1
            axarr[idx_picture].imshow(pick_one_pic(pick[idx_picture]), cmap='gray')
            axarr[idx_picture].set_title("ALR:{} ({:.2f})".format(pick_pred[idx_picture], pick_eps[idx_picture]))
            axarr[idx_picture].axis('off')
            # plt.subplot(236)
            idx_picture += 1
            axarr[idx_picture].imshow(pick_one_pic(pick[idx_picture]), cmap='gray')
            axarr[idx_picture].set_title("ATMC:{} ({:.2f})".format(pick_pred[idx_picture], pick_eps[idx_picture]))
            axarr[idx_picture].axis('off')

            idx_picture += 1
            axarr[idx_picture].imshow(pick_one_pic(pick[0]), cmap='gray')
            axarr[idx_picture].set_title("DA:{} ({:.2f})".format(pick_pred[0], pick_eps[0]))
            axarr[idx_picture].axis('off')
            plt.tight_layout()
            
            f.savefig("mnist/adv_increase_results/mnist-increase-example-{}.svg".format(base_tar), bbox_inches='tight')
            # forig
            with open('mnist/adv_increase_results/mnist-increase-example-{}-figure.pickle'.format(base_tar), 'wb') as stream:
                pkl.dump([[data] + pick, pick_pred, pick_eps], stream)
            # plt.show()
                    # return None

    return None