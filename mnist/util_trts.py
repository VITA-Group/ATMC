import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from utee import misc
import time

def model_distance(model0, model1, weight_name):
    if isinstance(model0, nn.DataParallel):
        model0 = model0.module
        model1 = model1.module
    model1_dict = dict(model1.named_parameters())
    distance = 0
    distance_layers = []
    relative_layers = []
    for n, w in model0.named_parameters():
        if n.split(".")[-1] in weight_name:
            cur = ((w - model1_dict[n]) ** 2).sum().item()
            distance += cur
            distance_layers.append(cur)
            if (w**2).sum().item() > 0:
                relative_layers.append(cur / (w**2).sum().item())
            else:
                relative_layers.append(cur)
    return distance, distance_layers, relative_layers

def acc_call(output, target, type="vanilla"):
    if type is "vanilla":
        pred = output.data.max(1)[1]
        correct = pred.cpu().eq(target).sum().item()
        acc = correct * 1. 
        return acc
    if type is "ensemble":
        with torch.no_grad():
            pred = []
            for op in output:
                op = F.softmax(op, dim=1)
                pred.append(op)
            pred = torch.fmod(torch.cat(pred, dim=1).data.max(1)[1], 10)
            correct = pred.cpu().eq(target).sum().item()
            acc = correct * 1.
        return acc
    return 0.


def model_train_proj_prune_admm_quant(models, epoch, data_loader, optimizer, \
    dfn_algo, dfn_eps, log_interval, iscuda=False, nnzfix=False, adv_iter=16, \
    criterion=F.cross_entropy, modelType="vanilla", prune_tk=None, \
    quantize_tk=None, rho=0.01, admm_interval=5, proj_interval=1, \
    weight_bits=None, layernnz=None, param_list=None):
    model, model_dual, model_u = models
    model.train()
    loss_ave = 0.
    acc_ave = 0.
    nb_data = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        nb_data += len(data)
        indx_target = target.clone()
        if iscuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if dfn_algo is None:
            data_adv = data
            output = model(data_adv)
            loss = criterion(output, target)
        else:
            data_adv = dfn_algo(x=data, y=target, criterion=criterion, rho=dfn_eps, model=model, steps=adv_iter, iscuda=iscuda)
            output_adv = model(data_adv)
            output = model(data)
            loss = 0.5 * criterion(output, target) + 0.5 * criterion(output_adv, target)

        if quantize_tk is not None:
            if isinstance(model, nn.DataParallel):
                loss += rho / 2. * model.module.admm_regularizer(model_u.module, model_dual.module)
            else:
                loss += rho / 2. * model.admm_regularizer(model_u, model_dual)
        loss.backward()
        optimizer.step()
        loss_ave += loss.item()
        if batch_idx % proj_interval == 0:
            with torch.no_grad():
                if prune_tk is not None:
                    if weight_bits is not None:
                        if nnzfix:
                            prune_tk(model)
                        else:
                            prune_tk(list(param_list(model)), weight_bits)
                        nnz = layernnz(model)
                    else:
                        prune_tk(model)
                if quantize_tk is not None:
                    if batch_idx % admm_interval == 0 and batch_idx > 0:
                        if isinstance(model_dual, nn.DataParallel):
                            model_dual.module.duplicate_plus(model.module, model_u.module)
                            if nnzfix:
                                prune_tk(model_dual)
                            if weight_bits is not None:
                                weight_bits, clusters = quantize_tk(list(param_list(model_dual)), nnz, list(param_list(model)))
                            else:
                                quantize_tk(model_dual)
                            model_u.module.duplicate_update(model.module, model_dual.module, rho)
                        else:
                            model_dual.duplicate_plus(model, model_u)
                            if nnzfix:
                                prune_tk(model_dual)
                            if weight_bits is not None:
                                weight_bits, clusters = quantize_tk(list(param_list(model_dual)), nnz, list(param_list(model)))
                            else:
                                quantize_tk(model_dual)
                            model_u.duplicate_update(model, model_dual, rho)
        
        if batch_idx % log_interval == 0 and batch_idx > 0:
            acc = acc_call(output, indx_target, type=modelType) 
            acc_ave += acc
            print('Train Epoch: {} [{}/{}] Loss: {:.5f} Acc: {:.4f} lr: {:.2e}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset), loss_ave/nb_data, acc_ave/nb_data * log_interval, optimizer.param_groups[0]['lr']
            ))
            if weight_bits is not None:
                print("\t num_bits {}".format(weight_bits))
                total = 0
                for i in range(len(weight_bits)):
                    total += weight_bits[i] * nnz[i]
                print("\t model size {}".format(total))

    if prune_tk is not None:
        if weight_bits is not None:
            if nnzfix:
                prune_tk(model)
            else:
                prune_tk(list(param_list(model)), weight_bits)
        else:
            prune_tk(model)
        
    acc_ave /= nb_data
    return acc_ave, weight_bits


def acc_test_call(output, target, type="vanilla"):
    if type is "vanilla":
        pred = output.data.max(1)[1]
        correct = pred.cpu().eq(target).sum().item()
        return correct
    if type is "ensemble":
        with torch.no_grad():
            # assert output is list
            pred = []
            for op in output:
                op = F.softmax(op, dim=1)
                pred.append(op)
            pred = torch.fmod(torch.cat(pred, dim=1).data.max(1)[1], 10)
            correct = pred.cpu().eq(target).sum().item()
        return correct
    return 0.


def model_test(model, epoch, data_loader, atk_algo, atk_eps, iscuda=False, adv_iter=16, criterion=F.cross_entropy, modelType="vanilla"):
    model.eval()
    test_loss, correct = 0, 0
    testLossAdv, correctAdv = 0, 0
    adv_l2dist, adv_linfdist = 0, 0
    nb_data = 0
    for data, target in data_loader:
        indx_target = target.clone()
        data_len = data.shape[0]
        nb_data += data_len
        if iscuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
        test_loss += criterion(output, target).data.item()

        correct += acc_test_call(output, indx_target, type=modelType)

        data_adv = atk_algo(x=data, y=target, criterion=criterion, model=model, rho=atk_eps, steps=adv_iter, iscuda=iscuda).data
        with torch.no_grad():
            output_adv = model(data_adv)
        testLossAdv += criterion(output_adv, target).data.item()

        correctAdv += acc_test_call(output_adv, indx_target, type=modelType)

        adv_l2dist += torch.norm((data - data_adv).view(data.size(0), -1), p=2, dim=-1).sum().item()
        tmp = torch.max((data - data_adv).view(data.size(0), -1).abs(), dim=-1)[0]
        adv_linfdist += tmp.sum().item()

    
    test_loss /= nb_data
    acc = float(100. * correct) / nb_data
    print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, nb_data, acc))

    testLossAdv /= nb_data
    accAdv = float(100. * correctAdv) / nb_data
    print('\tAdv set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(testLossAdv, correctAdv, nb_data, accAdv))

    adv_l2dist /= nb_data
    adv_linfdist /= nb_data
    print('\tAdv Dist: L2: {:.4f}, Linf: {:.4f}'.format(adv_l2dist, adv_linfdist))

    return acc, accAdv


