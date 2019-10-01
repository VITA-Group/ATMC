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
            # print(output)
            # assert output is list
            # output0, output1 = output[0], output[1]
            # pred0 = F.softmax(output0, dim=1)
            # pred1 = F.softmax(output1, dim=1)
            # pred = (pred0 + pred1).data.max(1)[1]
            pred = []
            for op in output:
                op = F.softmax(op, dim=1)
                pred.append(op)
            pred = torch.fmod(torch.cat(pred, dim=1).data.max(1)[1], 10)
            correct = pred.cpu().eq(target).sum().item()
            acc = correct * 1.
        return acc
    return 0.


def model_train(model, epoch, data_loader, optimizer, \
     dfn_algo, dfn_eps, log_interval, iscuda=False, adv_iter=16, criterion=F.cross_entropy, modelType="vanilla"):
    model.train()
    loss_ave = 0.
    nb_data = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        nb_data += len(data)
        indx_target = target.clone()
        if iscuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if dfn_algo is None:
            data_adv = data
        else:
            data_adv = dfn_algo(x=data, y=target, criterion=criterion, rho=dfn_eps, model=model, steps=adv_iter, iscuda=iscuda)
        output = model(data_adv)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_ave += loss.item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            acc = acc_call(output, indx_target, type=modelType) / len(data)
            # print(data_adv.shape)
            # pred = output.data.max(1)[1]
            # correct = pred.cpu().eq(indx_target).sum().item()
            # acc = correct * 1. / len(data)
            print('Train Epoch: {} [{}/{}] Loss: {:.5f} Acc: {:.4f} lr: {:.2e}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset), loss_ave/nb_data, acc, optimizer.param_groups[0]['lr']
            ))
    
    return model

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
            # loss = 0.5 * criterion(output, target) + 0.5 * criterion(output_adv, target)
            loss = criterion(output_adv, target)
        
        # if prune_tk is not None:
        #     loss += rho / 2. * model.module.admm_regularizer(modelu, model_dual)
        if quantize_tk is not None:
            if isinstance(model, nn.DataParallel):
                # print("what?")
                loss += rho / 2. * model.module.admm_regularizer(model_u.module, model_dual.module)
            else:
                # print("show?")
                loss += rho / 2. * model.admm_regularizer(model_u, model_dual)
        loss.backward()
        optimizer.step()
        loss_ave += loss.item()
        if batch_idx % proj_interval == 0:
            with torch.no_grad():
                if prune_tk is not None:
                    if weight_bits is not None:
                        # prune_tk(list(param_list(model)), weight_bits)
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
                            # print("quantization ensure")
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
    # prune_tk(model)
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


def model_train_admm(models, epoch, data_loader, optimizer, \
     dfn_algo, dfn_eps, log_interval, iscuda=False, adv_iter=16, \
     criterion=F.cross_entropy, modelType="vanilla", prune_tk=None, \
     quantize_tk=None, rho=0.01, admm_interval=5):
    model, modelz, modely, modelu, modelv = models
    model.train()
    loss_ave = 0.
    nb_data = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        nb_data += len(data)
        indx_target = target.clone()
        if iscuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if dfn_algo is None:
            data_adv = data
        else:
            data_adv = dfn_algo(x=data, y=target, criterion=criterion, rho=dfn_eps, model=model, steps=adv_iter, iscuda=iscuda)
        output = model(data_adv)
        loss = criterion(output, target) 
        if prune_tk is not None:
            loss += rho/2. * model.module.admm_regularizer(modelu, modelz) 
        if quantize_tk is not None:
            loss += rho/2. * model.module.admm_regularizer(modelv, modely)
        # loss = 1 + rho/2. * model.admm_regularizer(modelu, modelz)
        loss.backward()
        optimizer.step()
        loss_ave += loss.item()
        if batch_idx % admm_interval == 0:
            with torch.no_grad():
                if prune_tk is not None:
                    # prune_tk(model)
                    modelz.duplicate_plus(model, modelu)
                    prune_tk(modelz)
                    # prune_tk(weight_list, weight_bits)
                    modelu.duplicate_update(model, modelz, rho)
                
                if quantize_tk is not None:
                    modely.duplicate_plus(model, modelv)
                    quantize_tk(modely)
                    # quantize_tk(weight_list, weight_nnz)
                    modelv.duplicate_update(model, modely, rho)

        if batch_idx % log_interval == 0 and batch_idx > 0:
            acc = acc_call(output, indx_target, type=modelType) / len(data)
            # print(data_adv.shape)
            # pred = output.data.max(1)[1]
            # correct = pred.cpu().eq(indx_target).sum().item()
            # acc = correct * 1. / len(data)
            print('Train Epoch: {} [{}/{}] Loss: {:.5f} Acc: {:.4f} lr: {:.2e}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset), loss_ave/nb_data, acc, optimizer.param_groups[0]['lr']
            ))
    
    return model

def model_train_admm_ms(models, epoch, data_loader, optimizer, \
     dfn_algo, dfn_eps, log_interval, layernnz, param_list, iscuda=False, adv_iter=16, \
     criterion=F.cross_entropy, modelType="vanilla", prune_tk=None, \
     quantize_tk=None, rho=0.01, admm_interval=5, weight_bits=None):
    model, modelz, modely, modelu, modelv = models
    model.train()
    loss_ave = 0.
    nb_data = 0
    time_begin = time.time()
    prune_time = 0
    quant_time = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        nb_data += len(data)
        indx_target = target.clone()
        if iscuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if dfn_algo is None:
            data_adv = data
        else:
            data_adv = dfn_algo(x=data, y=target, criterion=criterion, rho=dfn_eps, model=model, steps=adv_iter, iscuda=iscuda)
        output = model(data_adv)
        loss = criterion(output, target) 
        if prune_tk is not None:
            if isinstance(model, nn.DataParallel):
                loss += rho/2. * model.module.admm_regularizer(modelu.module, modelz.module) 
            else:
                loss += rho/2. * model.admm_regularizer(modelu, modelz) 
        if quantize_tk is not None:
            if isinstance(model, nn.DataParallel):
                loss += rho/2. * model.module.admm_regularizer(modelv.module, modely.module)
            else:
                loss += rho/2. * model.admm_regularizer(modelv, modely)
        # loss = 1 + rho/2. * model.admm_regularizer(modelu, modelz)
        loss.backward()
        optimizer.step()
        loss_ave += loss.item()
        if batch_idx % admm_interval == 0:
            with torch.no_grad():
                if prune_tk is not None:
                    # prune_tk(model)
                    if isinstance(model, nn.DataParallel):
                        modelz.module.duplicate_plus(model.module, modelu.module)
                        # prune_tk(modelz)
                        ptime_begin = time.time()
                        prune_tk(list(param_list(modelz.module)), weight_bits)
                        prune_time += time.time() - ptime_begin
                        modelu.module.duplicate_update(model.module, modelz.module, rho)
                    else:
                        modelz.duplicate_plus(model, modelu)
                        # prune_tk(modelz)
                        ptime_begin = time.time()
                        prune_tk(list(param_list(modelz)), weight_bits)
                        prune_time += time.time() - ptime_begin
                        modelu.duplicate_update(model, modelz, rho)
                    nnz = layernnz(modelz)
                
                if quantize_tk is not None:
                    if isinstance(model, nn.DataParallel):
                        modely.module.duplicate_plus(model.module, modelv.module)
                        # quantize_tk(modely)
                        qtime_begin = time.time()
                        weight_bits, clusters = quantize_tk(list(param_list(modely.module)), nnz, list(param_list(modelz.module)))
                        quant_time += time.time() - qtime_begin
                        modelv.module.duplicate_update(model.module, modely.module, rho)
                    else:
                        modely.duplicate_plus(model, modelv)
                        # quantize_tk(modely)
                        qtime_begin = time.time()
                        weight_bits, clusters = quantize_tk(list(param_list(modely)), nnz, list(param_list(modelz)))
                        quant_time += time.time() - qtime_begin
                        modelv.duplicate_update(model, modely, rho)
            

        if batch_idx % log_interval == 0: #and batch_idx > 0:
            time_suspend = time.time()
            acc = acc_call(output, indx_target, type=modelType) / len(data)
            each_batch_period = (time_suspend - time_begin) / nb_data

            print('Train Epoch: {} [{}/{}] Loss: {:.5f} Acc: {:.4f} lr: {:.2e}, {:.4f} s/batch'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset), loss_ave/nb_data, acc, optimizer.param_groups[0]['lr'], each_batch_period
            ))
            print("\t nnz {}".format(nnz))
            print("\t num_bits {}".format(weight_bits))
            print("\t prune_time {:.6f} quant_time {:.6f}".format(prune_time/nb_data, quant_time/nb_data))
            clusters_cpu = [sorted(cluster.cpu().numpy()) for cluster in clusters]
            for cluster in clusters_cpu:
                # print("cluster [{}]".format("{:.3e}".format([k for k in cluster.cpu().numpy()])))
                print("\t cluster [" + " ".join("{:.3e}".format(k) for k in sorted(cluster)) + "]")
        # break
    
    return model, weight_bits, clusters_cpu

ALPHA, BETA, GAMMA = 1e3, 1., 5e-2

def model_uai_train(model_fea, model_pred, model_distortion, epoch, data_loader, optims, dfn_algo, dfn_eps, log_interval, iscuda=False, adv_iter=16, criterion=F.cross_entropy):
    model_fea.train()
    model_pred.train()
    model_distortion.train()
    loss_ave = 0.
    tLoss_ave = 0.
    distortionLoss_ave = 0.
    decLoss_ave = 0.
    nb_data = 0
    optim_pred, optim_distortion = optims[0], optims[1]
    model = lambda x: model_pred(model_fea(x))[0]
    for batch_idx, (data, target) in enumerate(data_loader):
        nb_data += len(data)
        indx_target = target.clone()
        if iscuda:
            data, target = data.cuda(), target.cuda()

        if dfn_algo is None:
            data_adv = data
        else:
            data_adv = dfn_algo(x=data, y=target, criterion=criterion, rho=dfn_eps, model=model, steps=adv_iter, iscuda=iscuda)

        optim_pred.zero_grad()
        # optim_distortion.zero_grad()

        sim = model_fea(data_adv)
        # output, decLoss = model_pred(sim)
        d1, d2 = model_distortion(sim)

        # loss = criterion(output, target)

        # loss += decLoss
        # tLoss = loss + d1 + d2
        dLoss = GAMMA * ( d1 + d2 )

        for iteration in range(5):
            optim_distortion.zero_grad()
            if iteration < 4:
                dLoss.backward(retain_graph=True)
            else:
                dLoss.backward()
            optim_distortion.step()

        xsim = model_fea(data)
        output, decLoss = model_pred(xsim)
        d1, d2 = model_distortion(xsim)
        loss = criterion(output, target)
        tLoss = ALPHA * loss + BETA * decLoss - GAMMA * ( d1 + d2 )

        optim_pred.zero_grad()
        tLoss.backward()
        optim_pred.step()

        loss_ave += loss.item()
        decLoss_ave += decLoss.item()
        tLoss_ave += tLoss.item()
        distortionLoss_ave -= d1.item() + d2.item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            pred = output.data.max(1)[1]
            correct = pred.cpu().eq(indx_target).sum().item()
            acc = correct * 1. / len(data)
            print('Train Epoch: {} [{}/{}] Loss: [{:.5f}, {:.5f}, {:.5f}, {:.5f}] Acc: {:.4f} lr: {:.2e}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset), loss_ave/nb_data, decLoss_ave/nb_data, tLoss_ave/nb_data, distortionLoss_ave/nb_data, acc, optim_pred.param_groups[0]['lr']
            ))

    return [model_fea, model_pred, model_distortion]

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
    std = torch.tensor([0.2471,    0.2435,    0.2616])

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
            std = std.cuda()
        with torch.no_grad():
            output = model(data)
        test_loss += criterion(output, target).data.item()
        # pred = output.data.max(1)[1]
        # correct += pred.cpu().eq(indx_target).sum()
        correct += acc_test_call(output, indx_target, type=modelType)

        data_adv = atk_algo(x=data, y=target, criterion=criterion, model=model, rho=atk_eps, steps=adv_iter, iscuda=iscuda).data
        with torch.no_grad():
            output_adv = model(data_adv)
        testLossAdv += criterion(output_adv, target).data.item()
        # pred_adv = output_adv.data.max(1)[1]
        # correctAdv += pred_adv.cpu().eq(indx_target).sum()
        correctAdv += acc_test_call(output_adv, indx_target, type=modelType)

        adv_l2dist += torch.norm((data - data_adv).view(data.size(0), -1), p=2, dim=-1).sum().item()
        linfdist_channels = torch.max((data-data_adv).view(data.size(0), data.size(1), -1).abs(), dim=-1)[0] * std
        linfdist = torch.max(linfdist_channels, dim=-1)[0]

        # tmp = torch.max((data - data_adv).view(data.size(0), -1).abs(), dim=-1)[0]
        # print(tmp)
        adv_linfdist += linfdist.sum().item()

    
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


def model_uai_test(model_fea, model_pred, epoch, data_loader, atk_algo, atk_eps, iscuda=False, adv_iter=16, criterion=F.cross_entropy):
    model_fea.eval()
    model_pred.eval()

    model = lambda x: model_pred(model_fea(x))[0]

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
            sim = model_fea(data)
            output, _ = model_pred(sim)

        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.cpu().eq(indx_target).sum()

        data_adv = atk_algo(x=data, y=target, criterion=criterion, model=model, rho=atk_eps, steps=adv_iter, iscuda=iscuda).data
        with torch.no_grad():
            output_adv = model(data_adv)
        testLossAdv += criterion(output_adv, target).data.item()
        pred_adv = output_adv.data.max(1)[1]
        correctAdv += pred_adv.cpu().eq(indx_target).sum()

        adv_l2dist += torch.norm((data - data_adv).view(data.size(0), -1), p=2, dim=-1).sum().item()
        adv_linfdist += torch.max((data - data_adv).view(data.size(0), -1).abs(), dim=-1)[0].sum().item()

    
    test_loss /= nb_data
    acc = float(100. * correct) / nb_data
    print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, nb_data, acc))

    testLossAdv /= nb_data
    accAdv = float(100. * correctAdv) / nb_data
    print('\tAdv set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(testLossAdv, correctAdv, nb_data, accAdv))

    adv_l2dist /= nb_data
    adv_linfdist /= nb_data
    print('\tAdv Dist: L2: {:.4f}, Linf: {:.4f}'.format(adv_l2dist, adv_linfdist))

    return accAdv