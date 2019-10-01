import numpy as np
import torch
import torch.nn as nn
# from dataset import attack_eps, attack_range

def attack_eps(rho):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    channels = [rho/255./s for s in std]
    return channels

def attack_range():
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    channels = []
    for i in range(len(std)):
        channels.append(
            [-mean[i]/std[i], (1-mean[i])/std[i]]
        )
    return channels

def cross_entropy(input, target, label_smoothing=0.0, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    if label_smoothing > 0:
        target = torch.clamp(target, max=1-label_smoothing, min=label_smoothing/9.0)

    logsoftmax = nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]

    return res


def l2ball_proj(center, radius, t, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t

    direction = t - center
    dist = direction.view(direction.size(0), -1).norm(p=2, dim=1, keepdim=True)
    direction.view(direction.size(0), -1).div_(dist)
    dist[dist > radius] = radius
    direction.view(direction.size(0), -1).mul_(dist)
    res.data.copy_(center + direction)
    return res

def linfball_proj(center, radius, t, in_place=True):
    return tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)

def edgecut(data, min, max, in_place=True):
    if not in_place:
        res = data.clone()
    else:
        res = data
    idx = res.data < min
    res.data[idx] = min
    idx = res.data > max
    res.data[idx] = max

    return res
    # return tensor_clamp(data, min=min, max=max, in_place=in_place)


_extra_args = {'alpha', 'steps', 'randinit', 'gamma', 'iscuda'}

def fgsm_gt(x, y, criterion, rho=None, model=None, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if rho is None:
        rho = 3

    eps = attack_eps(rho)
    rgb = attack_range()
    # Compute loss
    x_adv = x.clone()
    x_adv.requires_grad = True

    loss_adv0 = criterion(model(x_adv), y, reduction='sum')
    grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]

    for i in range(len(eps)):
        alpha = eps[i]
        # print(alpha)
        x_adv[:,i,:,:].data.add_(alpha * torch.sign(grad0.data[:,i,:,:]))
        tmp = linfball_proj(center=x[:,i,:,:], radius=alpha, t=x_adv[:,i,:,:])
        x_adv[:,i,:,:].data.copy_(tmp.data)
        edgecut(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])
        # x_adv[:,i,:,:] = torch.clamp(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])

    # x_adv.data.add_(eps * torch.sign(grad0.data))

    return x_adv.data

def rfgsm_gt(x, y, criterion, rho=None, model=None, iscuda=False, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if rho is None:
        rho = 3

    eps = attack_eps(rho)
    eps_torch = torch.tensor(eps).view(1, len(eps), 1, 1)
    if iscuda:
        eps_torch = eps_torch.cuda()
        # print("update eps to cuda")
    rgb = attack_range()
    # compute loss
    x_adv = x.clone()
    x_adv.requires_grad = True

    if randinit:
        # pertub = torch.sign( torch.randn_like(x_adv) )
        x_adv.data.add( (2.0 * torch.rand_like(x_adv) - 1.0) * eps_torch )
        for i in range(len(eps)):
            alpha = eps[i]
            # x_adv[:,i,:,:].data.add_(alpha * pertub[:,i,:,:])
            linfball_proj(center=x[:,i,:,:], radius=alpha, t=x_adv[:,i,:,:])
            edgecut(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])

    loss_adv0 = criterion(model(x_adv), y, reduction="sum")
    grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]

    for i in range(len(eps)):
        alpha = eps[i] / 2.
        x_adv[:,i,:,:].data.add_(alpha * torch.sign(grad0.data[:,i,:,:]))
        linfball_proj(center=x[:,i,:,:], radius=eps[i], t=x_adv[:,i,:,:])
        edgecut(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])
        # x_adv[:,i,:,:] = torch.clamp(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])

    return x_adv.data
        

def ifgsm_gt(x, y, criterion, rho=None, model=None, steps=3, randinit=False, iscuda=False, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if rho is None:
        rho = 3
    eps = attack_eps(rho)
    eps_torch = torch.tensor(eps).view(1, len(eps), 1, 1)
    if iscuda:
        eps_torch = eps_torch.cuda()
        # print("update eps to cuda")
    rgb = attack_range()
    # compute loss
    x_adv = x.clone()
    x_adv.requires_grad = True

    if randinit:
        # pertub = torch.sign( torch.randn_like(x_adv) )
        x_adv.data.add( (2.0 * torch.rand_like(x_adv) - 1.0) * eps_torch )
        for i in range(len(eps)):
            alpha = eps[i]
            # x_adv[:,i,:,:].data.add_(alpha * pertub[:,i,:,:])
            linfball_proj(center=x[:,i,:,:], radius=alpha, t=x_adv[:,i,:,:])
            edgecut(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])

    for _ in range(steps):
        loss_adv = criterion(model(x_adv), y, reduction="sum")
        grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            for i in range(len(eps)):
                alpha = (eps[i] * 1.25) / steps
                x_adv[:,i,:,:].data.add_(alpha * torch.sign(grad.data[:,i,:,:]))
                # print(eps[i])
                linfball_proj(center=x[:,i,:,:], radius=eps[i], t=x_adv[:,i,:,:])
                # x_adv[:,i,:,:].data.copy_(tmp.data)
                edgecut(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])

    # tmp = torch.max((x - x_adv).view(x.size(0), -1).abs(), dim=-1)[0]
    # tmp = x_adv.max()
    # print(x_adv.min(), x_adv.max())
                
                # x_adv[:,i,:,:].data.fill_(torch.clamp(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1]))

    return x_adv.data


def pgd_gt(x, y, criterion, rho=None, model=None, steps=3, randinit=True, iscuda=False, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if rho is None:
        rho = 3
    eps = attack_eps(rho)
    eps_torch = torch.tensor(eps).view(1, len(eps), 1, 1)
    if iscuda:
        eps_torch = eps_torch.cuda()
        # print("update eps to cuda")
    rgb = attack_range()
    # compute loss
    x_adv = x.clone()
    x_adv.requires_grad = True

    if randinit:
        # pertub = torch.sign( torch.randn_like(x_adv) )
        x_adv.data.add( (2.0 * torch.rand_like(x_adv) - 1.0) * eps_torch )
        for i in range(len(eps)):
            alpha = eps[i]
            # x_adv[:,i,:,:].data.add_(alpha * pertub[:,i,:,:])
            linfball_proj(center=x[:,i,:,:], radius=alpha, t=x_adv[:,i,:,:])
            edgecut(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])

    for _ in range(steps):
        loss_adv = criterion(model(x_adv), y, reduction="sum")
        grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            for i in range(len(eps)):
                alpha = (eps[i] * 1.25) / steps
                x_adv[:,i,:,:].data.add_(alpha * torch.sign(grad.data[:,i,:,:]))
                linfball_proj(center=x[:,i,:,:], radius=eps[i], t=x_adv[:,i,:,:])
                edgecut(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])
                # x_adv[:,i,:,:].data.fill_(torch.clamp(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1]))
    return x_adv.data

def grad_gt(x, y, criterion, rho=None, model=None, steps=3, randinit=False, iscuda=False, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if rho is None:
        rho = 3
    eps = attack_eps(rho)
    eps_torch = torch.tensor(eps).view(1, len(eps), 1, 1)
    if iscuda:
        eps_torch = eps_torch.cuda()
        # print("update eps to cuda")
    rgb = attack_range()
    # compute loss
    x_adv = x.clone()
    x_adv.requires_grad = True

    if randinit:
        # pertub = torch.sign( torch.randn_like(x_adv) )
        x_adv.data.add( (2.0 * torch.rand_like(x_adv) - 1.0) * eps_torch )
        for i in range(len(eps)):
            alpha = eps[i]
            # x_adv[:,i,:,:].data.add_(alpha * pertub[:,i,:,:])
            linfball_proj(center=x[:,i,:,:], radius=alpha, t=x_adv[:,i,:,:])
            edgecut(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])

    
    for _ in range(steps):
        loss_adv = criterion(model(x_adv), y, reduction="sum")
        grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            for i in range(len(eps)):
                alpha = (eps[i] * 1.25) / steps / grad.data[:, i, :, :].abs().mean()
                x_adv[:,i,:,:].data.add_(alpha * grad.data[:,i,:,:])
                linfball_proj(center=x[:,i,:,:], radius=eps[i], t=x_adv[:,i,:,:])
                # edgecut(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])
                # x_adv[:,i,:,:].data.fill_(torch.clamp(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1]))
    return x_adv.data


def wrm_gt(x, y, criterion, rho=None, model=None, steps=7, randinit=False, gamma=None, iscuda=False, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if gamma is None:
        gamma = 20

    eps = attack_eps(rho)
    eps_torch = torch.tensor(eps).view(1, len(eps), 1, 1)
    if iscuda:
        eps_torch = eps_torch.cuda()
        # print("update eps to cuda")
    rgb = attack_range()
    # compute loss
    x_adv = x.clone()
    x_adv.requires_grad = True

    if randinit:
        # pertub = torch.sign( torch.randn_like(x_adv) )
        x_adv.data.add( (2.0 * torch.rand_like(x_adv) - 1.0) * eps_torch )
        for i in range(len(eps)):
            alpha = eps[i]
            # x_adv[:,i,:,:].data.add_(alpha * pertub[:,i,:,:])
            linfball_proj(center=x[:,i,:,:], radius=alpha, t=x_adv[:,i,:,:])
            edgecut(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])

    x_adv.requires_grad = True
    ord = 2
    for t in range(steps):
        loss_adv = gamma * criterion(model(x_adv), y, reduction="sum") - \
            0.5 * torch.sum(torch.norm((x_adv - x.data).view(x_adv.size(0), -1), p=ord, dim=1) ** 2)
        grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
        scale = float(1./np.sqrt(t + 1))
        x_adv.data.add_(scale * grad.data)
        for i in range(len(eps)):
            # linfball_proj(center=x_adv[:,i,:,:], radius=alpha, t=x_adv[:,i,:,:])
            edgecut(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1])
    
    return x_adv.data



def wrm(x, preds, loss_fn, y=None, gamma=None, model=None, steps=3, randinit=False, eps=0.062, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if gamma is None:
        gamma = 1.3
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        # x_adv += torch.randn_like(x_adv).clamp_(min=-1.0, max=1.0) * eps
        x_adv += (2.0 * torch.rand_like(x_adv) - 1.0) * eps

    x_adv.requires_grad = True

    ord = 2
    for t in range(steps):
        loss_adv0 = gamma * loss_fn(model(x_adv), y, reduction="sum") - \
                    0.5 * torch.sum(torch.norm((x_adv - x.data).view(x_adv.size(0), -1), p=ord, dim=1) ** 2)
        
        # loss_adv0.backward()
        # grad0 = x_adv.grad
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        scale = float(1./np.sqrt(t+1))
        x_adv.data.add_(scale * grad0.data)
        # x_adv.grad.data.zero_()
        # print("intermedia_grad0:", torch.norm(grad0))

    linfball_proj(x, eps, x_adv, in_place=True)
    return x_adv


def fgm(x, preds, loss_fn, y=None, eps=None, model=None, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    x_adv.requires_grad = True

    # print("right")
    loss_adv0 = loss_fn(model(x_adv), y, reduction='sum')
    grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
    x_adv.data.add_(eps * torch.sign(grad0.data))

    return x_adv

def input_reg(x, preds, loss_fn, y=None, eps=None, model=None, label_smoothing=0.0):
    if eps is None:
        eps = 1e-4
    if y is None:
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # compute loss
    x_adv = x.clone()
    x_adv.requires_grad = True
    loss_adv0 = loss_fn(model(x_adv), y, reduction='sum')
    grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
    x_adv.requires_grad = False

    return eps * torch.sum(grad0.data **2)


def ifgm(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, alpha=None, randinit=False, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if alpha is None:
        alpha = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        # x_adv += torch.randn_like(x_adv).clamp_(min=-1.0, max=1.0) * eps
        x_adv += (2.0 * torch.rand_like(x_adv) - 1.0) * eps
    x_adv.requires_grad = True

    for _ in range(steps):
        loss_adv0 = loss_fn(model(x_adv), y, reduction='sum')
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(alpha * torch.sign(grad0.data))

        linfball_proj(x, eps, x_adv, in_place=True)
    return x_adv

def clamp(x, min_range, max_range):
    N, C, H, W = x.shape
    xadv = x.data.clone()
    for i in range(C):
        xadv[:,i,:,:] = torch.clamp(x[:,i,:,:], max=max_range[i], min=min_range[i])
    return xadv

def ifgm_attack(x, preds, loss_fn, y=None, eps=None, model=None, steps=3, alpha=None, randinit=False, **kwargs):
    if len(kwargs) > 0:
        assert set(kwargs.keys()).issubset(_extra_args)
    if eps is None:
        # eps = 0.07972772183418274
        # eps = 0.45474205
        eps = 0.062
    if alpha is None:
        alpha = (eps * 1.25) / steps
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    if randinit:
        # x_adv += torch.randn_like(x_adv).clamp_(min=-1.0, max=1.0) * eps
        x_adv += torch.sign(2.0 * torch.rand_like(x_adv) - 1.0) * eps
    x_adv.requires_grad = True

    for _ in range(steps):
        loss_adv0 = loss_fn(model(x_adv), y, reduction='sum')
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(alpha * torch.sign(grad0.data))

        linfball_proj(x, eps, x_adv, in_place=True)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    min_range = []
    max_range = []
    for i in range(3):
        max_range.append((1.0 - mean[i])/std[i])
        min_range.append((0.0 - mean[i])/std[i])
    x_adv = clamp(x_adv, min_range, max_range)
    return x_adv

def pgm(x, preds, loss_fn, y=None, eps=None, model=None, steps=16, **kwargs):
    raise DeprecationWarning
    if eps is None:
        # eps = 0.33910248303413393
        eps = 3.27090588
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = preds.data.max(1)[1]
        y = torch.equal(preds, preds_max).float()

    # Compute loss

    x_adv = x.clone()
    x_adv.requires_grad = True

    for t in range(steps):
        loss_adv0 = loss_fn(model(x_adv), y, reduction='sum')
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        scale = float(1./np.sqrt(t+1)) * 3.0
        x_adv.data.add_(scale * grad0.data)

        l2ball_proj(x, eps, x_adv, in_place=True)
    return x_adv


def eval_adv_model(model, data_loader, attack_algo=pgm, attack_eps=None, cuda=True):
    model.eval()
    test_loss = 0
    adv_loss = 0
    correct = 0
    correct_adv = 0
    adv_l2dist = 0.0
    adv_linfdist = 0.0
    for data, target in data_loader:
        indx_target = target.clone()

        target_ = torch.unsqueeze(target, 1)
        one_hot = torch.FloatTensor(target.size()[0], 10).zero_()
        one_hot.scatter_(1, target_, 1)

        if cuda:
            data, target = data.cuda(), one_hot.cuda()
        else:
            target = one_hot
        with torch.no_grad():
            output = model(data)

        data_adv = attack_algo(data, output, y=target, eps=attack_eps, model=model, label_smoothing=0.0).data
        adv_l2dist += torch.norm((data - data_adv).view(data.size(0), -1), p=2, dim=-1).sum().item()
        adv_linfdist += torch.max((data - data_adv).view(data.size(0), -1).abs(), dim=-1)[0].sum().item()
        with torch.no_grad():
            output_adv = model(data_adv)
        adv_loss += cross_entropy(output_adv, target, 0.0, size_average=False).data.item()
        pred_adv = output_adv.data.max(1)[1]
        correct_adv += pred_adv.cpu().eq(indx_target).sum()
        test_loss += cross_entropy(output, target, 0.0, size_average=False).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()

    test_loss /= len(data_loader.dataset)  # average over number of mini-batch
    acc = float(100. * correct) / len(data_loader.dataset)
    print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(data_loader.dataset), acc))

    adv_loss /= len(data_loader.dataset)
    acc_adv = float(100. * correct_adv) / len(data_loader.dataset)
    print('\tAdv set: Average loss: {:.4f}, Accuracy : {}/{} ({:.0f}%)'.format(
        adv_loss, correct_adv, len(data_loader.dataset), acc_adv
    ))
    adv_l2dist /= len(data_loader.dataset)
    adv_linfdist /= len(data_loader.dataset)
    print('\tAdv dist: L2: {:.4f}, Linf: {:.4f}'.format(adv_l2dist, adv_linfdist))

    return {'test_loss': test_loss, 'test_acc': acc, 'adv_loss': adv_loss, 'adv_acc': acc_adv, 'adv_l2dist': adv_l2dist,
            'adv_linfdist': adv_linfdist}