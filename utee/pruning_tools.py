import math
from collections import OrderedDict

import numpy as np
import torch.nn.functional as F

import torch
from torch import nn as nn
from torch.nn import Parameter, Sequential
from utee import misc
from torch.nn.modules.utils import _pair
import copy

print = misc.logger.info

# using hardware parameters from Eyeriss

default_s1 = int(100 * 1024 / 2)  # input cache, 100K (16-bit Fixed-Point)
default_s2 = 1 * int(8 * 1024 / 2)  # kernel cache, 8K (16-bit Fixed-Point)
default_m = 12
default_n = 14

# unit energy constants
default_e_mac = 1.0 + 1.0 + 1.0  # including both read and write RF
default_e_mem = 200.0
default_e_cache = 6.0
default_e_rf = 1.0

def param_list(model, param_name):
    for n, w in model.named_parameters():
        if n.split(".")[-1] in param_name:
            yield w

def modelsize_calculator(weight_nnz_list, weight_bits):
    assert isinstance(weight_bits, int) or len(weight_bits) == len(weight_nnz_list), "not match"

    if isinstance(weight_bits, int):
        modelsize = sum([item * weight_bits for item in weight_nnz_list])
    else:
        modelsize = 0
        for i in range(len(weight_bits)):
            modelsize += weight_bits[i] * weight_nnz_list[i]
    
    return modelsize


def prune_admm_ms(weight_list, weight_bits, model_size):
    param_flats = [p.data.view(-1) for p in weight_list]
    param_flats_all = torch.cat(param_flats, dim=0)
    knapsack_weight_all = torch.cat([torch.ones_like(p) * weight_bits[i] for i, p in enumerate(param_flats)], dim=0)
    score_all = torch.cat([p.abs() for p in param_flats], dim=0) * knapsack_weight_all

    _, sorted_idx = torch.sort(score_all, descending=True)
    cumsum = torch.cumsum(knapsack_weight_all[sorted_idx], dim=0)
    res_nnz = torch.nonzero(cumsum <= model_size).max() 
    z_idx = sorted_idx[-(param_flats_all.numel() - res_nnz):]
    param_flats_all[z_idx] = 0.0
    # in-place zero-out
    i = 0
    for p in weight_list:
        p.data.view(-1).copy_(param_flats_all[i:i+p.numel()])
        i += p.numel()


def k_means1D(X, Xnnz, n_clusters, niter=100):
    X = X.view(-1)
    Xnnz = Xnnz.view(-1)
    idx = torch.nonzero(Xnnz)
    if idx.shape[0] <= n_clusters - 1:
        indices = idx
        val_dict = torch.cat([X[indices].clone().view(-1), torch.zeros(n_clusters - idx.shape[0]).cuda()], dim=0)
        return val_dict
    else:
        indices = torch.randperm(idx.shape[0])[:n_clusters - 1]
        idx0 = idx[indices]
        val_dict = torch.cat([X[idx0].clone().view(-1), torch.zeros(1).cuda()], dim=0)

    tol = 1e-3
    pre_dist = None
    one_hot = torch.zeros(X.shape[0], n_clusters).cuda()
    for t in range(niter):
        # assign codes
        km_dist, km_code = torch.min((X.view(-1, 1) - val_dict) ** 2, dim=1)
        cur_dist = km_dist.sum().item()
        if pre_dist is not None and cur_dist > 0 and abs(pre_dist - cur_dist) / cur_dist < tol:
            return val_dict
        pre_dist = cur_dist

        one_hot = one_hot.zero_()
        one_hot.scatter_(1, km_code.unsqueeze(1), 1)

        Xp = (X.unsqueeze(1) * one_hot).sum(dim=0)
        Xsum = one_hot.sum(dim=0)
        idx = torch.nonzero(Xsum)
        Xsum[idx] = 1./Xsum[idx]
        val_dict = Xp * Xsum

    return val_dict


def get_optim_val_dict(input, input_nnz, nbits, niter=100):
    X = input.data
    nbins = 2 ** nbits + 1
    val_dict = k_means1D(X, input_nnz, nbins, niter=niter)
    return val_dict


def km_quantize_tensor(input, nbits, val_dict=None):
    if val_dict is None:
        val_dict = get_optim_val_dict(input, nbits)
    km_dist, km_code = torch.min((input.view(-1, 1) - val_dict) ** 2, dim=1)
    res = val_dict[km_code].view(input.shape)
    return res, km_dist


def mckp_greedy(profit, weight, group_size, budget, sorted_weights=True):
    """
    Greedy algorithm for Multi-Choice knapsack problem
    :param profit: items' profits
    :param weight: items' weights
    :param group_size: groups' size
    :param budget: weight budget
    :param sorted_weights: if each group's items are sorted by weights
    :return: binary solution of selected items
    """
    # get group offsets
    offset = [0] * len(group_size)
    temp = 0
    for i in range(len(group_size)):
        offset[i] = temp
        temp += group_size[i]

    if not sorted_weights:
        raw_sorted_idx = torch.zeros_like(profit, dtype=torch.long)
        for i in range(len(group_size)):
            if i + 1 < len(offset):
                indices = torch.argsort(weight[offset[i]:offset[i + 1]])
                raw_sorted_idx[offset[i]:offset[i+1]] = indices + offset[i]
            else:
                indices = torch.argsort(weight[offset[i]:])
                raw_sorted_idx[offset[i]:] = indices + offset[i]
        weight = weight[raw_sorted_idx].clone()
        profit = profit[raw_sorted_idx].clone()

    profit -= (profit.min() - 1e-6)
    # preprocess: remove the dominated items
    idx = torch.ones_like(profit, dtype=torch.uint8)
    reduced_group_size = copy.deepcopy(group_size)
    for gi, gs in enumerate(group_size):
        if gs <= 1:
            continue
        go = offset[gi]
        temp = profit[go]
        for i in range(1, gs):
            if profit[go+i] <= temp:
                idx[go+i] = 0
                reduced_group_size[gi] -= 1
            else:
                temp = profit[go+i]

        if reduced_group_size[gi] <= 2:
            continue
        stack = [(go, None)]
        for i in range(1, gs):
            cur_idx = go + i
            if bool(idx[cur_idx]):
                while True:
                    score = ((profit[cur_idx] - profit[stack[-1][0]]) / (weight[cur_idx] - weight[stack[-1][0]])).item()
                    if len(stack) <= 1 or score < stack[-1][1]:
                        stack.append((cur_idx, score))
                        break
                    else:
                        del_idx, del_score = stack.pop()
                        idx[del_idx] = 0
                        reduced_group_size[gi] -= 1
    # greedy algorithm
    R_profit = profit[idx]
    R_d_profit = R_profit.clone()
    R_d_profit[1:] -= R_profit[:-1]
    R_weight = weight[idx]
    R_d_weight = R_weight.clone()
    R_d_weight[1:] -= R_weight[:-1]

    R_score = R_d_profit / R_d_weight
    sorted_idx = sorted(range(len(R_score)), key=R_score.__getitem__, reverse=True)

    res = torch.zeros(len(R_score), dtype=torch.uint8)
    res_profit = 0.0
    res_weight = 0.0
    group_idices = []
    offset = [0] * len(reduced_group_size)
    temp = 0
    for i in range(len(reduced_group_size)):
        offset[i] = temp
        # select the first item in each group
        res[offset[i]] = 1
        res_profit += R_profit[offset[i]].item()
        res_weight += R_weight[offset[i]].item()
        temp += reduced_group_size[i]
        group_idices += [i] * reduced_group_size[i]

    offset = set(offset)
    finished_group_indices = set()
    for i in sorted_idx:
        if i not in offset:
            if res_weight + R_d_weight[i].item() > budget:
                # break
                finished_group_indices.add(group_idices[i])
            if group_idices[i] not in finished_group_indices:
                if res[i-1] != 1:
                    print('idx={} is not selected, but {} is selecting'.format(i-1, i))
                assert res[i-1] == 1, 'sorted idx={}, offset={}, profit={}, weight={}'.format(sorted_idx, offset,
                                                                                              R_d_profit, R_d_weight)
                res[i-1] = 0
                res[i] = 1
                assert R_d_profit[i].item() >= 0
                res_profit += R_d_profit[i].item()
                res_weight += R_d_weight[i].item()

    raw_res = torch.zeros_like(profit, dtype=torch.uint8)
    raw_res[idx] = res
    if not sorted_weights:
        raw_raw_res = torch.zeros_like(raw_res)
        raw_raw_res[raw_sorted_idx] = raw_res
        return raw_raw_res
    return raw_res


def quantize_admm_ms(weight_list, num_nnz, weight_nnz, model_size, include_dict=False):
    nbits_dict = [1, 2, 3, 4, 5, 6, 7, 8]
    nbits_dict = torch.tensor(nbits_dict, dtype=torch.float32)
    mckp_budget = model_size
    mckp_p = []
    mckp_w = []
    mckp_gs = []
    weight_bits = []
    val_dicts_list = []
    for i, p in enumerate(weight_list):
        dist4nbits = []
        val_dicts = {}
        pnorm = p.data.norm().item() ** 2
        for nbits in nbits_dict:
            nbits = int(nbits)
            val_dict = get_optim_val_dict(p.data, weight_nnz[i].data, nbits, niter=100)
            val_dicts[nbits] = val_dict
            dist = km_quantize_tensor(p.data, nbits, val_dict=val_dict)[1].sum().item()
            dist4nbits.append(dist / pnorm)

        mckp_p.append(-torch.tensor(dist4nbits, dtype=torch.float))
        dict_size = (2.0 ** nbits_dict + 1) * 32 if include_dict else 0.0
        mckp_w.append(nbits_dict * num_nnz[i] + dict_size)
        mckp_gs.append(len(dist4nbits))
        val_dicts_list.append(val_dicts)
        print("{} th layer {}".format(i, [round(dist, 5) for dist in dist4nbits]))
        if i == 3:
            print("\t")

    mckp_p = torch.cat(mckp_p, dim=0)
    mckp_w = torch.cat(mckp_w, dim=0)
    x = mckp_greedy(mckp_p, mckp_w, mckp_gs, mckp_budget, sorted_weights=True)
    offset = 0
    offered_cluster = []
    for i in range(len(mckp_gs)):
        nbits = int(nbits_dict[x[offset:offset + mckp_gs[i]]])
        val_dict = val_dicts_list[i][nbits]
        offered_cluster.append(val_dict)
        weight_list[i].data.copy_(km_quantize_tensor(weight_list[i].data, nbits, val_dict=val_dict)[0])
        offset += mckp_gs[i]
        weight_bits.append(nbits)

    return weight_bits, offered_cluster

def quantize_bit_compute(weight_list, num_nnz, weight_nnz, model_size, include_dict=False):
    nbits_dict = [1, 2, 3, 4, 5, 6, 7, 8]
    nbits_dict = torch.tensor(nbits_dict, dtype=torch.float32)
    mckp_budget = model_size
    mckp_p = []
    mckp_w = []
    mckp_gs = []
    weight_bits = []
    val_dicts_list = []
    for i, p in enumerate(weight_list):
        dist4nbits = []
        val_dicts = {}
        for nbits in nbits_dict:
            nbits = int(nbits)
            val_dict = get_optim_val_dict(p.data, weight_nnz[i].data, nbits, niter=100)
            val_dicts[nbits] = val_dict
            dist = km_quantize_tensor(p.data, nbits, val_dict=val_dict)[1].sum().item()
            dist4nbits.append(dist)

        mckp_p.append(-torch.tensor(dist4nbits, dtype=torch.float))
        dict_size = (2.0 ** nbits_dict) * 32 if include_dict else 0.0
        mckp_w.append(nbits_dict * num_nnz[i] + dict_size)
        mckp_gs.append(len(dist4nbits))
        val_dicts_list.append(val_dicts)
        print("{} th layer {}".format(i, [round(dist, 5) for dist in dist4nbits]))
        if i == 3:
            print("\t")

    mckp_p = torch.cat(mckp_p, dim=0)
    mckp_w = torch.cat(mckp_w, dim=0)
    x = mckp_greedy(mckp_p, mckp_w, mckp_gs, mckp_budget, sorted_weights=True)
    offset = 0
    offered_cluster = []
    for i in range(len(mckp_gs)):
        nbits = int(nbits_dict[x[offset:offset + mckp_gs[i]]])
        val_dict = val_dicts_list[i][nbits]
        offered_cluster.append(val_dict)
        offset += mckp_gs[i]
        weight_bits.append(nbits)

    return weight_bits, offered_cluster


def copy_model_weights(model, W_flat, W_shapes, param_name=['weight']):
    offset = 0
    if isinstance(W_shapes, list):
        W_shapes = iter(W_shapes)
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            name_, shape = next(W_shapes)
            if shape is None:
                continue
            assert name_ == name
            numel = W.numel()
            W.data.copy_(W_flat[offset: offset + numel].view(shape))
            offset += numel


def layers_nnz(model, normalized=True, param_name=['weight']):
    res = {}
    count_res = {}
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            layer_name = name
            W_nz = torch.nonzero(W.data)
            if W_nz.dim() > 0:
                if not normalized:
                    res[layer_name] = W_nz.shape[0]
                else:
                    res[layer_name] = float(W_nz.shape[0]) / torch.numel(W)
                count_res[layer_name] = W_nz.shape[0]
            else:
                res[layer_name] = 0
                count_res[layer_name] = 0

    return res, count_res

def layers_unique(model, param_name=['weight'], normalized=True):
    res = {}
    count_res = {}
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            layer_name = name
            W_nz = W.data
            if W_nz.dim() > 0:
                if not normalized:
                    res[layer_name] = W_nz.data.unique().shape[0]
                else:
                    res[layer_name] = float(W_nz.data.unique().shape[0]) / torch.numel(W)
                count_res[layer_name] = W_nz.data.unique().shape[0]
            else:
                res[layer_name] = 0
                count_res[layer_name] = 0
    return res, count_res


def layers_n(model, normalized=True, param_name=['weight']):
    res = {}
    count_res = {}
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            layer_name = name
            W_n = W.data.view(-1)
            if W_n.dim() > 0:
                if not normalized:
                    res[layer_name] = W_n.shape[0]
                else:
                    res[layer_name] = float(W_n.shape[0]) / torch.numel(W)
                count_res[layer_name] = W_n.shape[0]
            else:
                res[layer_name] = 0
                count_res[layer_name] = 0

    return res, count_res


def layers_checkprop(model, param_name=["weight"]):
    for m in model.param_modules():
        print("{} {}".format(type(m), m.weight.shape))


def layers_nnz_LR(model, normalized=False, param_names=["weight"]):
    res = {}
    count_res = {}
    model_size = {}
    for name, W in model.named_parameters():
        layer_name = ".".join(name.strip().split(".")[:-1])
        param_name = name.strip().split(".")[-1]
        if param_name not in param_names:
            continue
        res[layer_name] = res.get(layer_name, {})
        count_res[layer_name] = count_res.get(layer_name, {})
        model_size[layer_name] = model_size.get(layer_name, {"inputs":0, "outputs":0, "ranks": 0})
        if param_name == "weightA":
            model_size[layer_name]["ranks"] = W.shape[0]
            rank = model_size[layer_name]["ranks"]
            W_nz = torch.nonzero(W.view(rank, -1).sum(1)).squeeze()
            if W_nz.dim() == 0:
                res[layer_name]["weightA"] = {"ori-size": W.view(rank, -1).data.shape, "nz_idx": [W_nz.data], "num": 1}
                count_res[layer_name]["weightA"] = 0
                continue
            if not normalized:
                res[layer_name]["weightA"] = {"ori-size": W.view(rank, -1).data.shape, "nz_idx": W_nz, "num": W_nz.shape[0]}
            else:
                res[layer_name]["weightA"] = {"ori-size": W.view(rank, -1).data.shape, "nz_idx": W_nz, "num": float(W_nz.shape[0]) / torch.numel(W.data.sum(0))}
            count_res[layer_name]["weightA"] = W_nz.shape[0]
        if param_name == "weightB":
            model_size[layer_name]["outputs"] = W.shape[0]
            outputs = model_size[layer_name]["outputs"]
            W_nz = torch.nonzero(W.view(outputs, -1).sum(0)).squeeze()
            if W_nz.dim() == 0:
                res[layer_name]["weightB"] = {"ori-size": W.view(outputs, -1).data.shape, "nz_idx": [W_nz.data], "num": 1}
                count_res[layer_name]["weightB"] = 0
                continue
            if not normalized:
                res[layer_name]["weightB"] = {"ori-size": W.view(outputs, -1).data.shape, "nz_idx": W_nz, "num": W_nz.shape[0]}
            else:
                res[layer_name]["weightB"] = {"ori-size": W.view(outputs, -1).data.shape, "nz_idx": W_nz, "num": float(W_nz.shape[0]) / torch.numel(W.data.sum(0))}
            count_res[layer_name]["weightB"] = W_nz.shape[0]
    
    return res, count_res, model_size

def layers_stat(model, param_name='weight'):
    res = "########### layer stat ###########\n"
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            layer_name = name[:-len(param_name) - 1]
            W_nz = torch.nonzero(W.data)
            nnz = W_nz.shape[0] / W.data.numel() if W_nz.dim() > 0 else 0.0
            W_data_abs = W.data.abs()
            res += "{:>20}".format(layer_name) + 'abs(W): min={:.4e}, mean={:.4e}, max={:.4e}, nnz={:.4f}\n'.format(W_data_abs.min().item(), W_data_abs.mean().item(), W_data_abs.max().item(), nnz)

    res += "########### layer stat ###########"
    return res

def l0proj(model, k, normalized=True, param_name=['weightA', "weightB", "weightC"]):
    # get all the weights
    W_shapes = []
    res = []
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            if W.dim() == 1:
                W_shapes.append((name, None))
            else:
                W_shapes.append((name, W.data.shape))
                res.append(W.data.view(-1))
    
    res = torch.cat(res, dim=0)
    if normalized:
        assert 0.0 <= k <= 1.0
        nnz = round(res.shape[0] * k)
    else:
        assert k >= 1 and round(k) == k
        nnz = k
    if nnz == res.shape[0]:
        z_idx = []
    else:
        _, z_idx = torch.topk(torch.abs(res), int(res.shape[0] - nnz), largest=False, sorted=False)
        res[z_idx] = 0.0
        copy_model_weights(model, res, W_shapes, param_name)
    return z_idx, W_shapes


def l0proj_adam(optimizer, model, k, normalized=True, param_name=['weightA', "weightB", "weightC"]):
    eps = optimizer.param_groups[0]['eps']
    # get all the weights
    W_shapes = []
    res = []
    score = []
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            if W.dim() == 1:
                W_shapes.append((name, None))
            else:
                W_shapes.append((name, W.data.shape))
                res.append(W.data.view(-1))
                denom = optimizer.state[W]['exp_avg_sq'].sqrt().add_(eps)
                score.append((W.data.view(-1) ** 2) * denom.view(-1))

    res = torch.cat(res, dim=0)
    score = torch.cat(score, dim=0)
    if normalized:
        assert 0.0 <= k <= 1.0
        nnz = round(res.shape[0] * k)
    else:
        assert k >= 1 and round(k) == k
        nnz = k
    if nnz == res.shape[0]:
        z_idx = []
    else:
        _, z_idx = torch.topk(score, int(res.shape[0] - nnz), largest=False, sorted=False)
        res[z_idx] = 0.0
        copy_model_weights(model, res, W_shapes, param_name)
    return z_idx, W_shapes


def idxproj(model, z_idx, W_shapes, param_name=['weight']):
    assert isinstance(z_idx, torch.LongTensor) or isinstance(z_idx, torch.cuda.LongTensor)
    offset = 0
    i = 0
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            name_, shape = W_shapes[i]
            i += 1
            assert name_ == name
            if shape is None:
                continue
            mask = z_idx >= offset
            mask[z_idx >= (offset + W.numel())] = 0
            z_idx_sel = z_idx[mask]
            if len(z_idx_sel.shape) != 0:
                W.data.view(-1)[z_idx_sel - offset] = 0.0
            offset += W.numel()


class myConv2d(nn.Conv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(myConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        self.h_in = h_in
        self.w_in = w_in
        self.xi = Parameter(torch.LongTensor(1), requires_grad=False)
        self.xi.data[0] = stride
        self.g = Parameter(torch.LongTensor(1), requires_grad=False)
        self.g.data[0] = groups
        self.p = Parameter(torch.LongTensor(1), requires_grad=False)
        self.p.data[0] = padding

    def __repr__(self):
        s = ('{name}({h_in}, {w_in}, {in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class FixHWConv2d(myConv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(FixHWConv2d, self).__init__(h_in, w_in, in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias)

        self.hw = Parameter(torch.LongTensor(2), requires_grad=False)
        self.hw.data[0] = h_in
        self.hw.data[1] = w_in

    def forward(self, input):
        # Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        assert input.size(2) == self.hw.data[0] and input.size(3) == self.hw.data[1], 'input_size={}, but hw={}'.format(
            input.size(), self.hw.data)
        return super(FixHWConv2d, self).forward(input)


class SparseConv2d(myConv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SparseConv2d, self).__init__(h_in, w_in, in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)

        self.input_mask = Parameter(torch.Tensor(in_channels, h_in, w_in))
        self.input_mask.data.fill_(1.0)

    def forward(self, input):
        return super(SparseConv2d, self).forward(input * self.input_mask)


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


class MyLeNet5(nn.Module):
    def __init__(self, conv_class=FixHWConv2d):
        super(MyLeNet5, self).__init__()
        h = 32
        w = 32
        feature_layers = []
        # conv
        feature_layers.append(conv_class(h, w, 1, 6, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)
        # conv
        feature_layers.append(conv_class(h, w, 6, 16, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)

        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16 * 5 * 5)
        x = self.classifier(x)
        return x
