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
    # score_all = torch.cat([p ** 2 for p in param_flats], dim=0) * knapsack_weight_all
    score_all = torch.cat([p.abs() for p in param_flats], dim=0) * knapsack_weight_all
    # print(len(weight_list))
    # print(len(weight_bits))
    # print(len(param_flats))
    # model_size *= 0.99
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

    # indices = torch.randperm(n_clusters - 1)[]
    # indices = np.random.choice(X.numel(), n_clusters)
    # val_dict = X[indices].clone().view(-1)
    tol = 1e-3
    pre_dist = None
    one_hot = torch.zeros(X.shape[0], n_clusters).cuda()
    for t in range(niter):
        # assign codes
        km_dist, km_code = torch.min((X.view(-1, 1) - val_dict) ** 2, dim=1)
        cur_dist = km_dist.sum().item()
        # print(cur_dist)
        if pre_dist is not None and cur_dist > 0 and abs(pre_dist - cur_dist) / cur_dist < tol:
            return val_dict
        # print(t, cur_dist)
        pre_dist = cur_dist
        # update dictonary
        # for c in range(n_clusters):
        #     val_in_c = X[(km_code == c)]
        #     if val_in_c.numel() > 0:
        #         val_dict[c] = val_in_c.mean()
        one_hot = one_hot.zero_()
        one_hot.scatter_(1, km_code.unsqueeze(1), 1)
        # print(one_hot.shape)
        # print(X.shape)
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
        # print('idx1={}'.format(idx[go:go+gs]))
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
                        # print('profit={}, weight={}'.format(profit[del_idx], weight[del_idx]))
                        reduced_group_size[gi] -= 1
        # print('idx2={}'.format(idx[go:go+gs]))
    # greedy algorithm
    R_profit = profit[idx]
    R_d_profit = R_profit.clone()
    R_d_profit[1:] -= R_profit[:-1]
    R_weight = weight[idx]
    R_d_weight = R_weight.clone()
    R_d_weight[1:] -= R_weight[:-1]

    # print('profit={}'.format(profit))
    # print('weight={}'.format(weight))
    # print('gs={}'.format(reduced_group_size))

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
        # pnorm = p.data.norm().item() ** 2
        for nbits in nbits_dict:
            nbits = int(nbits)
            val_dict = get_optim_val_dict(p.data, weight_nnz[i].data, nbits, niter=100)
            val_dicts[nbits] = val_dict
            dist = km_quantize_tensor(p.data, nbits, val_dict=val_dict)[1].sum().item()
            # dist4nbits.append(dist / pnorm)
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
        # weight_list[i].data.copy_(km_quantize_tensor(weight_list[i].data, nbits, val_dict=val_dict)[0])
        offset += mckp_gs[i]
        weight_bits.append(nbits)

    return weight_bits, offered_cluster


class Layer_energy(object):
    def __init__(self, **kwargs):
        super(Layer_energy, self).__init__()
        self.h = kwargs['h'] if 'h' in kwargs else None
        self.w = kwargs['w'] if 'w' in kwargs else None
        self.c = kwargs['c'] if 'c' in kwargs else None
        self.d = kwargs['d'] if 'd' in kwargs else None
        self.xi = kwargs['xi'] if 'xi' in kwargs else None
        self.g = kwargs['g'] if 'g' in kwargs else None
        self.p = kwargs['p'] if 'p' in kwargs else None
        self.m = kwargs['m'] if 'm' in kwargs else None
        self.n = kwargs['n'] if 'n' in kwargs else None
        self.s1 = kwargs['s1'] if 's1' in kwargs else None
        self.s2 = kwargs['s2'] if 's2' in kwargs else None
        self.r = kwargs['r'] if 'r' in kwargs else None
        self.is_conv = True if self.r is not None else False

        if self.h is not None:
            self.h_ = max(0.0, math.floor((self.h + 2.0 * self.p - self.r) / float(self.xi)) + 1)
        if self.w is not None:
            self.w_ = max(0.0, math.floor((self.w + 2.0 * self.p - self.r) / float(self.xi)) + 1)

        self.cached_Xenergy = None

    def get_alpha(self, e_mem, e_cache, e_rf):
        if self.is_conv:
            return e_mem + \
                   (math.ceil((float(self.d) / self.g) / self.n) * (self.r ** 2) / float(self.xi ** 2)) * e_cache + \
                   ((float(self.d) / self.g) * (self.r ** 2) / (self.xi ** 2)) * e_rf
        else:
            if self.c <= default_s1:
                return e_mem + math.ceil(float(self.d) / self.n) * e_cache + float(self.d) * e_rf
            else:
                return math.ceil(float(self.d) / self.n) * e_mem + math.ceil(float(self.d) / self.n) * e_cache + float(
                    self.d) * e_rf

    def get_beta(self, e_mem, e_cache, e_rf, in_cache=None):
        if self.is_conv:
            n = 1 if in_cache else math.ceil(self.h_ * self.w_ / self.m)
            return n * e_mem + math.ceil(self.h_ * self.w_ / self.m) * e_cache + \
                   (self.h_ * self.w_) * e_rf
        else:
            return e_mem + e_cache + e_rf

    def get_gamma(self, e_mem, k=None):
        if self.is_conv:
            rows_per_batch = math.floor(self.s1 / float(k))
            assert rows_per_batch >= self.r
            # print(self.__dict__)
            # print('###########', rows_per_batch, self.s1, k)
            # print('conv input data energy (2):{:.2e}'.format(float(k) * (self.r - 1) * (math.ceil(float(self.h) / (rows_per_batch - self.r + 1)) - 1)))

            return (float(self.d) * self.h_ * self.w_) * e_mem + \
                   float(k) * (self.r - self.xi) * \
                   max(0.0, (math.ceil(float(self.h) / (rows_per_batch - self.r + self.xi)) - 1)) * e_mem
        else:
            return float(self.d) * e_mem

    def get_knapsack_weight_W(self, e_mac, e_mem, e_cache, e_rf, in_cache=None, crelax=False):
        if self.is_conv:
            if crelax:
                # use relaxed computation energy estimation (larger than the real computation energy)
                return self.get_beta(e_mem, e_cache, e_rf, in_cache) + e_mac * self.h_ * self.w_
            else:
                # computation energy will be included in other place
                return self.get_beta(e_mem, e_cache, e_rf, in_cache) + e_mac * 0.0
        else:
            return self.get_beta(e_mem, e_cache, e_rf, in_cache) + e_mac

    def get_knapsack_bound_W(self, e_mem, e_cache, e_rf, X_nnz, k):
        if self.is_conv:
            return self.get_gamma(e_mem, k) + self.get_alpha(e_mem, e_cache, e_rf) * X_nnz
        else:
            return self.get_gamma(e_mem) + self.get_alpha(e_mem, e_cache, e_rf) * X_nnz


def build_energy_info(model, m=default_m, n=default_n, s1=default_s1, s2=default_s2):
    res = {}
    for name, p in model.named_parameters():
        if name.endswith('input_mask'):
            layer_name = name[:-len('input_mask') - 1]
            if layer_name in res:
                res[layer_name]['h'] = p.size()[1]
                res[layer_name]['w'] = p.size()[2]
            else:
                res[layer_name] = {'h': p.size()[1], 'w': p.size()[2]}
        elif name.endswith('.hw'):
            layer_name = name[:-len('hw') - 1]
            if layer_name in res:
                res[layer_name]['h'] = float(p.data[0])
                res[layer_name]['w'] = float(p.data[1])
            else:
                res[layer_name] = {'h': float(p.data[0]), 'w': float(p.data[1])}
        elif name.endswith('.xi'):
            layer_name = name[:-len('xi') - 1]
            if layer_name in res:
                res[layer_name]['xi'] = float(p.data[0])
            else:
                res[layer_name] = {'xi': float(p.data[0])}
        elif name.endswith('.g'):
            layer_name = name[:-len('g') - 1]
            if layer_name in res:
                res[layer_name]['g'] = float(p.data[0])
            else:
                res[layer_name] = {'g': float(p.data[0])}
        elif name.endswith('.p'):
            layer_name = name[:-len('p') - 1]
            if layer_name in res:
                res[layer_name]['p'] = float(p.data[0])
            else:
                res[layer_name] = {'p': float(p.data[0])}
        elif name.endswith('weight'):
            if len(p.size()) == 2 or len(p.size()) == 4:
                layer_name = name[:-len('weight') - 1]
                if layer_name in res:
                    res[layer_name]['d'] = p.size()[0]
                    res[layer_name]['c'] = p.size()[1]
                else:
                    res[layer_name] = {'d': p.size()[0], 'c': p.size()[1]}
                if p.dim() > 2:
                    # (out_channels, in_channels, kernel_size[0], kernel_size[1])
                    assert p.dim() == 4
                    res[layer_name]['r'] = p.size()[2]
        else:
            continue

        res[layer_name]['m'] = m
        res[layer_name]['n'] = n
        res[layer_name]['s1'] = s1
        res[layer_name]['s2'] = s2

    for layer_name in res:
        res[layer_name] = Layer_energy(**(res[layer_name]))
        if res[layer_name].g is not None and res[layer_name].g > 1:
            res[layer_name].c *= res[layer_name].g
    return res


def reset_Xenergy_cache(energy_info):
    for layer_name in energy_info:
        energy_info[layer_name].cached_Xenergy = None
    return energy_info

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
        # print("{} {}".format(name, W.data.shape))
        if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            layer_name = name
            W_nz = torch.nonzero(W.data)
            # print("{} {}".format(layer_name, W.data.shape))
            if W_nz.dim() > 0:
                if not normalized:
                    res[layer_name] = W_nz.shape[0]
                else:
                    # print("{} layer nnz:{}".format(name, torch.nonzero(W.data)))
                    res[layer_name] = float(W_nz.shape[0]) / torch.numel(W)
                count_res[layer_name] = W_nz.shape[0]
            else:
                res[layer_name] = 0
                count_res[layer_name] = 0

    return res, count_res

def layers_unique(model, param_name=['weight'], normalized=True):
    res = {}
    count_res = {}
    # print(param_name)
    for name, W in model.named_parameters():
        # print(name.strip().split(".")[-1])
        if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            # print(name)
            layer_name = name
            # W_nz = torch.nonzero(W.data)
            W_nz = W.data
            # print("{} {}".format(layer_name, W.data.shape))
            if W_nz.dim() > 0:
                if not normalized:
                    res[layer_name] = W_nz.data.unique().shape[0]
                else:
                    # print("{} layer nnz:{}".format(name, torch.nonzero(W.data)))
                    res[layer_name] = float(W_nz.data.unique().shape[0]) / torch.numel(W)
                count_res[layer_name] = W_nz.data.unique().shape[0]
            else:
                res[layer_name] = 0
                count_res[layer_name] = 0
    # print(count_res)
    # print(res)
    return res, count_res

# def layers_unique(model, normalized=True, param_name=['weight']):
#     res = {}
#     count_res = {}
#     for name, W in model.named_parameters():
#         if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
#             layer_name = name
#             # W_nz = torch.nonzero(W.data)
#             W_nz = W.data
#             # print("{} {}".format(layer_name, W.data.shape))
#             if W_nz.dim() > 0:
#                 if not normalized:
#                     res[layer_name] = W_nz.data.unique().shape[0]
#                 else:
#                     # print("{} layer nnz:{}".format(name, torch.nonzero(W.data)))
#                     res[layer_name] = float(W_nz.data.unique().shape[0]) / torch.numel(W)
#                 count_res[layer_name] = W_nz.data.unique().shape[0]
#             else:
#                 res[layer_name] = 0
#                 count_res[layer_name] = 0

#     return res, count_res


def layers_n(model, normalized=True, param_name=['weight']):
    res = {}
    count_res = {}
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            layer_name = name
            # W_nz = torch.nonzero(W.data)
            W_n = W.data.view(-1)
            # print("{} {}".format(layer_name, W.data.shape))
            if W_n.dim() > 0:
                if not normalized:
                    res[layer_name] = W_n.shape[0]
                else:
                    # print("{} layer nnz:{}".format(name, torch.nonzero(W.data)))
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
                # print(W_nz)
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
            # print("Wshape {}".format(W.shape))
            W_nz = torch.nonzero(W.view(outputs, -1).sum(0)).squeeze()
            if W_nz.dim() == 0:
                # print(W.view(outputs, -1).sum(0))

                # print(W.view(outputs, -1).sum(1))
                # print(W_nz)
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
        # if name.endswith(param_name):
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
        # if name.endswith(param_name):
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

# def l0proj(model, k, normalized=True, param_name=['weightA', "weightB", "weightC"]):
#     # get all the weights
#     W_shapes = []
#     W_numel = []
#     res = []
#     for name, W in model.named_parameters():
#         # if name.endswith(param_name):
#         if name.strip().split(".")[-1] in param_name:
#             if W.dim() == 1:
#                 W_shapes.append((name, None))
#             else:
#                 W_shapes.append((name, W.data.shape))
#                 _, w_idx = torch.topk(W.data.view(-1), 1, sorted=False)
#                 W_numel.append((W.data.numel(), w_idx))
#                 res.append(W.data.view(-1))
    
#     res = torch.cat(res, dim=0)
#     if normalized:
#         assert 0.0 <= k <= 1.0
#         nnz = round(res.shape[0] * k)
#     else:
#         assert k >= 1 and round(k) == k
#         nnz = k
#     if nnz == res.shape[0]:
#         z_idx = []
#     else:
#         _, z_idx = torch.topk(torch.abs(res), int(res.shape[0] - nnz), largest=False, sorted=False)
#         offset = 0
#         ttl = res.shape[0]
#         WzeroInd = torch.zeros(ttl)
#         WzeroInd[z_idx] = 1.0
#         for item0, item1 in W_numel:
#             WzeroInd[offset+item1] = 0.0
#             offset += item0
#         z_idx = torch.nonzero(WzeroInd)
#         res[z_idx] = 0.0
#         copy_model_weights(model, res, W_shapes, param_name)
#     return z_idx, W_shapes


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

def conv_cache_overlap(X_supp, padding, kernel_size, stride, k_X):
    rs = X_supp.transpose(0, 1).contiguous().view(X_supp.size(1), -1).sum(dim=1).cpu()
    rs = torch.cat([torch.zeros(padding, dtype=rs.dtype, device=rs.device),
                   rs,
                    torch.zeros(padding, dtype=rs.dtype, device=rs.device)])
    res = 0
    beg = 0
    end = None
    while beg + kernel_size - 1 < rs.size(0):
        if end is not None:
            if beg < end:
                res += rs[beg:end].sum().item()
        n_elements = 0
        for i in range(rs.size(0) - beg):
            if n_elements + rs[beg+i] <= k_X:
                n_elements += rs[beg+i]
                if beg + i == rs.size(0) - 1:
                    end = rs.size(0)
            else:
                end = beg + i
                break
        assert end - beg >= kernel_size, 'can only hold {} rows with {} elements < {} rows in {}, cache size={}'.format(end - beg, n_elements, kernel_size, X_supp.size(), k_X)
        # print('map size={}. begin={}, end={}'.format(X_supp.size(), beg, end))
        beg += (math.floor((end - beg - kernel_size) / stride) + 1) * stride
    return res


def energy_eval(model, energy_info, e_mac=default_e_mac, e_mem=default_e_mem, e_cache=default_e_cache,
                e_rf=default_e_rf, verbose=False):
    X_nnz_dict = layers_nnz(model, normalized=False, param_name='input_mask')

    W_nnz_dict = layers_nnz(model, normalized=False, param_name='weight')

    W_energy = []
    C_energy = []
    X_energy = []
    X_supp_dict = {}
    for name, p in model.named_parameters():
        if name.endswith('input_mask'):
            layer_name = name[:-len('input_mask') - 1]
            X_supp_dict[layer_name] = (p.data != 0.0).float()

    for name, p in model.named_parameters():
        if name.endswith('weight'):
            if p is None or p.dim() == 1:
                continue
            layer_name = name[:-len('weight') - 1]
            einfo = energy_info[layer_name]

            if einfo.is_conv:
                X_nnz = einfo.h * einfo.w * einfo.c
            else:
                X_nnz = einfo.c
            if layer_name in X_nnz_dict:
                # this layer has sparse input
                X_nnz = X_nnz_dict[layer_name]

            if layer_name in X_supp_dict:
                X_supp = X_supp_dict[layer_name].unsqueeze(0)
            else:
                if einfo.is_conv:
                    X_supp = torch.ones(1, int(einfo.c), int(einfo.h), int(einfo.w), dtype=p.dtype, device=p.device)
                else:
                    X_supp = None

            unfoldedX = None

            # input data access energy
            if einfo.is_conv:
                h_, w_ = max(0.0, math.floor((einfo.h + 2 * einfo.p - einfo.r) / einfo.xi) + 1), max(0.0, math.floor((einfo.w + 2 * einfo.p - einfo.r) / einfo.xi) + 1)
                unfoldedX = F.unfold(X_supp, kernel_size=int(einfo.r), padding=int(einfo.p), stride=int(einfo.xi)).squeeze(0)
                assert unfoldedX.size(1) == h_ * w_, 'unfolded X size={}, but h_ * w_ = {}, W.size={}'.format(unfoldedX.size(), h_ * w_, p.size())
                unfoldedX_nnz = (unfoldedX != 0.0).float().sum().item()

                X_energy_cache = unfoldedX_nnz * math.ceil((float(einfo.d) / einfo.g) / einfo.n) * e_cache
                X_energy_rf = unfoldedX_nnz * math.ceil(float(einfo.d) / einfo.g) * e_rf

                X_energy_mem = X_nnz * e_mem + \
                               conv_cache_overlap(X_supp.squeeze(0), int(einfo.p), int(einfo.r), int(einfo.xi), default_s1) * e_mem + \
                               unfoldedX.size(1) * einfo.d * e_mem
                X_energy_this = X_energy_mem + X_energy_rf + X_energy_cache
            else:
                X_energy_cache = math.ceil(float(einfo.d) / einfo.n) * e_cache * X_nnz
                X_energy_rf = float(einfo.d) * e_rf * X_nnz
                X_energy_mem = e_mem * (math.ceil(float(einfo.d) / einfo.n) * max(0.0, X_nnz - default_s1)
                                        + min(X_nnz, default_s1)) + e_mem * float(einfo.d)

                X_energy_this = X_energy_mem + X_energy_rf + X_energy_cache

            einfo.cached_Xenergy = X_energy_this
            X_energy.append(X_energy_this)

            # kernel weights data access energy
            if einfo.is_conv:
                output_hw = unfoldedX.size(1)
                W_energy_cache = math.ceil(output_hw / einfo.m) * W_nnz_dict[layer_name] * e_cache
                W_energy_rf = output_hw * W_nnz_dict[layer_name] * e_rf
                W_energy_mem = (math.ceil(output_hw / einfo.m) * max(0.0, W_nnz_dict[layer_name] - default_s2)\
                               + min(default_s2, W_nnz_dict[layer_name])) * e_mem
                W_energy_this = W_energy_cache + W_energy_rf + W_energy_mem
            else:
                W_energy_this = einfo.get_beta(e_mem, e_cache, e_rf, in_cache=None) * W_nnz_dict[layer_name]
            W_energy.append(W_energy_this)

            # computation enregy
            if einfo.is_conv:
                N_mac = torch.sum(
                    F.conv2d(X_supp, (p.data != 0.0).float(), None, int(energy_info[layer_name].xi),
                             int(energy_info[layer_name].p), 1, int(energy_info[layer_name].g))).item()
                C_energy_this = e_mac * N_mac
            else:
                C_energy_this = e_mac * (W_nnz_dict[layer_name])

            C_energy.append(C_energy_this)

            if verbose:
                print("Layer: {}, W_energy={:.2e}, C_energy={:.2e}, X_energy={:.2e}".format(layer_name,
                                                                                            W_energy[-1],
                                                                                            C_energy[-1],
                                                                                            X_energy[-1]))

    return {'W': sum(W_energy), 'C': sum(C_energy), 'X': sum(X_energy)}


def energy_eval_relax(model, energy_info, e_mac=default_e_mac, e_mem=default_e_mem, e_cache=default_e_cache,
                e_rf=default_e_rf, verbose=False):
    W_nnz_dict = layers_nnz(model, normalized=False, param_name='weight')

    W_energy = []
    C_energy = []
    X_energy = []
    X_supp_dict = {}
    for name, p in model.named_parameters():
        if name.endswith('input_mask'):
            layer_name = name[:-len('input_mask') - 1]
            X_supp_dict[layer_name] = (p.data != 0.0).float()

    for name, p in model.named_parameters():
        if name.endswith('weight'):
            if p is None or p.dim() == 1:
                continue
            layer_name = name[:-len('weight') - 1]
            assert energy_info[layer_name].cached_Xenergy is not None
            X_energy.append(energy_info[layer_name].cached_Xenergy)
            assert X_energy[-1] > 0
            if not energy_info[layer_name].is_conv:
                # in_cache is not needed in fc layers
                in_cache = None
                W_energy.append(
                    energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache) * W_nnz_dict[layer_name])
                C_energy.append(e_mac * (W_nnz_dict[layer_name]))
                if verbose:
                    knapsack_weight1 = energy_info[layer_name].get_knapsack_weight_W(e_mac, e_mem, e_cache, e_rf,
                                                                                     in_cache=None, crelax=True)
                    if hasattr(knapsack_weight1, 'mean'):
                        knapsack_weight1 = knapsack_weight1.mean()
                    print(layer_name + " weight: {:.4e}".format(knapsack_weight1))

            else:
                beta1 = energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=True)
                beta2 = energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=False)

                W_nnz = W_nnz_dict[layer_name]
                W_energy_this = beta1 * min(energy_info[layer_name].s2, W_nnz) + beta2 * max(0, W_nnz - energy_info[
                    layer_name].s2)
                W_energy.append(W_energy_this)
                C_energy.append(e_mac * energy_info[layer_name].h_ * float(energy_info[layer_name].w_) * W_nnz)

            if verbose:
                print("Layer: {}, W_energy={:.2e}, C_energy={:.2e}, X_energy={:.2e}".format(layer_name,
                                                                                            W_energy[-1],
                                                                                            C_energy[-1],
                                                                                            X_energy[-1]))

    return {'W': sum(W_energy), 'C': sum(C_energy), 'X': sum(X_energy)}


def energy_proj(model, energy_info, budget, e_mac=default_e_mac, e_mem=default_e_mem, e_cache=default_e_cache,
                e_rf=default_e_rf, grad=False, in_place=True, preserve=0.0, param_name='weight'):
    knapsack_bound = budget
    param_flats = []
    knapsack_weight_all = []
    score_all = []
    param_shapes = []
    bound_bias = 0.0

    for name, p in model.named_parameters():
        if name.endswith(param_name):
            if p is None or (param_name == 'weight' and p.dim() == 1):
                # skip batch_norm layer
                param_shapes.append((name, None))
                continue
            else:
                param_shapes.append((name, p.data.shape))

            layer_name = name[:-len(param_name) - 1]
            assert energy_info[layer_name].cached_Xenergy is not None
            if grad:
                p_flat = p.grad.data.view(-1)
            else:
                p_flat = p.data.view(-1)
            score = p_flat ** 2

            if param_name == 'weight':
                knapsack_weight = energy_info[layer_name].get_knapsack_weight_W(e_mac, e_mem, e_cache, e_rf,
                                                                                in_cache=True, crelax=True)
                if hasattr(knapsack_weight, 'view'):
                    knapsack_weight = knapsack_weight.view(1, -1, 1, 1)
                knapsack_weight = torch.zeros_like(p.data).add_(knapsack_weight).view(-1)

                # preserve part of weights
                if preserve > 0.0:
                    if preserve > 1:
                        n_preserve = preserve
                    else:
                        n_preserve = round(p_flat.numel() * preserve)
                    _, preserve_idx = torch.topk(score, k=n_preserve, largest=True, sorted=False)
                    score[preserve_idx] = float('inf')

                if energy_info[layer_name].is_conv and p_flat.numel() > energy_info[layer_name].s2:
                    delta = energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=False) \
                            - energy_info[layer_name].get_beta(e_mem, e_cache, e_rf, in_cache=True)
                    assert delta >= 0
                    _, out_cache_idx = torch.topk(score, k=p_flat.numel() - energy_info[layer_name].s2, largest=False,
                                                  sorted=False)
                    knapsack_weight[out_cache_idx] += delta

                bound_const = energy_info[layer_name].cached_Xenergy

                assert bound_const > 0
                bound_bias += bound_const
                knapsack_bound -= bound_const

            else:
                raise ValueError('not supported parameter name')

            score_all.append(score)
            knapsack_weight_all.append(knapsack_weight)
            # print(layer_name, X_nnz, knapsack_weight)
            param_flats.append(p_flat)

    param_flats = torch.cat(param_flats, dim=0)
    knapsack_weight_all = torch.cat(knapsack_weight_all, dim=0)
    score_all = torch.cat(score_all, dim=0) / knapsack_weight_all

    _, sorted_idx = torch.sort(score_all, descending=True)
    cumsum = torch.cumsum(knapsack_weight_all[sorted_idx], dim=0)
    res_nnz = torch.nonzero(cumsum <= knapsack_bound).max()
    z_idx = sorted_idx[-(param_flats.numel() - res_nnz):]

    if in_place:
        param_flats[z_idx] = 0.0
        copy_model_weights(model, param_flats, param_shapes, param_name)
    return z_idx, param_shapes

# energy_info = build_energy_info(model)
# energy_estimator = lambda m: sum(energy_eval(m, energy_info, verbose=False).values())

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
        # print("###{}, {}".format(input.size(), self.input_mask.size()))
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


class MyCaffeLeNet(nn.Module):
    def __init__(self, conv_class=FixHWConv2d):
        super(MyCaffeLeNet, self).__init__()
        h = 28
        w = 28
        feature_layers = []
        # conv
        feature_layers.append(conv_class(h, w, 1, 20, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)
        # conv
        feature_layers.append(conv_class(h, w, 20, 50, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)

        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x

def masked_layers_hw(cfg, c_in, h_in, w_in, use_mask=True, batch_norm=False):
    afun = nn.ReLU()
    layers = []
    for i, v in enumerate(cfg):
        if v == 'M':
            Mstride = 2
            Mkernel = 2
            Mpadding = 0
            h_out = conv2d_out_dim(h_in, padding=Mpadding, kernel_size=Mkernel, stride=Mstride)
            w_out = conv2d_out_dim(w_in, padding=Mpadding, kernel_size=Mkernel, stride=Mstride)

            layers += [nn.MaxPool2d(kernel_size=Mkernel, stride=Mstride)]
        else:
            Cpadding = v[1] if isinstance(v, tuple) else 1
            c_out = v[0] if isinstance(v, tuple) else v
            Ckernel = 3
            h_out = conv2d_out_dim(h_in, padding=Cpadding, kernel_size=Ckernel)
            w_out = conv2d_out_dim(w_in, padding=Cpadding, kernel_size=Ckernel)

            if use_mask or i == 0:
                conv2d = SparseConv2d(h_in, w_in, c_in, c_out, kernel_size=Ckernel, padding=Cpadding)
            else:
                conv2d = FixHWConv2d(h_in, w_in, c_in, c_out, kernel_size=Ckernel, padding=Cpadding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c_out, affine=False), afun]
            else:
                layers += [conv2d, afun]
            c_in = c_out

        h_in = h_out
        w_in = w_out
    return nn.Sequential(*layers)


class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def cifar10(n_channel, masked=False, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = masked_layers_hw(cfg, c_in=3, h_in=32, w_in=32, use_mask=masked, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
    if pretrained is not None:
        assert not masked
        m = model_zoo.load_url(model_urls['cifar10'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


##########################################
################# MobileNetV2 for CIFAR-10
##########################################

def myconv_bn(h_in, w_in, inp, oup, stride, conv_class=FixHWConv2d):
    return Sequential(
        conv_class(h_in, w_in, inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def myconv_1x1_bn(h_in, w_in, inp, oup, conv_class=FixHWConv2d):
    return Sequential(
        conv_class(h_in, w_in, inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class MyInvertedResidual(nn.Module):
    def __init__(self, h_in, w_in, inp, oup, stride, expand_ratio, conv_class=FixHWConv2d):
        super(MyInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        h, w = h_in, w_in
        conv_layers = []

        # pw
        conv_layers.append(conv_class(h, w, inp, inp * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False))
        h = conv2d_out_dim(h, kernel_size=1, stride=1, padding=0)
        w = conv2d_out_dim(w, kernel_size=1, stride=1, padding=0)
        conv_layers.append(nn.BatchNorm2d(inp * expand_ratio))
        conv_layers.append(nn.ReLU6(inplace=True))

        # dw
        conv_layers.append(conv_class(h, w, inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, padding=1, groups=inp * expand_ratio, bias=False))
        h = conv2d_out_dim(h, kernel_size=3, stride=stride, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, stride=stride, padding=1)
        conv_layers.append(nn.BatchNorm2d(inp * expand_ratio))
        conv_layers.append(nn.ReLU6(inplace=True))

        # pw-linear
        conv_layers.append(conv_class(h, w, inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0, bias=False))
        h = conv2d_out_dim(h, kernel_size=1, stride=1, padding=0)
        w = conv2d_out_dim(w, kernel_size=1, stride=1, padding=0)
        conv_layers.append(nn.BatchNorm2d(oup))

        self.conv = Sequential(*conv_layers)

        self.h_out = h
        self.w_out = w

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MyMobileNetV2(nn.Module):
    def __init__(self, n_class=10, input_size=32, width_mult=1., conv_class=FixHWConv2d):
        super(MyMobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size == 32
        h = input_size
        w = input_size
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.features = [myconv_bn(h, w, 3, input_channel, stride=1, conv_class=conv_class)]
        h = conv2d_out_dim(h, kernel_size=3, stride=1, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, stride=1, padding=1)
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    inv_res_block = MyInvertedResidual(h, w, input_channel, output_channel, s, t, conv_class=conv_class)
                else:
                    inv_res_block = MyInvertedResidual(h, w, input_channel, output_channel, 1, t, conv_class=conv_class)

                h = inv_res_block.h_out
                w = inv_res_block.w_out
                self.features.append(inv_res_block)
                input_channel = output_channel
        # building last several layers
        self.features.append(myconv_1x1_bn(h, w, input_channel, self.last_channel, conv_class=conv_class))
        self.features.append(nn.AvgPool2d(kernel_size=4))
        # make it nn.Sequential
        self.features = Sequential(*self.features)

        # building classifier
        self.classifier = Sequential(
            # nn.Dropout(p=0.5 if dropout else 0.0), # NOTE: disable dropout
            nn.Linear(self.last_channel, n_class),
        )

        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x


######################################################
#####################Low Rank Layer###################
######################################################

class Linearlr(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(Linearlr, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        print("rank {}, in_features {}, out_features {}".format(rank, in_features, out_features))
        assert rank <= min(in_features, out_features)
        self.rank = rank
        self.weightA = Parameter(torch.Tensor(rank, in_features))
        self.weightB = Parameter(torch.Tensor(out_features, rank))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weightA.size(1))
        self.weightA.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weightB.size(1))
        self.weightB.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input.matmul(self.weightA.t()), self.weightB, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2dlr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dlr, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        assert groups == 1, 'does not support grouped convolution yet'
        assert rank <= min(in_channels * kernel_size[0] * kernel_size[1], out_channels)
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weightA = Parameter(torch.Tensor(rank, in_channels, *kernel_size))
        self.weightB = Parameter(torch.Tensor(out_channels, rank, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weightA.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        n = self.rank
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weightB.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, rank={rank}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def forward(self, input):
        return F.conv2d(F.conv2d(input, self.weightA, None, self.stride, self.padding, self.dilation, self.groups),
                        self.weightB, self.bias, 1, 0, 1, 1)


class MobileNetV2CifarLR(nn.Module):
    @staticmethod
    def conv_bn_lr(inp, oup, stride, rank):
        return nn.Sequential(
            Conv2dlr(inp, oup, 3, rank=rank, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

    @staticmethod
    def conv_1x1_bn_lr(inp, oup, rank):
        return nn.Sequential(
            Conv2dlr(inp, oup, 1, rank=rank, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

    class InvertedResidualLR(nn.Module):
        def __init__(self, inp, oup, stride, expand_ratio, ranks):
            super().__init__()
            self.stride = stride
            assert stride in [1, 2]

            self.use_res_connect = self.stride == 1 and inp == oup

            conv_layers = []

            # pw
            conv_layers.append(
                Conv2dlr(inp, inp * expand_ratio, kernel_size=1, rank=ranks[0], stride=1, padding=0, bias=False))

            conv_layers.append(nn.BatchNorm2d(inp * expand_ratio))
            conv_layers.append(nn.ReLU6(inplace=True))

            # dw
            conv_layers.append(
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, padding=1,
                          groups=inp * expand_ratio, bias=False))

            conv_layers.append(nn.BatchNorm2d(inp * expand_ratio))
            conv_layers.append(nn.ReLU6(inplace=True))

            # pw-linear
            conv_layers.append(
                Conv2dlr(inp * expand_ratio, oup, kernel_size=1, rank=ranks[1], stride=1, padding=0, bias=False))

            conv_layers.append(nn.BatchNorm2d(oup))

            self.conv = nn.Sequential(*conv_layers)

        def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

    def __init__(self, n_class=10, input_size=32, width_mult=1., ranks=None):
        super(MobileNetV2CifarLR, self).__init__()
        if ranks is None:
            ranks = MobileNetV2Cifar().get_ranks()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size == 32
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        ridx = 0
        self.features = [self.conv_bn_lr(3, input_channel, stride=1, rank=ranks[ridx])]
        ridx += 1
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    inv_res_block = self.InvertedResidualLR(input_channel, output_channel, s, t, ranks=(ranks[ridx], ranks[ridx+1]))
                    ridx += 2
                else:
                    inv_res_block = self.InvertedResidualLR(input_channel, output_channel, 1, t, ranks=(ranks[ridx], ranks[ridx+1]))
                    ridx += 2
                self.features.append(inv_res_block)
                input_channel = output_channel
        # building last several layers
        self.features.append(self.conv_1x1_bn_lr(input_channel, self.last_channel, ranks[ridx]))
        ridx += 1
        self.features.append(nn.AvgPool2d(kernel_size=4))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        print("ridx {} rank_length {}".format(ridx, len(ranks)))
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5 if dropout else 0.0), # NOTE: disable dropout
            Linearlr(self.last_channel, n_class, ranks[ridx]),
        )
        ridx += 1

        assert ridx == len(ranks), 'low rank layer number: {} != {}'.format(ridx, len(ranks))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def set_weights(self, weights_list):
        i = 0
        for m in self.factorized_modules():
            weightA, weightB = weights_list[i]
            m.weightA.data.view(-1).copy_(weightA.view(-1))
            m.weightB.data.view(-1).copy_(weightB.view(-1))
            i += 1
    
    def factorized_modules(self):
        for module in self.modules():
            if isinstance(module, Conv2dlr) or isinstance(module, Linearlr):
                yield module


class MobileNetV2Cifar(nn.Module):
    @staticmethod
    def conv_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

    @staticmethod
    def conv_1x1_bn(inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

    class InvertedResidual(nn.Module):
        def __init__(self, inp, oup, stride, expand_ratio):
            super().__init__()
            self.stride = stride
            assert stride in [1, 2]

            self.use_res_connect = self.stride == 1 and inp == oup

            conv_layers = []

            # pw
            conv_layers.append(nn.Conv2d(inp, inp * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False))

            conv_layers.append(nn.BatchNorm2d(inp * expand_ratio))
            conv_layers.append(nn.ReLU6(inplace=True))

            # dw
            conv_layers.append(
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, padding=1,
                          groups=inp * expand_ratio, bias=False))

            conv_layers.append(nn.BatchNorm2d(inp * expand_ratio))
            conv_layers.append(nn.ReLU6(inplace=True))

            # pw-linear
            conv_layers.append(nn.Conv2d(inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0, bias=False))

            conv_layers.append(nn.BatchNorm2d(oup))

            self.conv = nn.Sequential(*conv_layers)

        def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

    def __init__(self, n_class=10, input_size=32, width_mult=1.):
        super(MobileNetV2Cifar, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size == 32
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.features = [self.conv_bn(3, input_channel, stride=1)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    inv_res_block = self.InvertedResidual(input_channel, output_channel, s, t)
                else:
                    inv_res_block = self.InvertedResidual(input_channel, output_channel, 1, t)

                self.features.append(inv_res_block)
                input_channel = output_channel
        # building last several layers
        self.features.append(self.conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(kernel_size=4))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5 if dropout else 0.0), # NOTE: disable dropout
            nn.Linear(self.last_channel, n_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def param_modules(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                yield module

    def get_ranks(self):
        ranks = []
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d):
                if m.groups != 1:
                    continue
                c, d = m.in_channels, m.out_channels
                rank = min(c * m.kernel_size[0] * m.kernel_size[1], d)
            else:
                c, d = m.in_features, m.out_features
                rank = min(c, d)
            ranks.append(rank)
        return ranks

    def factorize_weights(self, ranks):
        weights_list = []
        i = 0
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                # print('m.weight {}'.format(m.weight.size()))
                U, S, V = torch.svd(m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data)
            else:
                U, S, V = torch.svd(m.weight.data)

            S, indexes = torch.sort(S, descending=True)
            # print(indexes)
            # print(S)

            S_sqrt = torch.sqrt(S[:ranks[i]])
            weightB = U[:, :ranks[i]] * S_sqrt
            weightA = S_sqrt.view(-1, 1) * V.t()[:ranks[i], :]
            weights_list.append((weightA, weightB))
            # print(torch.dist(m.weight.data.view(-1), weightB.mm(weightA).view(-1)))
            i += 1
        return weights_list
