import torch.nn as nn
import torch
# from utee import misc
from torch.nn.modules.utils import _pair
import scipy.linalg as sl
import torch.nn.functional as F
from torch.nn import Parameter, Sequential
from utee.kmeans import lloyd_nnz_fixed_0_center
from torch.autograd.function import InplaceFunction, Function

import math

class DeepOriginalModel(nn.Module):
    def __init__(self):
        super(DeepOriginalModel, self).__init__()

    def param_modules(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                yield module

    def named_param_modules(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                yield name, module

    def empty_all(self):
        for m in self.param_modules():
            m.weight.data.fill_(0.)

    def replace_bias(self, model_src, weight_name):
        src_dict = dict(model_src.named_parameters())
        src_buffer_dict = dict(model_src.named_buffers())
        for n, w in self.named_parameters():
            if n.split(".")[-1] not in weight_name or n.split(".")[-2][:2] == "bn":
                w.data.copy_(src_dict[n].data)

        for n, buffer in self.named_buffers():
            buffer.data.copy_(src_buffer_dict[n].data)
                

    def duplicate_plus(self, extra_w, extra_u):
        extra_ws = dict(extra_w.named_param_modules())
        extra_us = dict(extra_u.named_param_modules())
        for n, m in self.named_param_modules():
            m.weight.data.copy_(extra_ws[n].weight.data + extra_us[n].weight.data)

    def duplicate_update(self, extra_w, extra_z, rho):
        extra_ws = dict(extra_w.named_param_modules())
        extra_zs = dict(extra_z.named_param_modules())
        flag = True
        b = 0
        c = 0
        for n, m in self.named_param_modules():
            m.weight.data.copy_(m.weight.data + (extra_ws[n].weight.data - extra_zs[n].weight.data))


    def admm_regularizer(self, extra_u, extra_z):
        extra_us = dict(extra_u.named_param_modules())
        extra_zs = dict(extra_z.named_param_modules())
        loss = 0
        for n, m in self.named_param_modules():
            extra_us[n].weight.requires_grad = False
            extra_zs[n].weight.requires_grad = False
            loss += ((m.weight + extra_us[n].weight.data - extra_zs[n].weight.data)**2).sum()
        
        return loss

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

    @staticmethod
    def scale_tosame(u, v, w):
        print("dist0 {}".format(torch.norm(v.matmul(u) - w)))
        uscale = torch.sqrt(torch.mean((u * u).view(-1)))
        vscale = torch.sqrt(torch.mean((v * v).view(-1)))
        wscale = torch.sqrt(torch.mean((w * w).view(-1)))
        print("scale pre, U {}, V{}".format(uscale, vscale))
        u = u / uscale * torch.sqrt(uscale * vscale)
        v = v / vscale * torch.sqrt(vscale * uscale)
        print("scale now, U {}, V{}".format(torch.sqrt(torch.mean((u * u).view(-1))), torch.sqrt(torch.mean((v * v).view(-1)))))
        print("dist1 {}".format(torch.norm(v.matmul(u) - w)))
        t = (5 ** (0.5) - 1) / (2.)
        alpha = (1-t) ** 0.5
        print("t {}, alpha {}".format(t, alpha))
        print("dist2 {}".format(torch.norm(v.matmul(u) * alpha * alpha - w * (1-t))))
        print("scale now, U {}, V {}, C {}".format(
            torch.sqrt(torch.mean((u * u * alpha **4).view(-1))), 
            torch.sqrt(torch.mean((v * v * alpha **4).view(-1))),
            torch.sqrt(torch.mean((w * w * t **2).view(-1)))
        ))
        return u * alpha, v * alpha, w * t

    def raw_weights(self, ranks):
        weights_list=[]
        i = 0
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            elif isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
                if ranks[i] == m.in_channels * m.kernel_size[0] * m.kernel_size[1]:
                    weightA = None
                    weightB = weight
                    weightC = None
                else:
                    weightA = weight
                    weightB = None
                    weightC = None
            else:
                weight = m.weight.data
                if ranks[i] == m.in_features:
                    weightA = None
                    weightB = weight
                    weightC = None
                else:
                    weightA = weight
                    weightB = None
                    weightC = None
        
            i+= 1
            
            weights_list.append((weightA, weightB, weightC))
        return weights_list

    def svd_weights(self, ranks=None):
        weights_list = []
        i = 0
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
                U, S, V = torch.svd(weight)
            else:
                weight = m.weight.data
                U, S, V = torch.svd(weight)
            
            if ranks is None:
                S_sqrt = torch.sqrt(S)
                weight_B = U * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()
            else:
                S_sqrt = torch.sqrt(S[:ranks[i]])
                weight_B = U[:, :ranks[i]] * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()[:ranks[i], :]
            
            weightA, weightB, weightC = self.scale_tosame(weight_A, weight_B, weight)
            restore_w = weightB.matmul(weightA) + weightC

            print("dist {}".format(torch.norm(restore_w - weight)))
            weights_list.append((weightA, weightB, weightC))
            i += 1
        return weights_list

    def svd_weights_v2(self, ranks=None):
        weights_list = []
        i = 0
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
            else:
                weight = m.weight.data
            
            t = 0.0
            if torch.cuda.is_available():
                C = t * weight * (torch.cuda.FloatTensor(weight.size()).uniform_() > 0.5).type(torch.FloatTensor)
            else:
                C = t * weight * (torch.FloatTensor(weight.size()).uniform_() > 0.5).type(torch.FloatTensor)
            
            U, S, V = torch.svd(weight - C)
            
            if ranks is None:
                S_sqrt = torch.sqrt(S)
                weight_B = U * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()
            else:
                S_sqrt = torch.sqrt(S[:ranks[i]])
                weight_B = U[:, :ranks[i]] * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()[:ranks[i], :]
            
            u, v = weight_A, weight_B
            uscale = torch.sqrt(torch.mean((u * u).view(-1)))
            vscale = torch.sqrt(torch.mean((v * v).view(-1)))
            u = u / uscale * torch.sqrt(uscale * vscale)
            v = v / vscale * torch.sqrt(vscale * uscale)
            weightA, weightB, weightC = u, v, C
            restore_w = weightB.matmul(weightA) + weightC

            print("dist {}".format(torch.norm(restore_w - weight)))
            weights_list.append((weightA, weightB, weightC))
            i += 1
        return weights_list


    def svd_weights_v3(self, ranks):
        weights_list = []
        i = 0
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
            else:
                weight = m.weight.data
            
            t = 0.0
            if torch.cuda.is_available():
                C = t * weight * (torch.cuda.FloatTensor(weight.size()).uniform_() > 0.5).type(torch.FloatTensor)
            else:
                C = t * weight * (torch.FloatTensor(weight.size()).uniform_() > 0.5).type(torch.FloatTensor)
            
            U, S, V = torch.svd(weight - C)
            
            S_sqrt = torch.sqrt(S)
            weight_B = U * S_sqrt
            weight_A = S_sqrt.view(-1, 1) * V.t()
            
            if torch.cuda.is_available():
                eye_term = torch.eye(ranks[i])
            else:
                eye_term = torch.eye(ranks[i])

            u, v = weight_A, weight_B
            
            uscale = torch.sqrt(torch.mean((u * u).view(-1)))
            vscale = torch.sqrt(torch.mean((v * v).view(-1)))
            u = u / uscale * torch.sqrt(uscale * vscale)
            v = v / vscale * torch.sqrt(vscale * uscale)
            if (isinstance(m, nn.Conv2d) and ranks[i] == m.in_channels * m.kernel_size[0] * m.kernel_size[1]) or (isinstance(m, nn.Linear) and ranks[i] == m.in_features):
                u = u - eye_term
                restore_w = v.matmul(u + eye_term) + C
            else:
                v = v - eye_term
                restore_w = (v+eye_term).matmul(u) + C
            weightA, weightB, weightC = u, v, C

            print("dist {}".format(torch.norm(restore_w - weight)))
            weights_list.append((weightA, weightB, weightC))
            i += 1
        return weights_list

    def svd_lowrank_weights(self, ranks=None):
        weights_list = []
        i = 0
        for m in self.param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
            else:
                weight = m.weight.data
            
            U, S, V = torch.svd(weight)
            
            if ranks is None:
                S_sqrt = torch.sqrt(S)
                weight_B = U * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()
            else:
                S_sqrt = torch.sqrt(S[:ranks[i]])
                weight_B = U[:, :ranks[i]] * S_sqrt
                weight_A = S_sqrt.view(-1, 1) * V.t()[:ranks[i], :]
            
            restore_w = weight_B.matmul(weight_A)

            print("dist {}".format(torch.norm(restore_w - weight)))
            weights_list.append((weight_A, weight_B))
            i += 1
        return weights_list

    def svd_global_lowrank_weights(self, k):
        weights_list = []
        i = 0
        res = []
        resU = []
        resV = []
        Sshapes = []
        orig_weights = []
        name_list = []
        for name, m in self.named_param_modules():
            if isinstance(m, nn.Conv2d) and m.groups != 1:
                continue
            if isinstance(m, nn.Conv2d):
                weight = m.weight.view(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]).data
            else:
                weight = m.weight.data
            orig_weights.append(weight)
            U, S, V = torch.svd(weight)
            Sshapes.append(S.shape)

            res.append(S)
            resU.append(U)
            resV.append(V)

            name_list.append(name)
        
        res1 = torch.cat(res, dim=0)
        _, z_idx = torch.topk(torch.abs(res1), int(res1.shape[0] * (1-k)), largest=False, sorted=False)
        res1[z_idx] = 0.0
        offset = 0
        ranks = []
        for i in range(len(res)):
            S, U, V = res[i], resU[i], resV[i]
            numel = S.numel()
            nnz_idx = torch.nonzero(res1[offset: offset+numel])
            rank = nnz_idx.shape[0]
            print(rank)
            if rank ==0:
                rank = 1
            ranks.append(rank)
            S_sqrt = torch.sqrt(S[:rank])
            weight_B = U[:, :rank] * S_sqrt
            weight_A = S_sqrt.view(-1, 1) * V.t()[:rank, :]
            print("{} {} {}".format(S_sqrt.shape, weight_B.shape, weight_A.shape))
            restore_w = weight_B.matmul(weight_A)

            print("dist {}".format(torch.norm(restore_w - orig_weights[i])))
            weights_list.append((name_list[i], weight_A, weight_B))
            offset += numel
        return weights_list, ranks

    def update_weight_bit(self, weight_bits):
        offset = 0
        for module in self.modules():
            if isinstance(module, Linear_QuantForward) or isinstance(module, Conv2d_QuantForward):
                module.bit = weight_bits[offset]
                offset += 1

#######################################################################
#########################-low_rank-######################
#######################################################################

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
        # print("B {}, A{}, C{}".format(self.weightB.size(), self.weightA.size(), self.weightC.size()))
        weight = self.weightB.matmul(self.weightA)
        return F.linear(input, weight, self.bias)
        # return F.linear(input.matmul(self.weightA.t()), self.weightB, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2dlr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1, padding=0, dilation=1, groups=1, bias=True):
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
        self.weight_size = [out_channels, in_channels, *kernel_size]
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
        # print("B {}, A{}, C{}".format(self.weightB.size(), self.weightA.size(), self.weightC.size()))
        # print(self.extra_repr())
        # print(input[0][0][0])
        weightA = self.weightA.view(self.rank, -1)
        weightB = self.weightB.view(self.out_channels, -1)
        weight = weightB.matmul(weightA).view(self.weight_size)
        # exit(0)
        # print("B {}, A {}, C {}, W {}".format(weightB.data.cpu()[0][0][0], weightA.data.cpu()[0][0][0], self.weightC.data.cpu()[0][0][0], weight.data.cpu()[0][0][0]))
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class CaffeLeNetLR(nn.Module):
    def __init__(self, ranks):
        super(CaffeLeNetLR, self).__init__()
        feature_layers = []
        # conv
        # feature_layers.append(conv_class(h, w, 1, 20, kernel_size=5))
        feature_layers.append(Conv2dlr(in_channels=1, out_channels=20, kernel_size=5, rank=ranks[0]))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # conv
        # feature_layers.append(conv_class(h, w, 20, 50, kernel_size=5))
        feature_layers.append(Conv2dlr(in_channels=20, out_channels=50, kernel_size=5, rank=ranks[1]))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            Linearlr(50 * 4 * 4, 500, rank=ranks[2]),
            nn.ReLU(inplace=True),
            Linearlr(500, 10, rank=ranks[3]),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        return x

    def set_weights(self, weights_list):
        i = 0
        for m in self.factorized_modules():
            _, weightA, weightB = weights_list[i]
            print(m.weightA.shape)
            print(weightA.shape)
            print("A nan? {}".format(weightA.data != weightA.data))
            print("B nan? {}".format(weightB.data != weightB.data))
            m.weightA.data.view(-1).copy_(weightA.contiguous().view(-1))
            m.weightB.data.view(-1).copy_(weightB.contiguous().view(-1))
            i += 1
    
    def factorized_modules(self):
        for module in self.modules():
            if isinstance(module, Conv2dlr) or isinstance(module, Linearlr):
                yield module

class DeepLRModel(nn.Module):
    def __init__(self):
        super(DeepLRModel, self).__init__()

    def param_modules(self):
        for module in self.modules():
            if isinstance(module, Conv2dlr) or isinstance(module, Linearlr):
                yield module

    def named_param_modules(self):
        for name, module in self.named_modules():
            if isinstance(module, Conv2dlr) or isinstance(module, Linearlr):
                yield name, module

    def set_weights(self, weights_list):
        i = 0
        for name, m in self.factorized_named_modules():
            name_out, weightA, weightB = weights_list[i]
            print("{} {} {} {} {}".format(name_out, weightA.shape, weightB.shape, m.weightA.shape, m.weightB.shape))
            # m.weightA.data.view(-1).copy_(weightA.view(-1))
            # m.weightB.data.view(-1).copy_(weightB.view(-1))
            print("A nan? {}".format((weightA.data != weightA.data).sum().item()))
            print("A nan? {}".format(torch.isnan(weightA).sum().item()))
            print("B nan? {}".format((weightB.data != weightB.data).sum().item()))
            print("B nan? {}".format(torch.isnan(weightB).sum().item()))
            m.weightA.data.copy_(weightA.view(m.weightA.data.shape))
            m.weightB.data.copy_(weightB.view(m.weightB.data.shape))
            i += 1
    
    def factorized_modules(self):
        for module in self.modules():
            if isinstance(module, Conv2dlr) or isinstance(module, Linearlr):
                yield module

    def factorized_named_modules(self):
        for name, module in self.named_modules():
            if isinstance(module, Conv2dlr) or isinstance(module, Linearlr):
                yield name, module

    def empty_all(self):
        for m in self.param_modules():
            m.weightA.data.fill_(0.)
            m.weightB.data.fill_(0.)

    def replace_bias(self, model_src, weight_name):
        src_dict = dict(model_src.named_parameters())
        src_buffer_dict = dict(model_src.named_buffers())
        for n, w in self.named_parameters():
            # print(src_dict.keys())
            if n.split(".")[-1] not in weight_name or n.split(".")[-2][:2] == "bn":
                # print(n.split(".")[-2][:2])
                w.data.copy_(src_dict[n].data)
            # print(n)
            # print(n.strip())
        for n, buffer in self.named_buffers():
            buffer.data.copy_(src_buffer_dict[n].data)

                

    def duplicate_plus(self, extra_w, extra_u):
        extra_ws = dict(extra_w.factorized_named_modules())
        extra_us = dict(extra_u.factorized_named_modules())
        for n, m in self.factorized_named_modules():
            m.weightA.data.copy_(extra_ws[n].weightA.data + extra_us[n].weightA.data)
            m.weightB.data.copy_(extra_ws[n].weightB.data + extra_us[n].weightB.data)

    def duplicate_update(self, extra_w, extra_z, rho):
        rho = 1.
        extra_ws = dict(extra_w.factorized_named_modules())
        extra_zs = dict(extra_z.factorized_named_modules())
        flag = True
        b = 0
        c = 0
        for n, m in self.factorized_named_modules():
            m.weightA.data.copy_(m.weightA.data + rho * (extra_ws[n].weightA.data - extra_zs[n].weightA.data))
            m.weightB.data.copy_(m.weightB.data + rho * (extra_ws[n].weightB.data - extra_zs[n].weightB.data))
        #     b += (m.weight.data ** 2).sum().item()
        #     c += ((extra_ws[n].weight.data - extra_zs[n].weight.data)**2).sum().item()
        # print(b)
        # print(c)
            # if flag:
            #     print((m.weight.data ** 2).sum().item())
            #     flag = False

    def admm_regularizer(self, extra_u, extra_z):
        extra_us = dict(extra_u.factorized_named_modules())
        extra_zs = dict(extra_z.factorized_named_modules())
        loss = 0
        for n, m in self.factorized_named_modules():
            extra_us[n].weightA.requires_grad = False
            extra_zs[n].weightA.requires_grad = False
            extra_us[n].weightB.requires_grad = False
            extra_zs[n].weightB.requires_grad = False
            loss += ((m.weightA + extra_us[n].weightA.data - extra_zs[n].weightA.data)**2).sum()
            loss += ((m.weightB + extra_us[n].weightB.data - extra_zs[n].weightB.data)**2).sum()
        
        return loss

#######################################################################
#########################-sp_1-######################
#######################################################################

class Linearsp(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(Linearsp, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        print("rank {}, in_features {}, out_features {}".format(rank, in_features, out_features))
        assert rank <= min(in_features, out_features)
        self.rank = rank
        self.weightA = Parameter(torch.Tensor(rank, in_features))
        self.weightB = Parameter(torch.Tensor(out_features, rank))
        self.weightC = Parameter(torch.Tensor(out_features, in_features))
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

        stdv = 1. / math.sqrt(self.weightC.size(1))
        self.weightC.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # print("B {}, A{}, C{}".format(self.weightB.size(), self.weightA.size(), self.weightC.size()))
        weight = self.weightB.matmul(self.weightA) + self.weightC
        return F.linear(input, weight, self.bias)
        # return F.linear(input.matmul(self.weightA.t()), self.weightB, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2dsp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dsp, self).__init__()
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
        self.weightC = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
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

        n = self.in_channels * self.out_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weightC.data.uniform_(-stdv, stdv)

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
        weightA = self.weightA.view(self.rank, -1)
        weightB = self.weightB.view(self.out_channels, -1)
        weight = weightB.matmul(weightA).view(self.weightC.size()) + self.weightC
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        

class DeepSPModel(nn.Module):
    def __init__(self):
        super(DeepSPModel, self).__init__()

    def set_weights(self, weights_list):
        i = 0
        for m in self.factorized_modules():
            weightA, weightB, weightC = weights_list[i]
            m.weightA.data.view(-1).copy_(weightA.view(-1))
            m.weightB.data.view(-1).copy_(weightB.view(-1))
            m.weightC.data.view(-1).copy_(weightC.view(-1))
            i += 1
    
    def factorized_modules(self):
        for module in self.modules():
            if isinstance(module, Conv2dsp) or isinstance(module, Linearsp):
                yield module

#######################################################################
#########################-SP_2-######################
#######################################################################

class Linearsp_v2(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(Linearsp_v2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # print("rank {}, in_features {}, out_features {}".format(rank, in_features, out_features))
        assert rank <= min(in_features, out_features)
        self.rank = rank
        self.weightA = Parameter(torch.zeros(rank, in_features))
        self.weightB = Parameter(torch.zeros(out_features, rank))
        self.weightC = Parameter(torch.zeros(out_features, in_features))
        # if torch.cuda.is_available():
        #     self.eye = torch.eye(rank).cuda()
        # else:
        #     self.eye = torch.eye(rank)
        self.eye = torch.eye(rank)
        self.register_buffer('eye_const', self.eye)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weightA.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.rank == self.in_features:
            weight = self.weightB.matmul(self.weightA + self.eye_const) + self.weightC
        else:
            weight = (self.weightB + self.eye_const).matmul(self.weightA) + self.weightC
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2dsp_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dsp_v2, self).__init__()
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
        self.weightA = Parameter(torch.zeros(rank, in_channels, *kernel_size))
        self.weightB = Parameter(torch.zeros(out_channels, rank, 1, 1))
        self.weightC = Parameter(torch.zeros(out_channels, in_channels, *kernel_size))
        # if torch.cuda.is_available():
        #     self.eye = torch.eye(rank).cuda()
        # else:
        #     self.eye = torch.eye(rank)
        self.eye = torch.eye(rank)
        self.register_buffer('eye_const', self.eye)
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
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

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
        weightA = self.weightA.view(self.rank, -1)
        weightB = self.weightB.view(self.out_channels, -1)
        if self.rank == self.in_channels * self.kernel_size[0] * self.kernel_size[1]:
            # print("{} {} {} {}".format(weightB.type(), weightA.type(), self.eye.type(), self.weightC.type()))
            weight = weightB.matmul(weightA + self.eye_const).view(self.weightC.size()) + self.weightC
        else:
            weight = (weightB + self.eye_const).matmul(weightA).view(self.weightC.size()) + self.weightC
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        

class DeepSP_v2Model(nn.Module):
    def __init__(self):
        super(DeepSP_v2Model, self).__init__()

    def set_weights(self, weights_list):
        i = 0

        for m in self.factorized_modules():
            weightA, weightB, weightC = weights_list[i]
            # print(weights_list[i])
            if not(weightA is None):
                print("setA")
                m.weightA.data.view(-1).copy_(weightA.view(-1))

            if not(weightB is None):
                print("setB")
                m.weightB.data.view(-1).copy_(weightB.view(-1))
            # m.weightC.data.view(-1).copy_(weightC.view(-1))
            i += 1
    
    def factorized_modules(self):
        for module in self.modules():
            if isinstance(module, Conv2dsp_v2) or isinstance(module, Linearsp_v2):
                yield module

    def named_factorized_modules(self):
        for name, module in self.named_modules():
            if isinstance(module, Conv2dsp_v2) or isinstance(module, Linearsp_v2):
                yield name, module

    def replace_bias(self, model_src, weight_name):
        src_dict = dict(model_src.named_parameters())
        src_buffer_dict = dict(model_src.named_buffers())
        for n, w in self.named_parameters():
            # print(src_dict.keys())
            if n.split(".")[-1] not in weight_name or n.split(".")[-2][:2] == "bn":
                # print(n.split(".")[-2][:2])
                w.data.copy_(src_dict[n].data)
            # print(n)
            # print(n.strip())
        for n, buffer in self.named_buffers():
            buffer.data.copy_(src_buffer_dict[n].data)

    def empty_all(self):
        for m in self.factorized_modules():
            m.weightA.data.fill_(0.)
            m.weightB.data.fill_(0.)
            m.weightC.data.fill_(0.)

    def duplicate_plus(self, extra_w, extra_u):
        extra_ws = dict(extra_w.named_factorized_modules())
        extra_us = dict(extra_u.named_factorized_modules())
        for n, m in self.named_factorized_modules():
            m.weightA.data.copy_(extra_ws[n].weightA.data + extra_us[n].weightA.data)
            m.weightB.data.copy_(extra_ws[n].weightB.data + extra_us[n].weightB.data)
            m.weightC.data.copy_(extra_ws[n].weightC.data + extra_us[n].weightC.data)

    def duplicate_update(self, extra_w, extra_z, rho):
        rho = 1.
        extra_ws = dict(extra_w.named_factorized_modules())
        extra_zs = dict(extra_z.named_factorized_modules())
        for n, m in self.named_factorized_modules():
            m.weightA.data.copy_(m.weightA.data + rho * (extra_ws[n].weightA.data - extra_zs[n].weightA.data) )
            m.weightB.data.copy_(m.weightB.data + rho * (extra_ws[n].weightB.data - extra_zs[n].weightB.data) )
            m.weightC.data.copy_(m.weightC.data + rho * (extra_ws[n].weightC.data - extra_zs[n].weightC.data) )

    def admm_regularizer(self, extra_u, extra_z):
        extra_us = dict(extra_u.named_factorized_modules())
        extra_zs = dict(extra_z.named_factorized_modules())
        loss = 0
        for n, m in self.named_factorized_modules():
            extra_us[n].weightA.requires_grad = False
            extra_zs[n].weightA.requires_grad = False
            loss += ((m.weightA + extra_us[n].weightA.data - extra_zs[n].weightA.data)**2).sum()
            extra_us[n].weightB.requires_grad = False
            extra_zs[n].weightB.requires_grad = False
            loss += ((m.weightB + extra_us[n].weightB.data - extra_zs[n].weightB.data)**2).sum()
            extra_us[n].weightC.requires_grad = False
            extra_zs[n].weightC.requires_grad = False
            loss += ((m.weightC + extra_us[n].weightC.data - extra_zs[n].weightC.data)**2).sum()

        return loss
