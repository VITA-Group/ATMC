import torch
from skimage.io import imsave
import os 
import numpy as np
from kmeans import lloyd, lloyd_nnz, lloyd_fixed_nnz, lloyd_nnz_fixed_0_center

def print_params(model):
	for i, (param_name, W) in enumerate(model.named_parameters()):
		# print(param_name, W.size(), type(W))
		if i == 0:
			print(W[0,0,:,:].data.cpu().numpy())

def print_params_by_name(model, param_idx=0, save_file_name=None):
	for i, (param_name, W) in enumerate(model.named_parameters()):
		print(param_name)
		print(W.size())
		array = W.flatten(start_dim=1, end_dim=-1).detach().numpy()
		print(array.shape)
		array -= np.amin(array)
		array /= np.amax(array)
		print(np.amax(array), np.amin(array))
		imsave(os.path.join('./', '%s_%s.png' % (save_file_name, param_name)), array)
		if i >= 0:
			break

def quantize_kmeans(model, bit_depth=8, quant_bias=False, verbose=False):
	'''
	Quant model parameters using kmeans.
	:param model: input full-precision model
	:param bit_depth: compress to how many bit depth
	:param quant_bias: whether to quantize bias
	'''
	K = int(2**bit_depth) # cluster number in kmeans
	for param_name, W in model.named_parameters():
		# if W.dim()==4 or (W.dim()==2 and 'linear' in param_name): # Only quant weight in conv, FC layers.
		if param_name.strip().split(".")[-1] in ["weight", "weightA", "weightB", "weightC"] and param_name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
			if verbose:
				print(param_name, W.size(), type(W))

			if W.view(-1).shape[0] <= K:
				continue
			choice_cluster, centers = lloyd(W.view(-1).unsqueeze(-1), K)
			
			if verbose:
				print('choice_cluster:', choice_cluster.size())
				print('centers:', centers.size())

			W.data.copy_(centers[choice_cluster.type(torch.LongTensor)].view(W.size()).data)
			# print("{}: {}".format(param_name, W.data.unique().shape[0]))

def quantize_kmeans_nnz(model, modelp, bit_depth=8, quant_bias=False, verbose=False):
	'''
	Quant model parameters using kmeans.
	:param model: input full-precision model
	:param modelp: indicate nnz position
	:param bit_depth: compress to how many bit depth
	:param quant_bias: whether to quantize bias
	'''
	K = int(2**bit_depth)
	sparse_dict = dict(modelp.named_parameters())
	for param_name, W in model.named_parameters():

		if param_name.strip().split(".")[-1] in ["weight", "weightA", "weightB", "weightC"] and param_name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
			if W.view(-1).shape[0] <= K:
				continue
			Wp = sparse_dict[param_name]
			if len(torch.nonzero(Wp)) <= K-1:
				continue
			choice_cluster, centers = lloyd_nnz(W.view(-1).unsqueeze(-1), Wp.view(-1).unsqueeze(-1), K)

			W.data.copy_(centers[choice_cluster.type(torch.LongTensor)].view(W.shape).data)

def quantize_kmeans_nnz_fixed_0_center(model, modelp, bit_depth=8, quant_bias=False, verbose=False):
	'''
	Quant model parameters using kmeans.
	:param model: input full-precision model
	:param modelp: indicate nnz position
	:param bit_depth: compress to how many bit depth
	:param quant_bias: whether to quantize bias
	'''
	K = int(2**bit_depth)
	sparse_dict = dict(modelp.named_parameters())
	for param_name, W in model.named_parameters():

		if param_name.strip().split(".")[-1] in ["weight", "weightA", "weightB", "weightC"] and param_name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
			if W.view(-1).shape[0] <= K:
				continue
			Wp = sparse_dict[param_name]
			if len(torch.nonzero(Wp)) <= K-1:
				continue
			choice_cluster, centers = lloyd_nnz_fixed_0_center(W.view(-1).unsqueeze(-1), Wp.view(-1).unsqueeze(-1), K)

			W.data.copy_(centers[choice_cluster.type(torch.LongTensor)].view(W.shape).data)

def quantize_kmeans_fixed_nnz(model, modelp, bit_depth=8, quant_bias=False, verbose=False):
	'''
	Quant model parameters using kmeans.
	:param model: input full-precision model
	:param modelp: indicate nnz position
	:param bit_depth: compress to how many bit depth
	:param quant_bias: whether to quantize bias
	'''
	K = int(2**bit_depth)
	sparse_dict = dict(modelp.named_parameters())
	for param_name, W in model.named_parameters():

		if param_name.strip().split(".")[-1] in ["weight", "weightA", "weightB", "weightC"] and param_name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
			if W.view(-1).shape[0] <= K:
				continue
			Wp = sparse_dict[param_name]
			if len(torch.nonzero(Wp)) <= K-1:
				continue
			# print("W and Wp", W.shape, Wp.shape)
			choice_cluster, centers = lloyd_fixed_nnz(W.view(-1).unsqueeze(-1), Wp.view(-1).unsqueeze(-1), K)

			# print(choice_cluster)

			nnz_idx = torch.nonzero(Wp.view(-1))

			# print(nnz_idx)
			# print(W.view(-1)[nnz_idx].shape, W.shape, param_name)

			cpW = W.clone().view(-1)
			oldW = W.clone()
			# print(cpW.shape)
			centers_data = centers[choice_cluster.type(torch.LongTensor)].data
			# print("centers_data", centers_data.shape)
			# cpW[nnz_idx].data.copy_(centers_data)
			# print(cpW.shape)
			# exit(0)

			cpW[nnz_idx] = centers_data
			W.data.copy_(cpW.view(W.shape).data)
			# print(torch.norm(oldW - W))
			# W.view(-1)[nnz_idx].data.copy_(centers[choice_cluster.type(torch.LongTensor)].view(W[nnz_idx].shape).data)