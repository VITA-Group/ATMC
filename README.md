## Model Compression with Adversarial Robustness: A Unified Optimization Framework (ATMC)
### Authors:
* [Shupeng Gui](https://sites.google.com/view/shupeng-gui/home)*
* [Haotao Wang](http://people.tamu.edu/~htwang/)*
* [Haichuan Yang](https://www.cs.rochester.edu/u/hyang36/)
* [Chen Yu](https://www.linkedin.com/in/lukecyu/en)
* [Zhangyang Wang](https://www.atlaswang.com/)
* [Ji Liu](https://scholar.google.com/citations?user=RRzVwKkAAAAJ&hl=en)

*: Equal Contribution

### Overview
In this repo, we present one example implementation of ATMC robust learning framework from NeurIPS 2019 paper [Model Compression with Adversarial Robustness: A Unified Optimization Framework](https://arxiv.org/abs/1902.03538).

We propose a noval *Adversarially Trained Model Compression* (ATMC) framework, which conducts a unified constrained optimization formulation, where existing compression means (pruning, factorization, quantization) are all integrated into the constraints. An extensive group of experiments are presented, demonstrating that ATMC obtains remarkably more favorable trade-off among model size, accuracy and robustness, over currently available alternatives in various settings.
![ATMC Experiments Results on Various Compression Ratio](ATMC_exps.png)

### Requirements
All experiments were executed on a Linux machine with Intel I7-6700k, 64 GB memory and two GTX1080 Graphics Card. To reproduce the experiment results in the paper, some experiment parameter settings could be tuned for the user case (such as batch size).

The software environment bases on Pytorch (>=1.0.0).

### Experiment Example
One example shows how to setup an experiment on CIFAR-10 dataset.

First of all, we need to obtain a dense model for successors.
```
python cifar/train_proj_admm_quant.py \
--raw_train \
--epochs 200 \
--lr 0.05 \
--decreasing_lr 80,120,150 \
--gpu 0 \
--savedir log/resnet/pretrain \          
--data_root [cifar/data/dir] \            
--attack_algo pgd \
--attack_eps 4 \
--defend_algo pgd \
--defend_eps 4 \
--defend_iter 7 \
--save_model_name cifar10_res_pgd_raw.pth \
--quantize_bits 32
```
The dense model will be stored in 'log/resnet/pretrain'. Then, the second round execution of the python script will be operated with
```
python cifar/train_proj_admm_quant.py \
--epochs 200 \
--lr 0.01 \
--decreasing_lr 30,60,90,120 \
--gpu 0 \
--savedir log/resnet/l0 \
--loaddir log/resnet/pretrain \
--model_name cifar10_res_pgd_raw.pth \  
--data_root [cifar/data/dir] \       
--quantize_bits 32 \
--attack_algo pgd \
--attack_eps 4 \
--defend_algo pgd \                                             
--defend_eps 4 \
--defend_iter 7 \
--save_model_name cifar10_resnet_pgd_4_l0proj_0.005.pth \
--quantize_bits 32 \ 
--prune_algo l0proj \
--prune_ratio 0.005 \
```
After the process above, we get an sparse model with 0.005 compression ratio. Then we operate the ATMC process based on this pre-trained model.
```
python cifar/train_proj_admm_quant.py \
--epochs 200 \                     
--lr 0.005 \     
--decreasing_lr 60,80,120 \   
--gpu 0 \                          
--savedir log/resnet/atmc \                                             
--loaddir log/resnet/l0 \                                               
--save_model_name cifar10_resnet_pgd_4_atmc_0.005_32bit.pth \ 
--data_root [cifar/data/dir] \     
--attack_algo pgd \
--attack_eps 4 \
--defend_algo pgd \                                             
--defend_eps 4 \    
--defend_iter 7 \                                              
--prune_algo l0proj \
--abc_special \        
--prune_ratio 0.005 \ 
--quantize_bits 32 \
--defend_iter 7 \
--model_name sparse_cifar10_resnet_pgd_4_l0proj_0.005.pth
```
If we want to apply the unified pruning and quantization strategy, we are going to run
```
python cifar100/train_proj_admm_quant.py \
--epochs 150 \
--lr 0.005 \
--decreasing_lr 60,80,120 \
--gpu 0 \
--savedir log/resnet/atmc8bit \
--loaddir log/resnet/atmc \
--save_model_name cifar10_pgd_4_atmc_0.005_8bit.pth \
--data_root [cifar/data/dir] \
--attack_algo pgd \
--attack_eps 4 \
--abc_special \
--abc_initialize \
--defend_algo pgd \
--defend_eps 4 \
--defend_iter 7 \
--quantize_algo kmeans_nnz_fixed_0_center \
--quant_interval 10 \
--prune_algo l0proj \
--prune_ratio 0.005 \
--quantize_bits 8 \
--model_name sparse_cifar10_pgd_4_atmc_0.005_32bit.pth
```

If you find this repo useful, please cite:
```
@InProceedings{gui2019ATMC,
  title = 	 {Model Compression with Adversarial Robustness: A Unified Optimization Framework},
  author = 	 {Gui, Shupeng and Wang, Haotao and Yang, Haichuan and Yu, Chen and Wang, Zhangyang and Liu, Ji},
  booktitle = 	 {Proceedings of the 33rd Conference on Neural Information Processing Systems},
  year = 	 {2019},
}
```

### Reference Implementation
Thanks to the reference repo [pytorch-playground](https://github.com/aaron-xichen/pytorch-playground).

### Dependencies
- pytorch (>=1.0.0)
