# Online GNN Evaluation Under Test-Time Graph Distribution Shifts

This is the Pytorch implementation for ICLR-24:"Online GNN Evaluation Under Test-Time Graph Distribution Shifts"
We are trying to solve the online GNN evaluation problem when serving on unseen graphs (without labels and the training
graph) as:

The framework is:

Welcome to kindly cite our work and discuss with xin.zheng@monash.edu:

```
@article{zheng2023gnnevaluator,
  title={Online GNN Evaluation Under Test-Time Graph Distribution Shifts},
  author={Zheng, Xin and Song, Dongjin and Wen, Qingson and Du, Bo and Pan, Shirui},
  journal={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

### Requirements

```
pyg==2.3.0 (py39_torch_1.13.0_cu117 )
pytorch==1.13.1 (py3.9_cuda11.7_cudnn8.5.0_0)
scikit-learn==1.2.2
torch-scatter==2.1.0+pt113cu117
torch-sparse==0.6.16+pt113cu117
```

## Instructions

We run experiments on ACM, DBLP, and Citation, for domain shifts in the folder of 'LEBED-acm-dblp-citation', and Cora
and Amazon for feature shifts in 'LEBED-cora-amz', as well as temporal shifts on OGB-arxiv in 'LEBED-tempo-arxiv'.

In each folder, you could:

### Step1: Obtaining well-trained GNNs

```
CUDA_VISIBLE_DEVICES=0 python pretrain-gnn-arxiv.py --dataset='ogb-arxiv' --lr=0.01 \
--wd=0.0005 --epochs=1000 --model='SAGE' --num_layers=2 --seed=0
```

### Step-2: Evaluating the well-trained online GNNs with our LEBED score

```
CUDA_VISIBLE_DEVICES=1 python evaluator_lebed_arxiv.py  --dataset='ogb-arxiv' \
--model='GIN' --num_layers=2 \
--model_path='../logs-arxiv/Model_pretrain/ogb-arxiv-GIN-0-20230910-090821-747157' \
--lr_frominit=${lr} --wd_frominit=${wd} --epochs_frominit=${epochs} --atleastepoch=${atleast} \
--seed=0 --lp_margin_rate=${margin}
```

You could also directly run Step-2 with our well-trained models in '../logs-arxiv/Model_pretrain/','
../logs-cora/Model_pretrain/', '../logs-amazon/Model_pretrain/', '../logs-domain/Model_pretrain/'; 
And all parameters setting can be found in 'param_space.txt'
