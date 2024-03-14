# Step-1: Pretrain GNN Models
CUDA_VISIBLE_DEVICES=0 python pretrain-gnn-arxiv.py --dataset='ogb-arxiv' --lr=0.01 \
--wd=0.0005 --epochs=1000 --model='SAGE' --num_layers=2 --seed=0

# Step-2: Evaluate the pretrained models

CUDA_VISIBLE_DEVICES=1 python evaluator_lebed_arxiv.py  --dataset='ogb-arxiv' \
--model='GIN' --num_layers=2 \
--model_path='../logs-arxiv/Model_pretrain/ogb-arxiv-GIN-0-20230910-090821-747157' \
--lr_frominit=${lr} --wd_frominit=0.0005 --epochs_frominit=${epochs} --atleastepoch=${atleast} \
--seed=0 --lp_margin_rate=${margin}