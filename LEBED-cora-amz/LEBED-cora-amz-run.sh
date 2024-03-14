# Step-1: Pretrain GNN Models
python pretrain-lebed.py --dataset='amazon-photo' \
--lr=0.001 --wd=5e-4 --epochs=500 --model='SAGE' --num_layers=2 --gnn_gen='gat'

python pretrain-lebed.py --dataset='cora' --lr=0.0005 \
--wd=0.0005 --epochs=300 --model='GIN' --num_layers=2 --gnn_gen='gat'


# Step-2: Evaluate the pretrained models

CUDA_VISIBLE_DEVICES=0 python evaluator_lebed_cora-amz.py --dataset='cora' --device='cuda:0' \
--model='GIN' --num_layers=2 --lr_frominit=0.0001 --gnn_gen='gat' \
--wd_frominit=0 --epochs_frominit=500 \
--model_path='logs/Model_pretrain/cora-GIN-0-gengat-20230826-114421-756884' \
--seed=0 --lp_margin=0.05

CUDA_VISIBLE_DEVICES=2 python evaluator_lebed_cora-amz.py --dataset='amazon-photo' --device='cuda:0' \
--model='GIN' --num_layers=2 --lr_frominit=1e-4 --gnn_gen='gat' \
--wd_frominit=0 --epochs_frominit=500 \
--model_path='logs/Model_pretrain/amazon-photo-GIN-0-gengat-20230830-114143-613458' \
--seed=0 --lp_margin=0.02 --atleastepoch=20