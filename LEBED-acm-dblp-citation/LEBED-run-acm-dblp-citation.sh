# Step-1: Pretrain GNN Models
CUDA_VISIBLE_DEVICES=0 python pretrain_gnn_domain.py --source='acm' --target1='dblp' --target2='network' \
--hid_dim=256 --encoder_dim=32 \
--lr=5.00E-04 --wd=5.00E-05 --epochs=200 --model='GIN' --num_layers=2

CUDA_VISIBLE_DEVICES=0 python pretrain_gnn_domain.py --source='dblp' --target1='acm' --target2='network' \
--hid_dim=256 --encoder_dim=32 \
--lr=1.00E-03 --wd=1.00E-04 --epochs=200 --model='GCN' --num_layers=2

CUDA_VISIBLE_DEVICES=0 python pretrain_gnn_domain.py --source='network' --target1='acm' --target2='dblp' \
--hid_dim=256 --encoder_dim=32 \
--lr=5.00E-04 --wd=1.00E-04 --epochs=200 --model='GAT' --num_layers=2

# Step-2: Evaluate the pretrained models
python evaluator_lebed_domain.py  --source='acm' --target1='dblp' --target2='network' \
--hid_dim=256 --encoder_dim=32 --model='GCN' --num_layers=2 \
--model_path='logs/Models_tra/acm-to-dblp-network-GCN-0-20230906-092805-309675' \
--lr_frominit=${lr} --wd_frominit=1e-5 --epochs_frominit=${epochs} --atleastepoch=${atleast} \
--seed=0 --lp_margin_rate=${margin} --domain_ind='both' --mode '1' '2' '3' '4'

python evaluator_lebed_domain.py  --source='dblp' --target1='acm' --target2='network' \
--hid_dim=256 --encoder_dim=32 --model='MLP' --num_layers=2 \
--model_path='logs/Models_tra/dblp-to-acm-network-MLP-0-20230906-094147-042228' \
--lr_frominit=${lr} --wd_frominit=${wd} --epochs_frominit=${epochs} --atleastepoch=${atleast} \
--seed=0 --lp_margin_rate=${margin} --domain_ind='both' --mode '1' '2' '3' '4'

python evaluator_lebed_domain.py  --source='network' --target1='acm' --target2='dblp' \
--hid_dim=256 --encoder_dim=32 --model='GCN' --num_layers=2 \
--model_path='logs/Models_tra/network-to-acm-dblp-GCN-0-20230906-094311-188577' \
--lr_frominit=${lr} --wd_frominit=1.00E-04 --epochs_frominit=${epochs} --atleastepoch=${atleast} \
--seed=0 --lp_margin_rate=${margin} --domain_ind='both' --mode '1' '2' '3' '4'