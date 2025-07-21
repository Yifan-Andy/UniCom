# Pretrain

python3 pretrain.py --cohesive_subgraphs --dataset dblp --batch_size 17716 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001 --weight_decay 1e-05 --device 1

python3 pretrain.py --cohesive_subgraphs --dataset computers --batch_size 13752 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001 --weight_decay 1e-05 --device 1

python3 pretrain.py --cohesive_subgraphs --dataset instagram --batch_size 11339 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001 --weight_decay 1e-05 --device 1
