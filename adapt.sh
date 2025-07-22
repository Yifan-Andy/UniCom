############################################################ Cora ############################################################
# run cora from DBLP
python3 cs_train.py \
    --dataset cora --batch_size 2708 --dropout 0.5 --hidden_dim 512 --hops 5 \
    --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001 --end_lr 0.0001 --weight_decay 5e-4 --epochs 30 \
    --n_pos_neg_sample 3 --cohesive_subgraphs --alpha 0.5 \
    --pretrain_dataset dblp --pretrain_dim 1642 \
    --embedding_path ./checkpoints/prompt_result/ --device 0 \

# run cora from Computers
python3 cs_train.py \
    --dataset cora --batch_size 2708 --dropout 0.5 --hidden_dim 512 --hops 5 \
    --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001 --end_lr 0.0001 --weight_decay 5e-4 --epochs 30 \
    --n_pos_neg_sample 3 --cohesive_subgraphs --alpha 0.5 \
    --pretrain_dataset computers --pretrain_dim 770 \
    --embedding_path ./checkpoints/prompt_result/ --device 0 \

# run cora from Instagram
python3 cs_train.py \
    --dataset cora --batch_size 2708 --dropout 0.5 --hidden_dim 512 --hops 5 \
    --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001 --end_lr 0.0001 --weight_decay 5e-4 --epochs 30 \
    --n_pos_neg_sample 3 --cohesive_subgraphs --alpha 0.5 \
    --pretrain_dataset instagram --pretrain_dim 503 \
    --embedding_path ./checkpoints/prompt_result/ --device 0 \

# run ensemble
python3 cs_ensemble.py --dataset cora
