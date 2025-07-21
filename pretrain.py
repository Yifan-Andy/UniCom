import time
import utils
import random
import numpy as np
import torch
from early_stop import EarlyStopping, Stop_args
from model import PretrainModel
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data

from data_loader import get_dataset
from utils import parse_args, transform_coo_to_csr, transform_sp_csr_to_coo
from subgraph import get_mean_features, get_kmeans_communities, get_modularity_communities

if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # load dataset
    adj, features = get_dataset(args.dataset, args.pe_dim)
    print(f"get_dataset adj {adj.shape} features {features.shape} no label")

    # process feature
    start_feature_processing = time.time()
    if not os.path.exists(f"checkpoints/processed_features/{args.dataset}.pt"):
        start_feature_processing = time.time()
        processed_features = utils.re_features(adj, features, args.hops)        # [N, hops+1, d]
        if processed_features.shape[0] < 10000:
            indicator = utils.conductance_hop(adj, args.hops)                   # [N, hops+1]
            indicator = indicator.unsqueeze(2).repeat(1, 1, features.shape[1])
            processed_features = processed_features * indicator
        torch.save(processed_features, f"checkpoints/processed_features/{args.dataset}.pt")
    else:
        processed_features = torch.load(f"checkpoints/processed_features/{args.dataset}.pt")
    t_feature_precessing = time.time() - start_feature_processing
    print("feature process time: {:.4f}s".format(t_feature_precessing))
    
    if args.cohesive_subgraphs:
        # ============================== Use K-means to generate community embedding for node ============================
        kmeans_communities = get_kmeans_communities(features, 10)
        kmeans_features = get_mean_features(features, kmeans_communities).unsqueeze(1)
        processed_features = torch.cat([processed_features, kmeans_features], dim=1)
        args.hops += 1
        # ============================== Use Modularity to generate community embedding for node ============================
        modularity_communities = get_modularity_communities(adj)
        modularity_features = get_mean_features(features, modularity_communities).unsqueeze(1)
        processed_features = torch.cat([processed_features, modularity_features], dim=1)
        args.hops += 1
        # ===================================================================================================================
        print(f"Final processed feature shape {processed_features.shape}")

    # generate data mini batch
    start = time.time()
    print("starting transformer to coo")
    # transform to csr to support slicing operation
    adj = transform_coo_to_csr(adj)
    print("start mini batch processing")
    # transform to coo to support tensor operation
    adj_batch, minus_adj_batch = transform_sp_csr_to_coo(adj, args.batch_size, features.shape[0])
    print(len(adj_batch[0]), len(minus_adj_batch[0]))
    print("adj process time: {:.4f}s".format(time.time() - start))
    
    data_loader = Data.DataLoader(processed_features, batch_size=args.batch_size, shuffle=False)

    # model configuration
    model = PretrainModel(input_dim=processed_features.shape[2], config=args).to(args.device)

    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0
    )
    
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)

    print("starting training...")
    # model train
    model.train()

    t_start = time.time()

    loss_train_b = []
    for epoch in range(args.epochs):
        for index, item in enumerate(data_loader):
            start_index = index * args.batch_size
            nodes_features = item.to(args.device)
            adj_ = adj_batch[index].to(args.device)
            minus_adj = minus_adj_batch[index].to(args.device)
            # print(nodes_features.shape)
            optimizer.zero_grad()
            node_tensor, neighbor_tensor = model(nodes_features)

            # print(node_tensor.shape, neighbor_tensor.shape, adj_.shape, minus_adj.shape)
            loss_train = model.contrastive_link_loss(node_tensor, neighbor_tensor, adj_, minus_adj)
            loss_train.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_train_b.append(loss_train.item())
            # break

        if early_stopping.simple_check(loss_train_b):
            break

        print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()))
        # 'loss_train: {:.4f}'.format(np.mean(np.array(loss_train_b)))
    
    print("Optimization Finished!")
    print("Train time: {:.4f}s".format(time.time() - t_start + t_feature_precessing))

    # model save
    print(f"Save Model to {args.save_path + args.dataset + '.pth'}")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)

    torch.save(model.state_dict(), args.save_path + args.dataset + '.pth')
