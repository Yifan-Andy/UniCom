import os
import time
import utils
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, jaccard_score, accuracy_score

from data_loader import get_dataset
from early_stop import EarlyStopping, Stop_args
from model import PretrainModel
from lr import PolynomialDecayLR
from utils import parse_args, load_query_train_n_gt, load_query_test_n_gt
from prompt import AdaptPrompt
from subgraph import get_mean_features, get_kmeans_communities, get_modularity_communities
from answering import AnsweringCrossAttention
from head import HeadMLP
from loss import CMMDLoss

def extract_representative_nodes(features: torch.Tensor, num_representatives: int = 10):
    # change numpy
    features_np = features.detach().cpu().numpy()

    # KMeans clustering
    kmeans = KMeans(n_clusters=num_representatives, random_state=42, n_init='auto').fit(features_np)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # center nodes
    representative_indices = []
    for i in range(num_representatives):
        cluster_indices = np.where(labels == i)[0]
        cluster_points = features_np[cluster_indices]
        distances = np.linalg.norm(cluster_points - centers[i], axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        representative_indices.append(int(closest_index))

    return representative_indices

if __name__ == "__main__":
    args = parse_args()
    print(args)

    # seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # seed cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # load dataset
    adj, features = get_dataset(args.dataset, args.pe_dim)
    print(f"get_dataset adj {adj.shape} features {features.shape} pe {args.pe_dim}")

    # process feature
    start_feature_processing = time.time()
    if not os.path.exists(f"checkpoints/processed_features/{args.dataset}.pt"):
        processed_features = utils.re_features(adj, features, args.hops)  # return (N, hops+1, d)
        if processed_features.shape[0] < 10000:
            indicator = utils.conductance_hop(adj, args.hops) # return (N, hops+1)
            indicator = indicator.unsqueeze(2).repeat(1, 1, features.shape[1])
            processed_features = processed_features * indicator
        torch.save(processed_features, f"checkpoints/processed_features/{args.dataset}.pt")
        print("feature saved")
    else:
        processed_features = torch.load(f"checkpoints/processed_features/{args.dataset}.pt")
        print("feature process from saved checkpoint")

    t_feature_precessing = time.time() - start_feature_processing
    print("feature process time: {:.4f}s".format(t_feature_precessing))
    
    num_nodes = features.shape[0]
    if args.cohesive_subgraphs:
        # ============================== Use K-means to generate cohesive prompt ============================
        kmeans_communities = get_kmeans_communities(features, 10)
        kmeans_features = get_mean_features(features, kmeans_communities).unsqueeze(1)
        processed_features = torch.cat([processed_features, kmeans_features], dim=1)
        args.hops += 1
        # ============================== Use Modularity to generate cohesive prompt ============================
        if num_nodes < 50000:
            modularity_communities = get_modularity_communities(adj)
            modularity_features = get_mean_features(features, modularity_communities).unsqueeze(1)
            processed_features = torch.cat([processed_features, modularity_features], dim=1)
            args.hops += 1
        # ===================================================================================================================
        print(f"Final processed feature shape {processed_features.shape}")

    # batch
    data_loader = Data.DataLoader(processed_features, batch_size=args.batch_size, shuffle=False)

    # model configuration
    model = PretrainModel(input_dim=args.pretrain_dim, config=args).to(args.device)
    # model.load_state_dict(torch.load(args.save_path + args.dataset + '.pth', map_location=f"cuda:{args.device}"))
    model.load_state_dict(torch.load(args.save_path + args.pretrain_dataset + '.pth', map_location=f"cuda:{args.device}"))
    print(f"Use pretrain model trained on {args.pretrain_dataset} with dim {args.pretrain_dim}")
    
    # head
    head = HeadMLP(processed_features.shape[2], args.pretrain_dim).to(args.device)
    
    # prompt
    prompt = AdaptPrompt(processed_features.shape[2], args.n_prompt).to(args.device)
    
    # answering
    answering = AnsweringCrossAttention(args.hidden_dim * 2).to(args.device)
    
    # optimizer
    model_param_group = [
        {"params": prompt.parameters()},
        {"params": answering.parameters()},
        {"params": head.parameters()}
    ]
    optimizer = torch.optim.AdamW(model_param_group, lr=args.peak_lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'] if p.requires_grad)
    print(f"Total trainable parameters in optimizer: {total_params}")

    # lr model
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0
    )

    # early stop
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)

    # data load
    train_query, _ = load_query_train_n_gt("./dataset/", args.dataset, processed_features.shape[0])
    train_query = train_query.to(args.device)
    test_query, test_labels = load_query_test_n_gt("./dataset/", args.dataset, processed_features.shape[0])
    test_query, test_labels = test_query.to(args.device), test_labels.to(args.device)

    # train set
    fixed_selected_indices = torch.load(f"dataset/{args.dataset}/{args.dataset}_train_indices.pt")
    fixed_selected_indices = [x.to(args.device) for x in fixed_selected_indices]
    fixed_selected_labels = torch.load(f"dataset/{args.dataset}/{args.dataset}_train_labels.pt")
    fixed_selected_labels = [x.to(args.device) for x in fixed_selected_labels]
    print("Load train indices and labels from checkpoints")

    print(f"For each query sample {fixed_selected_labels[0].shape[0]} for train e.g. {fixed_selected_indices[0]} with labels {fixed_selected_labels[0]}")

    # cmmd loss initialize
    cmmd_loss_fn = CMMDLoss()
    
    # load pretrain features
    pretrain_adj, pretrain_features_raw = get_dataset(args.pretrain_dataset, 3)
    pretrain_features = torch.load(f"checkpoints/processed_features/{args.pretrain_dataset}.pt")
    if args.cohesive_subgraphs:
        # ============================== Use K-means to generate community embedding for node ============================
        kmeans_communities = get_kmeans_communities(pretrain_features_raw, 10)
        kmeans_features = get_mean_features(pretrain_features_raw, kmeans_communities).unsqueeze(1)
        pretrain_features = torch.cat([pretrain_features, kmeans_features], dim=1)
        # ============================== Use Modularity to generate community embedding for node ============================
        modularity_communities = get_modularity_communities(pretrain_adj)
        modularity_features = get_mean_features(pretrain_features_raw, modularity_communities).unsqueeze(1)
        pretrain_features = torch.cat([pretrain_features, modularity_features], dim=1)
        # ===================================================================================================================
        print(f"Final pretrain feature shape {pretrain_features.shape}")
    print(f"Target: {processed_features.shape} and Pretrain {pretrain_features.shape}")

    pretrain_features_mean = pretrain_features.mean(dim=1).to(args.device)
    indices = extract_representative_nodes(pretrain_features_mean, num_representatives=10)
    pretrain_features_mean = pretrain_features_mean[indices]
    print(f"Selected important nodes from pretrain dataset {indices}")

    # running
    print("starting prompting...")
    t_start = time.time()

    print(f"queries number is {train_query.shape[0]}")
    # ============================ Train ============================
    t_start = time.time()
    for epoch in range(args.epochs):
        total_loss = 0
        total_bce_loss = 0
        total_cmmd_loss = 0

        for i in range(train_query.shape[0]):
            head.train()
            prompt.train()
            answering.train()
            model.eval()
            optimizer.zero_grad()

            # get (pos and neg) sample and query node index
            selected_indices = fixed_selected_indices[i]                        # [N]
            selected_gt = fixed_selected_labels[i]                              # [N]
            query_indices = torch.where(train_query[i] == 1)[0]                 # [Q]

            # concat query and (pos and neg) samples
            all_indices = torch.cat([selected_indices, query_indices], dim=0)   # [N+Q]
            all_indices = all_indices.to(processed_features.device)             # set features and mask on same device
            
            # generate training features
            all_features = processed_features[all_indices]                      # [N+Q, hop, dim]
            all_features = all_features.to(args.device)
            
            # add prompt
            prompted_features = prompt.add(all_features)                        # [N+Q, hop, dim]
            
            # mlp to map from current dataset dim to pretrain dim
            prompted_features = head(prompted_features)
            
            # cmmd loss compute
            prompted_features_mean = prompted_features.mean(dim=1)
            cmmd_loss = cmmd_loss_fn(pretrain_features_mean, prompted_features_mean)

            # forward pass
            node_tensor, neighbor_tensor = model(prompted_features)             # both [N+Q, d]
            node_embeddings = torch.cat((node_tensor, neighbor_tensor), dim=1)  # [N+Q, 2d]

            # seperate query, (pos and neg) emb
            num_sample = selected_indices.shape[0]
            sample_embeddings = node_embeddings[:num_sample]                    # pos samples
            query_embeddings = node_embeddings[num_sample:]                     # query nodes

            # get query emb with mean pool
            query_vec = query_embeddings.mean(dim=0)                            # [2d]

            # get pred with answering function
            logits = answering(sample_embeddings, query_vec)                    # [N]
            pred = torch.sigmoid(logits)
            bce_loss = F.binary_cross_entropy(pred, selected_gt)
            loss = bce_loss + args.alpha * cmmd_loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            total_bce_loss += bce_loss.item()
            total_cmmd_loss += cmmd_loss.item()
            
        print(f"Total Loss: {total_loss}")
        t_train = time.time() - t_start

        # ============================ Evaluation ============================
        head.eval()
        prompt.eval()
        answering.eval()

        all_test_probs = []

        if (epoch + 1) == args.epochs:
            t_eval = time.time()
            with torch.no_grad():
                node_embeddings = []
                for _, item in enumerate(data_loader):
                    nodes_features = item.to(args.device)
                    prompted_features = prompt.add(nodes_features)
                    prompted_features = head(prompted_features)
                    node_tensor, neighbor_tensor = model(prompted_features)
                    embeddings = torch.cat((node_tensor, neighbor_tensor), dim=1)   # [batch_size, 2d]
                    node_embeddings.append(embeddings)
                node_embeddings = torch.cat(node_embeddings, dim=0)                 # [total_nodes, 2d]

                for i in range(test_query.shape[0]):
                    query_mask = test_query[i]
                    gt_mask = test_labels[i].float()

                    query_indices = torch.where(query_mask == 1)[0]
                    if query_indices.numel() == 0:
                        continue

                    query_vec = node_embeddings[query_indices].mean(dim=0)          # [2d]
                    logits = answering(node_embeddings, query_vec)                  # [N]
                    all_test_probs.append(logits.cpu())
            t_eval = time.time() - t_eval

            # Concat all
            all_test_probs = torch.cat(all_test_probs).numpy()

    # save probs for ensemble
    print(f"Save probs results to ./checkpoints/ensemble/{args.pretrain_dataset}_to_{args.dataset}.pt")
    torch.save(all_test_probs, f"./checkpoints/ensemble/{args.pretrain_dataset}_to_{args.dataset}.pt")
    print(f"Ensemble size {all_test_probs.shape}")
    print(f"Optimization Finished for {args.dataset}!")
