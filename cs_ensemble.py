import os
import torch
import random
import numpy as np
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, jaccard_score

from data_loader import get_dataset
from utils import parse_args, load_query_test_n_gt

if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # data load
    adj, features = get_dataset(args.dataset, args.pe_dim)
    num_nodes = features.shape[0]
    all_test_labels = []
    test_query, test_labels = load_query_test_n_gt("./dataset/", args.dataset, num_nodes)
    for test_label in test_labels:
        all_test_labels.append(test_label)
    all_test_labels = torch.cat(all_test_labels).numpy()

    # probs load
    test_probs1 = torch.load(f"./checkpoints/ensemble/dblp_to_{args.dataset}.pt")
    test_probs2 = torch.load(f"./checkpoints/ensemble/computers_to_{args.dataset}.pt")
    test_probs3 = torch.load(f"./checkpoints/ensemble/instagram_to_{args.dataset}.pt")

    # thresh set
    if args.dataset == "cora":
        thresh = 0.75
    else:
        raise Exception(f"Define threshold for dataset {args.dataset}")

    file_path = f"logs/{args.dataset}_CS_Ensemble.log"
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            print("Create Logs")

    with open(file_path, 'a') as file:

        # single transfer
        for probs, name in zip(
            [test_probs1, test_probs2, test_probs3],
            ['dblp', 'computers', 'instagram']):

            label_prefix = f"[Source: {name} → {args.dataset}]"
            print(f"\n===== {label_prefix} =====")
            file.write(f"===== {label_prefix} =====\n")

            probs_tensor = torch.tensor(probs)
            preds = (torch.sigmoid(probs_tensor) > thresh).float()

            # eval
            f1 = f1_score(all_test_labels, preds)
            nmi = normalized_mutual_info_score(all_test_labels, preds)
            ari = adjusted_rand_score(all_test_labels, preds)
            jac = jaccard_score(all_test_labels, preds)
            print(f"{label_prefix} [All] F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, JAC: {jac:.4f}")
            file.write(f"{label_prefix} [All] F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, JAC: {jac:.4f}\n")

            # top-k
            num_queries = len(test_labels)
            if args.dataset in ["mag-cs", "mag-eng", "reddit"]:
                top_k = 1000
            elif args.dataset[:2] == "fb":
                top_k = 30
            else:
                top_k = 150

            probs_for_topk = probs_tensor.view(num_queries, num_nodes)
            all_test_labels_tensor = torch.tensor(all_test_labels).view(num_queries, num_nodes).float()
            preds_topk = torch.zeros((num_queries, num_nodes))
            labels_topk = torch.zeros((num_queries, num_nodes))

            for i in range(num_queries):
                topk_idx = torch.topk(probs_for_topk[i], k=top_k).indices
                preds_topk[i, topk_idx] = 1.0
                labels_topk[i, topk_idx] = all_test_labels_tensor[i, topk_idx]

            flat_preds = preds_topk.view(-1).numpy()
            flat_labels = labels_topk.view(-1).numpy()

            f1 = f1_score(flat_labels, flat_preds)
            nmi = normalized_mutual_info_score(flat_labels, flat_preds)
            ari = adjusted_rand_score(flat_labels, flat_preds)
            jac = jaccard_score(flat_labels, flat_preds)
            print(f"{label_prefix} [Top-{top_k}] F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, JAC: {jac:.4f}")
            file.write(f"{label_prefix} [Top-{top_k}] F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, JAC: {jac:.4f}\n")

        # ensemble eval
        test_probs = torch.tensor((test_probs1 + test_probs2 + test_probs3) / 3)
        all_test_preds = (torch.sigmoid(test_probs) > thresh).float()
        f1 = f1_score(all_test_labels, all_test_preds)
        nmi = normalized_mutual_info_score(all_test_labels, all_test_preds)
        ari = adjusted_rand_score(all_test_labels, all_test_preds)
        jac = jaccard_score(all_test_labels, all_test_preds)

        ensemble_label = f"[Ensemble: dblp + computers + instagram → {args.dataset}]"
        print(f"\n{ensemble_label} [Top-CSD] F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, JAC: {jac:.4f}")
        file.write(f"\n{ensemble_label} [Top-CSD] F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, JAC: {jac:.4f}\n")

        # ensemble top-k
        test_probs = test_probs.view(num_queries, num_nodes)
        all_test_labels_tensor = torch.tensor(all_test_labels).view(num_queries, num_nodes).float()
        preds_topk = torch.zeros((num_queries, num_nodes))
        labels_topk = torch.zeros((num_queries, num_nodes))

        for i in range(num_queries):
            topk_idx = torch.topk(test_probs[i], k=top_k).indices
            preds_topk[i, topk_idx] = 1.0
            labels_topk[i, topk_idx] = all_test_labels_tensor[i, topk_idx]

        flat_preds = preds_topk.view(-1).numpy()
        flat_labels = labels_topk.view(-1).numpy()

        f1 = f1_score(flat_labels, flat_preds)
        nmi = normalized_mutual_info_score(flat_labels, flat_preds)
        ari = adjusted_rand_score(flat_labels, flat_preds)
        jac = jaccard_score(flat_labels, flat_preds)

        print(f"{ensemble_label} [Top-{top_k}] F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, JAC: {jac:.4f}")
        file.write(f"{ensemble_label} [Top-{top_k}] F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, JAC: {jac:.4f}\n")
        file.write('==============================================\n\n')

    print("Logs Write Successful")
