import torch
import random

seed = 0
# seed everything
random.seed(seed)
torch.manual_seed(seed)

def load_query_train_n_gt(path, dataset, vec_length):
    # load query and ground truth
    query = []
    file_query = open(path + dataset + '/' + dataset + "_train.query", 'r')
    for line in file_query:
        vec = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        for i in line:
            vec[int(i)] = 1
        query.append(vec)

    gt = []
    file_gt = open(path + dataset + '/' + dataset + "_train.gt", 'r')
    for line in file_gt:
        vec = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        for i in line:
            vec[int(i)] = 1
        gt.append(vec)
    
    return torch.Tensor(query), torch.Tensor(gt)

dataset = "cora"
num_nodes = 2708
train_query, train_labels = load_query_train_n_gt("../", dataset, num_nodes)

fixed_selected_indices = []
fixed_selected_labels = []
for i in range(train_query.shape[0]):
    query_mask = train_query[i]
    gt_mask = train_labels[i].float()

    positive_indices = torch.where(gt_mask == 1)[0]
    negative_indices = torch.where(gt_mask == 0)[0]

    num_pos = min(3, positive_indices.shape[0])
    num_neg = min(3, negative_indices.shape[0])

    pos_sampled = positive_indices[torch.randperm(len(positive_indices))[:num_pos]]
    neg_sampled = negative_indices[torch.randperm(len(negative_indices))[:num_neg]]
    selected = torch.cat([pos_sampled, neg_sampled], dim=0)

    fixed_selected_indices.append(selected)
    fixed_selected_labels.append(gt_mask[selected])
torch.save([x.cpu() for x in fixed_selected_indices], f"{dataset}_train_indices.pt")
torch.save([x.cpu() for x in fixed_selected_labels], f"{dataset}_train_labels.pt")
print("Save train indices and labels")
