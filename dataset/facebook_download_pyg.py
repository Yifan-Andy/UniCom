# First Download Facebook dataset from NOCD paper

import numpy as np
import torch

def edge_index_to_sparse_coo(edge_index):
    row = edge_index[0].long()
    col = edge_index[1].long()
    num_nodes = torch.max(edge_index) + 1
    size = (num_nodes.item(), num_nodes.item())
    values = torch.ones_like(row)
    edge_index_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, size)
    return edge_index_sparse

def load_fb_ego_data(data_dir, ego_id):
    ego_path = f"{data_dir}/{ego_id}"

    ego_feat = np.genfromtxt(f"{ego_path}.egofeat", dtype=np.int32)
    feat = np.genfromtxt(f"{ego_path}.feat", dtype=np.int32)
    features = np.vstack((ego_feat, feat[:, 1:]))

    node_map = {int(ego_id): 0}
    for i, row in enumerate(feat):
        node_map[int(row[0])] = i + 1

    edges_raw = np.genfromtxt(f"{ego_path}.edges", dtype=np.int32)
    edge_list = []
    for e in edges_raw:
        if e[0] in node_map and e[1] in node_map:
            edge_list.append((node_map[e[0]], node_map[e[1]]))
    for i in range(1, features.shape[0]):
        edge_list.append((0, i))

    edge_index_np = np.array(edge_list).T
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    with open(f"{ego_path}.circles", 'r') as f:
        lines = f.readlines()
    num_nodes = features.shape[0]
    num_comms = len(lines)
    labels = np.zeros((num_nodes, num_comms), dtype=np.int32)
    for k, line in enumerate(lines):
        ids = line.strip().split()[1:]
        for nid in ids:
            if int(nid) in node_map:
                idx = node_map[int(nid)]
                labels[idx, k] = 1

    x = torch.tensor(features, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    edge_index_sparse = edge_index_to_sparse_coo(edge_index).to(torch.long)

    return edge_index_sparse, x, y

ego_id = "1684"
edge_index_sparse, x, y = load_fb_ego_data("./facebook", ego_id)
torch.save([edge_index_sparse, x, y], f"./processed/fb{ego_id}_pyg.pt")

print(f"Saved: ./processed/fb{ego_id}_pyg.pt")
print("edge_index_sparse shape:", edge_index_sparse.shape)
