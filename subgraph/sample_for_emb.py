import torch
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

def get_mean_features(features, labels):
    """
    features: [N, D] - node features
    labels:   [N, C] - one-hot or multi-hot community assignment
    return:   [N, D] - node mean features based on input
    """
    labels = labels.float()
    features = features.float()

    comm_size = labels.sum(dim=0, keepdim=True).T  # [C, 1]
    comm_size[comm_size == 0] = 1  # prevent divide by 0

    mean_feats_per_comm = torch.matmul(labels.T, features) / comm_size  # [C, D]

    norm = labels.sum(dim=1, keepdim=True)
    norm[norm == 0] = 1
    mean_feats_per_node = torch.matmul(labels, mean_feats_per_comm) / norm  # [N, D]

    return mean_feats_per_node

def get_kmeans_communities(features, num_clusters):
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(features)
    cluster_ids = kmeans.labels_  # shape: [N]

    num_nodes = features.shape[0]
    labels = np.zeros((num_nodes, num_clusters), dtype=np.float32)
    for i, cid in enumerate(cluster_ids):
        labels[i, cid] = 1.0

    print(f"[k-means] Total communities: {num_clusters}")

    return torch.tensor(labels, dtype=torch.float32)

def get_modularity_communities(adj, method='louvain'):
    if isinstance(adj, torch.Tensor):
        adj = adj.to_dense().cpu().numpy() if adj.is_sparse else adj.cpu().numpy()
    G = nx.from_numpy_array(adj)

    if method == 'louvain':
        import community as community_louvain  # pip install python-louvain
        partition = community_louvain.best_partition(G, resolution=1.5)
    elif method == 'leiden':
        import igraph as ig
        import leidenalg  # pip install leidenalg

        sources, targets = np.where(np.triu(adj) > 0)
        edges = list(zip(sources.tolist(), targets.tolist()))
        g = ig.Graph(edges=edges, directed=False)

        partition_result = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        membership = partition_result.membership  # [node_id] -> cluster_id
        partition = {i: cid for i, cid in enumerate(membership)}
    else:
        raise ValueError("method must be 'louvain' or 'leiden'")

    node_ids = sorted(partition.keys())
    cluster_ids = np.array([partition[i] for i in node_ids]).reshape(-1, 1)

    enc = OneHotEncoder(sparse=False)
    onehot = enc.fit_transform(cluster_ids)  # [N, C]

    num_clusters = onehot.shape[1]
    print(f"[{method}] Total communities: {num_clusters}")

    return torch.tensor(onehot, dtype=torch.float32)
