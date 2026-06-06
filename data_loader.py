import utils
import dgl
import torch

from dgl.data import AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from sklearn.decomposition import TruncatedSVD

def get_dataset(dataset, pe_dim):
    if dataset in {
        "texas", "cornell", "wisconsin",
        "cora", "citeseer", "pubmed",
        "dblp", "reddit", "products", "cocs",
        "computers", "instagram", "photo", "wikics",
        "fb0", "fb107", "fb348", "fb1912", "fb1684",
        "mag-med", "mag-eng", "mag-cs", "mag-chem"
    }:
        # Load the dataset file
        file_path = "dataset/" + dataset + "_pyg.pt"
        data_list = torch.load(file_path)

        # Extract adjacency matrix and node features
        adj = data_list[0]
        features = data_list[1]

        # Convert PyTorch sparse adjacency to SciPy sparse matrix
        adj_scipy = utils.torch_adj_to_scipy(adj)

        # Construct a DGL graph from the SciPy adjacency matrix
        graph = dgl.from_scipy(adj_scipy)

        # Compute Laplacian positional encoding (LPE)
        lpe = utils.laplacian_positional_encoding(graph, pe_dim)  # shape: [num_nodes, pe_dim]

        # Concatenate LPE and node feature
        features = torch.cat((features, lpe), dim=1)

    print(type(adj), type(features))
    print(f"Labels {data_list[2].shape}")

    return adj.cpu().type(torch.LongTensor), features.long()

def get_dataset_svd(dataset, pe_dim):
    if dataset in {
        "texas", "cornell", "wisconsin",
        "cora", "citeseer", "pubmed",
        "dblp", "reddit", "products", "cocs",
        "computers", "instagram", "photo", "wikics",
        "fb0", "fb107", "fb348", "fb1912", "fb1684"
    }:
        # Load the dataset file
        file_path = "dataset/" + dataset + "_pyg.pt"
        data_list = torch.load(file_path)

        # Extract adjacency matrix and node features
        adj = data_list[0]
        features = data_list[1]
        svd = TruncatedSVD(n_components=2000)
        features = torch.tensor(svd.fit_transform(features))

        # Convert PyTorch sparse adjacency to SciPy sparse matrix
        adj_scipy = utils.torch_adj_to_scipy(adj)

        # Construct a DGL graph from the SciPy adjacency matrix
        graph = dgl.from_scipy(adj_scipy)

        # Compute Laplacian positional encoding (LPE)
        lpe = utils.laplacian_positional_encoding(graph, pe_dim)  # shape: [num_nodes, pe_dim]

        # Concatenate LPE and node feature
        features = torch.cat((features, lpe), dim=1)
    
    return adj.cpu().type(torch.LongTensor), features.long()

class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, train_query, fixed_selected_indices, fixed_selected_labels):
        self.train_query = train_query
        self.fixed_selected_indices = fixed_selected_indices
        self.fixed_selected_labels = fixed_selected_labels

    def __len__(self):
        return self.train_query.shape[0]

    def __getitem__(self, i):
        selected_indices = self.fixed_selected_indices[i]                                       # [N]
        selected_labels = self.fixed_selected_labels[i]                                         # [N]
        query_indices = torch.where(self.train_query[i] == 1)[0]                                # [Q]

        return {
            "sample_indices": selected_indices,
            "sample_labels": selected_labels,
            "query_indices": query_indices
        }
