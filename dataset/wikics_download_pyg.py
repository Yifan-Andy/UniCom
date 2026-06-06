import torch

def edge_index_to_sparse_coo(edge_index):
    row = edge_index[0].long()
    col = edge_index[1].long()

    num_nodes = torch.max(edge_index) + 1
    size = (num_nodes.item(), num_nodes.item())

    values = torch.ones_like(row)
    edge_index_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, size)

    return edge_index_sparse

dataset_str = 'wikics'

from torch_geometric.datasets import WikiCS
dataset = WikiCS(root='./data/wikics')
graph = dataset[0]

torch.save([edge_index_to_sparse_coo(graph.edge_index).type(torch.LongTensor), graph.x.type(torch.LongTensor), graph.y.type(torch.LongTensor)], "./dataset/"+dataset_str+"_pyg.pt")

