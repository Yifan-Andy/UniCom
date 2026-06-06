# First Download MAG dataset from NOCD paper

import numpy as np
import torch
import scipy.sparse as sp

def load_and_save(file_name, save_path):
    """Load a graph from a .npz file and save it as a PyTorch .pt file."""
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        
        # --- A ---
        A = sp.csr_matrix((loader['adj_matrix.data'], loader['adj_matrix.indices'],
                           loader['adj_matrix.indptr']), shape=loader['adj_matrix.shape'])
        A = A.tolil()
        A.setdiag(0)
        A = A.tocsr()

        # --- X ---
        if 'attr_matrix.data' in loader.keys():
            X = sp.csr_matrix((loader['attr_matrix.data'], loader['attr_matrix.indices'],
                               loader['attr_matrix.indptr']), shape=loader['attr_matrix.shape'])
            X = torch.tensor(X.toarray(), dtype=torch.float32)
        else:
            X = None

        # --- Z ---
        Z = sp.csr_matrix((loader['labels.data'], loader['labels.indices'],
                           loader['labels.indptr']), shape=loader['labels.shape'])
        y = torch.tensor(Z.toarray(), dtype=torch.float32)  # multi-hot [N, C]

        # --- edge_index_sparse ---
        coo = A.tocoo()
        indices = np.vstack((coo.row, coo.col))
        values = np.ones(indices.shape[1], dtype=np.float32)
        edge_index_sparse = torch.sparse_coo_tensor(
            torch.from_numpy(indices),
            torch.from_numpy(values),
            (A.shape[0], A.shape[1])
        ).coalesce()

        torch.save([edge_index_sparse, X, y], save_path)
        print(f"Saved to {save_path}")

load_and_save('./mag/mag_chem', './processed/mag_chem_pyg.pt')
