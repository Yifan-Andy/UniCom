import torch

class HeadMLP(torch.nn.Module):
    def __init__(self, dim_dst, dim_src, hidden_dim=128):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_dst, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, dim_src)
        )

    def forward(self, x):
        # x: [N, 7, dim_dst]
        return self.mlp(x)  # output: [N, 7, dim_src]