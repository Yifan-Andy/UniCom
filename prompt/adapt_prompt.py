import torch
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F

class AdaptPrompt(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(AdaptPrompt, self).__init__()
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        # x: [N, H, D]
        x = x.clone()
        N, H, D = x.shape

        x_flat = x.view(N * H, D)        # [N*H, D]
        score = self.a(x_flat)           # [N*H, p_num]
        weight = F.softmax(score, dim=1) # [N*H, p_num]
        p = weight @ self.p_list         # [N*H, D]
        p = p.view(N, H, D)              # [N, H, D]

        return x + p  # [N, H, D]
