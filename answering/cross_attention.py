import torch

class AnsweringCrossAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.out_proj = torch.nn.Linear(input_dim, 1)

    def forward(self, node_embeddings, query_vec):
        # node_embeddings: [N, d], query_vec: [d]
        _, d = node_embeddings.shape

        # query_vec and node similarity
        scores = torch.matmul(node_embeddings, query_vec) / (d ** 0.5)  # [N]
        attn_weights = torch.softmax(scores, dim=0).unsqueeze(-1)       # [N, 1]

        # calculate context from weights and embeddings
        context = torch.sum(attn_weights * node_embeddings, dim=0)      # [d]

        # from context and each node
        combined = context * node_embeddings                            # [N, d]
        logits = self.out_proj(combined).squeeze(-1)                    # [N]

        return logits
