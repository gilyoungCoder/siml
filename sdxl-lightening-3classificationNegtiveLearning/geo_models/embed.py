import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitEmbedding(nn.Module):
    def __init__(self, embed, num_fixed):
        """
        Args:
            embed (nn.Embedding)
            num_fixed: default to be the first num_fixed tokens
        """
        super(SplitEmbedding, self).__init__()
        self.embed_origin = embed
        self.padding_idx = embed.padding_idx
        self.max_norm = embed.max_norm
        self.norm_type = embed.norm_type
        self.scale_grad_by_freq = embed.scale_grad_by_freq
        self.sparse = embed.sparse
        
        self.num_embeddings, self.embedding_dim = embed.weight.shape
        self.num_fixed = num_fixed
        self.num_tuned = self.num_embeddings - num_fixed
        print("Spliting original embedding with shape ({}, {}) to ({}, {}) and ({}, {})".format(self.num_embeddings, self.embedding_dim, self.num_fixed, self.embedding_dim, self.num_tuned, self.embedding_dim))
    
        self.fixed_tokens = nn.Parameter(torch.zeros(self.num_fixed, self.embedding_dim), requires_grad=False)
        self.tuned_tokens = nn.Parameter(torch.zeros(self.num_tuned, self.embedding_dim))
        self.fixed_tokens.data.copy_(embed.weight.data[:num_fixed])
        self.tuned_tokens.data.copy_(embed.weight.data[num_fixed:])
        
    def forward(self, input):
        weight = torch.cat([self.fixed_tokens, self.tuned_tokens], dim=0)
        return F.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)