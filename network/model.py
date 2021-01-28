"""Main model skeleton"""

import torch
from torch import nn
import torch.nn.functional as F

from network.neighbour_attention import MultiHeadAttention
from network.neighbour_embedding import NeighbourEmbedding


class Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim, neighbours, heads):
        super().__init__()

        self.cand_embed = nn.Linear(2, 128)
        self.field_embed = nn.Linear(3, embedding_dim)
        self.embedding_dimension = embedding_dim
        self.neighbour_embeddings = NeighbourEmbedding(vocab_size, embedding_dim)

        self.attention_encodings = MultiHeadAttention(heads, embedding_dim * 2)
        self.linear_projection = nn.Linear(neighbours * embedding_dim * 2, 4 * embedding_dim * 2)
        self.linear_projection_2 = nn.Linear(128 + (2 * embedding_dim), embedding_dim)
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, field_id, candidate, neighbour_words, neighbour_positions, masks):
        # Field and candidate embeddings
        id_embed = self.field_embed(field_id)
        cand_embed = self.cand_embed(candidate)

        # Neighbour embeddings
        neighbour_embeds = self.neighbour_embeddings(neighbour_words, neighbour_positions)

        # Attention encodings
        self_attention = self.attention_encodings(neighbour_embeds, neighbour_embeds, neighbour_embeds, mask=masks)

        # Linear projection of attention to concatenate with candidate embedding
        bs = self_attention.size(0)
        self_attention = self_attention.view(bs, -1)
        linear_proj = F.relu(self.linear_projection(self_attention))

        linear_proj = linear_proj.view(bs, 4, -1)

        pooled_attention = F.max_pool2d(linear_proj, 2, 2)

        unrolled_attention = pooled_attention.view(bs, -1)

        # Concatenating Candidate embedding and Attention
        concat = torch.cat((cand_embed, unrolled_attention), dim=1)

        # Re-projecting concatenated embedding to calculate cosing similarity
        projected_candidate_encoding = F.relu(self.linear_projection_2(concat))

        # Calculating cosine similarity and scaling to [0,1]
        similarity = self.cos_sim(id_embed, projected_candidate_encoding).view(bs, -1)
        scores = (similarity + 1) / 2

        return scores
