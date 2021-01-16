"""Main model skeleton"""

import torch
from torch import nn

from network import embedding

class ReLIE(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, n_neighbours):
        super().__init__()
        
        # Field Embedding (Probably One hot encoded data)
        
        # Candidate Embedding Layer
        # One dense layer
        
        # Neighbour Word Embedding and Positional Embedding
        # Word Embedding - Embedding layer
        # Positional Embedding - 2 blocks of (Dense, Relu, Dropout, Dense, Relu, Dropout)
        # N number of Neighbours
        
        # Each neighbour goes through a Neighbour Encoding Attention layer of its own
        
        # All the neighbour encodings are concatenated together into one block
        # This concatenated encoding is concatenated with Candidate Positon Encoding for Candidate Encoding
        
        # This candidate encoding is merged with Field Embedding and produces a binary output score
        
        self.cand_embed = nn.Linear(2, 128)
        self.field_embed = nn.Linear(2, 128)
        
        self.neighbour_embeddings = embedding.Embedder(vocab_size, embedding_dim, n_neighbours)
        
        
    
    def forward(self, x):
        neighbour_words = x.view([-1,6,3])[:,1:,:1].type(torch.LongTensor).view(-1, 5)
        neighbour_cords = x.view([-1,6,3])[:,1:,1:]
        
        embeds = self.neighbour_embeddings(neighbour_words, neighbour_cords)
        
        return embeds