"""Module with positional and word embeddings for candidates."""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class NeighbourEmbedding(nn.Module):
    
    def __init__(self, vocab_size, dimension):
        super().__init__()
        
        self.word_embed = nn.Embedding(vocab_size, dimension)
        self.linear1 = nn.Linear(2, 128)
        self.linear2 = nn.Linear(128, dimension)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
    
    def forward(self, word, position):
        
        embedding = self.word_embed(word)
        
        pos = F.relu(self.linear1(position))
        pos = self.dropout1(pos)
        pos = F.relu(self.linear2(pos))
        pos = self.dropout1(pos)
        
        concat = torch.cat((embedding, pos), dim=-1)
        
        return concat

class Embedder(nn.Module):
    
    def __init__(self, vocab_size, dimension, n_neighbours):
        super().__init__()
        
        self.neighbour_embeddings = []
        self.n_neighbours = n_neighbours
        for idx in range(self.n_neighbours):
            self.neighbour_embeddings.append(NeighbourEmbedding(vocab_size, dimension))
            
    def forward(self, words, positions):
        embedding_outputs = []
        for idx in range(self.n_neighbours):
            embedding_outputs.append(self.neighbour_embeddings[idx](words[:, idx], positions[:, idx, :]))
        
        return torch.cat(embedding_outputs, dim=1)