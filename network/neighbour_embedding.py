"""Module with positional and word embeddings for candidates."""

import torch
import torch.nn.functional as F
from torch import nn


class NeighbourEmbedding(nn.Module):

    def __init__(self, vocab_size, dimension):
        super().__init__()

        self.word_embed = nn.Embedding(vocab_size, dimension)
        self.linear1 = nn.Linear(2, 128)
        self.linear2 = nn.Linear(128, dimension)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, words, positions):

        # Word embedding
        embedding = self.word_embed(words)

        # Position embedding
        pos = F.relu(self.linear1(positions))
        pos = self.dropout1(pos)
        pos = F.relu(self.linear2(pos))
        pos = self.dropout1(pos)

        # Concatenating word and position embeddings
        neighbour_embedding = torch.cat((embedding, pos), dim=2)

        return neighbour_embedding
