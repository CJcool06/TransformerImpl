import numpy as np
import torch
import torch.nn as nn
import math


class Encoder(nn.Module):

    def __init__(self, heads: int, keys_dimension: int, values_dimension: int):
        super().__init__()
        self.heads = heads
        self.keys_dimension = keys_dimension
        self.values_dimension = values_dimension



    def forward(self):
        pass

class Decoder(nn.Module):

    def __init__(self, heads: int, keys_dimension: int, values_dimension: int):
        super().__init__()
        self.heads = heads
        self.keys_dimension = keys_dimension
        self.values_dimension = values_dimension



    def forward(self):
        pass

class EncoderLayer(nn.Module):

    def __init__(self, heads: int, keys_dimension: int, values_dimension: int):
        super().__init__()
        self.heads = heads
        self.keys_dimension = keys_dimension
        self.values_dimension = values_dimension

        self.attention = MultiHeadAttention(heads, keys_dimension, values_dimension)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        result = self.attention(queries, keys, values)
        # result = nn.functional.layer_norm(result + )

class DecoderLayer(nn.Module):

    def __init__(self, heads: int, keys_dimension: int, values_dimension: int):
        super().__init__()
        self.heads = heads
        self.keys_dimension = keys_dimension
        self.values_dimension = values_dimension



    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        pass

class MultiHeadAttention(nn.Module):

    def __init__(self, heads: int, keys_dimension: int, values_dimension: int):
        super().__init__()
        self.heads = heads
        self.keys_dimension = keys_dimension
        self.values_dimension = values_dimension
        self.model_dimension = keys_dimension * heads
        
        self.queries_projections = nn.ModuleList([nn.Linear(keys_dimension, self.model_dimension) for _ in range(heads)])
        self.keys_projections = nn.ModuleList([nn.Linear(keys_dimension, self.model_dimension) for _ in range(heads)])
        self.values_projections = nn.ModuleList([nn.Linear(values_dimension, self.model_dimension) for _ in range(heads)])

        self.multihead_weights = nn.Linear(heads * values_dimension, self.model_dimension)
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """
        queries:    n x dk matrix
        keys:       m x dk matrix
        values:     l x dv matrix
        """

        heads = []

        for head in range(self.heads):
            heads.append(
                self.ScaledDotProductAttention(
                    self.queries_projections[head](queries),
                    self.keys_projections[head](keys),
                    self.values_projections[head](values)
                )
            )
        
        return self.multihead_weights(torch.concat(heads, dim=0).sum(dim=0).reshape(1, self.model_dimension))
    
    def ScaledDotProductAttention(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """
        Compute the dot products of the query with all keys, divide each by sqrt(keys_dim), 
        apply a softmax function to obtain the weights on the values, and dot product with 
        the values.

        For large values of keys_dim, it is suspected suspected that the dot products 
        grow large in magnitude, pushing the softmax function into regions where it 
        has extremely small gradients. To counteract this, we scale the dot products 
        by 1/keys_dim.

        Parameters:
            queries:    n x dk matrix
            keys:       m x dk matrix
            values:     m x dv matrix

        Returns:
            n x dv matrix

        """

        return nn.functional.softmax((queries @ keys.T) / math.sqrt(self.keys_dimension), dim=1) @ values
    