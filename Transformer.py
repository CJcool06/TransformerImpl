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
    """
    Instead of performing a single attention function (say, model_dimension=512 for queries, keys, and values),
    we project the queries, keys, and values h times with different linear projections to dk, dk, and dv 
    dimensions (say, 64), respectively.

    Essentially, instead of using one large attention function, we use a number (say, 8) of individual 
    attention functions that have been projected down to a dimension of model_dimension/heads 
    (in our case, 512/8=64).

    We then perform the attention function in parallel, yielding dv-dimensional outputs. These are 
    concatenated to dv*heads dimensions and once again projected. The final output has model_dimension 
    dimensions.
    """

    def __init__(self, heads: int, keys_dimension: int, values_dimension: int, masked: bool = False):
        super().__init__()
        self.heads = heads
        self.keys_dimension = keys_dimension
        self.values_dimension = values_dimension
        self.model_dimension = keys_dimension * heads
        self.masked = masked
        self.words = 1
        
        self.queries_projections = nn.ModuleList([nn.Linear(keys_dimension, self.keys_dimension) for _ in range(heads)])
        self.keys_projections = nn.ModuleList([nn.Linear(keys_dimension, self.keys_dimension) for _ in range(heads)])
        self.values_projections = nn.ModuleList([nn.Linear(values_dimension, self.values_dimension) for _ in range(heads)])

        self.multihead_weights = nn.Linear(self.model_dimension, self.model_dimension)
    
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
        
        # print(heads[0])
        return self.multihead_weights(torch.concat(heads, dim=1))
    
    def ScaledDotProductAttention(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """
        Compute the dot products of the query with all keys, divide each by sqrt(keys_dim), 
        apply a softmax function to obtain the weights on the values, and dot product with 
        the values.

        For large values of keys_dim, it is suspected suspected that the dot products 
        grow large in magnitude, pushing the softmax function into regions where it 
        has extremely small gradients. To counteract this, we scale the dot products 
        by 1/keys_dim.

        A mask is applied to ignore padding in inputs.

        Parameters:
            queries:    n x dk matrix
            keys:       m x dk matrix
            values:     m x dv matrix
            mask:       n x m  matrix

        Returns:
            n x dv matrix

        """

        # Calculate the combinations of the inputs.
        scaled_dot_product = (queries @ keys.T) / math.sqrt(self.keys_dimension)

        # Mask out the combinations that contain tokens we shouldn't be able to see.
        if self.masked:
            scaled_dot_product -= 1e9 * self.LookAheadMask(scaled_dot_product)

        return nn.functional.softmax((queries @ keys.T) / math.sqrt(self.keys_dimension), dim=1) @ values
    
    def LookAheadMask(self, dot_product):
        """
        We need to prevent leftward information flow in the decoder. An upper matrix of ones 
        is generated and then rotated to the bottom right. This produces a mask that will 
        prevent using information from future tokens the a sequence.
        """

        return torch.Tensor(np.rot90(np.triu(np.ones(dot_product.shape), dot_product.shape[0] - self.words), -1).copy())