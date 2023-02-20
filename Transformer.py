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
        self.model_dimension = keys_dimension * heads



    def forward(self):
        pass

class Decoder(nn.Module):

    def __init__(self, heads: int, keys_dimension: int, values_dimension: int):
        super().__init__()
        self.heads = heads
        self.keys_dimension = keys_dimension
        self.values_dimension = values_dimension
        self.model_dimension = keys_dimension * heads



    def forward(self):
        pass

class EncoderLayer(nn.Module):
    """
    An encoder layer consists of:
    - Multi-head attention
    - Residual connection + layer-norm
    - Feed forward network
    - Residual connection + layer-norm
    """

    def __init__(self, heads: int, keys_dimension: int, values_dimension: int):
        super().__init__()
        self.heads = heads
        self.keys_dimension = keys_dimension
        self.values_dimension = values_dimension
        self.model_dimension = keys_dimension * heads

        self.attention = MultiHeadAttention(heads, keys_dimension, values_dimension)
        self.layer1 = nn.Linear(self.model_dimension, self.model_dimension * 4)
        self.layer2 = nn.Linear(self.model_dimension * 4, self.model_dimension)

    def forward(self, input: torch.Tensor):

        # Attention
        result = self.attention(input, input, input)

        # Residual connection and layer-norm.
        add_and_norm = nn.functional.layer_norm(result + input, result.shape)

        # Feed forward.
        result = self.FeedForwardNN(add_and_norm)

        # Residual connection and layer-norm.
        add_and_norm = nn.functional.layer_norm(result + add_and_norm, result.shape)

        return add_and_norm
    
    def FeedForwardNN(self, input):
        """
        Passes the weighted attention through a feed forward network consisting of 
        a hidden layer and an output layer. The hidden layer has model_dimension*4 
        dimensions and the output layer has model_dimension dimensions.
        """

        # Linear + ReLU
        result = nn.functional.relu(self.layer1(input))

        # Linear
        result = self.layer2(result)

        return result


class DecoderLayer(nn.Module):

    def __init__(self, heads: int, keys_dimension: int, values_dimension: int):
        super().__init__()
        self.heads = heads
        self.keys_dimension = keys_dimension
        self.values_dimension = values_dimension
        self.model_dimension = keys_dimension * heads


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
        
        self.queries_projections = nn.ModuleList([nn.Linear(self.model_dimension, self.keys_dimension) for _ in range(heads)])
        self.keys_projections = nn.ModuleList([nn.Linear(self.model_dimension, self.keys_dimension) for _ in range(heads)])
        self.values_projections = nn.ModuleList([nn.Linear(self.model_dimension, self.values_dimension) for _ in range(heads)])

        self.multihead_weights = nn.Linear(self.model_dimension, self.model_dimension)
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """
        Parameters:
            queries:    n x dk matrix
            keys:       m x dk matrix
            values:     l x dv matrix

        Output:
            n x dv matrix
        """

        heads = []

        # Perform scaled dot-product attention for each head
        for head in range(self.heads):
            heads.append(
                self.ScaledDotProductAttention(
                    self.queries_projections[head](queries),
                    self.keys_projections[head](keys),
                    self.values_projections[head](values)
                )
            )
        
        # Pass through projection layer
        projection = self.multihead_weights(torch.concat(heads, dim=1))

        return projection
    
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

        return nn.functional.softmax(scaled_dot_product, dim=1) @ values
    
    def LookAheadMask(self, dot_product):
        """
        We need to prevent leftward information flow in the decoder. An upper matrix of ones 
        is generated and then rotated to the bottom right. This produces a mask that will 
        prevent using information from future tokens the a sequence.
        """

        return torch.Tensor(np.rot90(np.triu(np.ones(dot_product.shape), dot_product.shape[0] - self.words), -1).copy())