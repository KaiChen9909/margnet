import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential

class TimeEmbedding(Module):
    def __init__(self, embedding_dim):
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.time_fc = Linear(embedding_dim, embedding_dim//2)

    def forward(self, t):
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)  # (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch_size, embedding_dim)
        emb = self.time_fc(emb)
        return emb


class Residual(Module):
    def __init__(self, i, o, embedding_dim):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()
        self.time_fc = Linear(embedding_dim, o)

    def forward(self, input, t_emb):
        out = self.fc(input)
        out = out + self.time_fc(t_emb)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class DenoiserModel(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        super(DenoiserModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.time_embedding = TimeEmbedding(2*embedding_dim)
        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [Residual(dim, item, embedding_dim)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input, t):
        t_emb = self.time_embedding(t)  # (batch_size, embedding_dim)
        data = self.seq[0](input, t_emb)
        for layer in self.seq[1:-1]:
            data = layer(data, t_emb)
        data = self.seq[-1](data)
        return data




class SimpleResidual(Module):
    def __init__(self, i, o):
        super(SimpleResidual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)

class Generator(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [SimpleResidual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data



# ModuleType = Union[str, Callable[..., nn.Module]]

# def reglu(x: Tensor) -> Tensor:
#     """The ReGLU activation function from [1].
#     References:
#         [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
#     """
#     assert x.shape[-1] % 2 == 0
#     a, b = x.chunk(2, dim=-1)
#     return a * F.relu(b)


# def geglu(x: Tensor) -> Tensor:
#     """The GEGLU activation function from [1].
#     References:
#         [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
#     """
#     assert x.shape[-1] % 2 == 0
#     a, b = x.chunk(2, dim=-1)
#     return a * F.gelu(b)

# class ReGLU(nn.Module):
#     """The ReGLU activation function from [shazeer2020glu].

#     Examples:
#         .. testcode::

#             module = ReGLU()
#             x = torch.randn(3, 4)
#             assert module(x).shape == (3, 2)

#     References:
#         * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
#     """

#     def forward(self, x: Tensor) -> Tensor:
#         return reglu(x)

# class GEGLU(nn.Module):
#     """The GEGLU activation function from [shazeer2020glu].

#     Examples:
#         .. testcode::

#             module = GEGLU()
#             x = torch.randn(3, 4)
#             assert module(x).shape == (3, 2)

#     References:
#         * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
#     """

#     def forward(self, x: Tensor) -> Tensor:
#         return geglu(x)


# def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
#     return (
#         (
#             ReGLU()
#             if module_type == 'ReGLU'
#             else GEGLU()
#             if module_type == 'GEGLU'
#             else getattr(nn, module_type)(*args)
#         )
#         if isinstance(module_type, str)
#         else module_type(*args)
#     )

# def timestep_embedding(timesteps, dim, max_period=10000):
#     """
#     Create sinusoidal timestep embeddings.

#     :param timesteps: a 1-D Tensor of N indices, one per batch element.
#                       These may be fractional.
#     :param dim: the dimension of the output.
#     :param max_period: controls the minimum frequency of the embeddings.
#     :return: an [N x dim] Tensor of positional embeddings.
#     """
#     half = dim // 2
#     freqs = torch.exp(
#         -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#     ).to(device=timesteps.device)
#     args = timesteps[:, None].float() * freqs[None]
#     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2:
#         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#     return embedding


# class MLP(nn.Module):
#     """The MLP model used in [gorishniy2021revisiting].

#     The following scheme describes the architecture:

#     .. code-block:: text

#           MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
#         Block: (in) -> Linear -> Activation -> Dropout -> (out)

#     Examples:
#         .. testcode::

#             x = torch.randn(4, 2)
#             module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
#             assert module(x).shape == (len(x), 1)

#     References:
#         * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
#     """

#     class Block(nn.Module):
#         """The main building block of `MLP`."""

#         def __init__(
#             self,
#             *,
#             d_in: int,
#             d_out: int,
#             bias: bool,
#             activation: ModuleType,
#             dropout: float,
#         ) -> None:
#             super().__init__()
#             self.linear = nn.Linear(d_in, d_out, bias)
#             # self.batch_norm = nn.BatchNorm1d(d_out)
#             self.activation = _make_nn_module(activation)
#             self.dropout = nn.Dropout(dropout)

#         def forward(self, x: Tensor) -> Tensor:
#             # return self.dropout(self.activation(self.batch_norm(self.linear(x))))
#             return self.dropout(self.activation(self.linear(x)))


#     def __init__(
#        self,
#        *,
#        d_in: int,
#        d_layers: List[int],
#        dropouts: Union[float, List[float]],
#        activation: Union[str, Callable[[], nn.Module]],
#        d_out: int
#    ) -> None:
#        super().__init__()
#        if isinstance(dropouts, float):
#            dropouts = [dropouts] * len(d_layers)
#        assert len(d_layers) == len(dropouts)
#        assert activation not in ['ReGLU', 'GEGLU']
 
#        self.blocks = nn.ModuleList(
#            [
#                MLP.Block(
#                    d_in=d_layers[i - 1] if i else d_in,
#                    d_out=d,
#                    bias=True,
#                    activation=activation,
#                    dropout=dropout,
#                )
#                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
#            ]
#        )
#        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

#     @classmethod
#     def make_baseline(
#         cls: Type['MLP'],
#         d_in: int,
#         d_layers: List[int],
#         dropout: float,
#         d_out: int,
#     ) -> 'MLP':
#         """Create a "baseline" `MLP`.

#         This variation of MLP was used in [gorishniy2021revisiting]. Features:

#         * :code:`Activation` = :code:`ReLU`
#         * all linear layers except for the first one and the last one are of the same dimension
#         * the dropout rate is the same for all dropout layers

#         Args:
#             d_in: the input size
#             d_layers: the dimensions of the linear layers. If there are more than two
#                 layers, then all of them except for the first and the last ones must
#                 have the same dimension. Valid examples: :code:`[]`, :code:`[8]`,
#                 :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`. Invalid
#                 example: :code:`[1, 2, 3, 4]`.
#             dropout: the dropout rate for all hidden layers
#             d_out: the output size
#         Returns:
#             MLP

#         References:
#             * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
#         """
#         assert isinstance(dropout, float)
#         if len(d_layers) > 2:
#             assert len(set(d_layers[1:-1])) == 1, (
#                 'if d_layers contains more than two elements, then'
#                 ' all elements except for the first and the last ones must be equal.'
#             )
#         return MLP(
#             d_in=d_in,
#             d_layers=d_layers,  # type: ignore
#             dropouts=dropout,
#             activation='ReLU',
#             d_out=d_out,
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         x = x.float()
#         for block in self.blocks:
#             x = block(x)
#         x = self.head(x)
#         return x


# class MLPDiffusion(nn.Module):
#     def __init__(self, d_in, rtdl_params, dim_t = 128):
#         super().__init__()
#         self.dim_t = dim_t

#         # d0 = rtdl_params['d_layers'][0]

#         rtdl_params['d_in'] = dim_t
#         rtdl_params['d_out'] = d_in

#         self.mlp = MLP.make_baseline(**rtdl_params)
        
#         self.proj = nn.Linear(d_in, dim_t)
#         self.time_embed = nn.Sequential(
#             nn.Linear(dim_t, dim_t),
#             nn.SiLU(),
#             nn.Linear(dim_t, dim_t)
#         )
    
#     def forward(self, x, timesteps):
#         if x.dtype != torch.float32:
#             x = x.to(torch.float32)
#         emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
#         x = self.proj(x) + emb
#         return self.mlp(x)