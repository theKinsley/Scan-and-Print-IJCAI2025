import math
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import LongTensor, Tensor

def PositionalEncode(name, **kwargs):
    return {
        '1D': PositionalEncoding1d(**kwargs),
        '2D': PositionEmbeddingSine(**kwargs)
    }[name]

class ImageReshaper(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        assert x.size(1) == self.d_model, f"{x.size(1)} != {self.d_model}"
        x = rearrange(x, "b c h w -> b (h w) c")
        return x
    
class PositionalEncoding1d(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = True,
        scale_input: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.scale_input = scale_input

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        if batch_first:
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: LongTensor) -> Tensor:
        """
        Args:
            x: Tensor, shape :
                [seq_len, batch_size, embedding_dim] (if batch_first)
                [batch_size, seq_len, embedding_dim] (else)
        """
        h = x * math.sqrt(self.d_model) if self.scale_input else x
        if self.batch_first:
            S = h.size(1)
            h = h + self.pe[:, :S]
        else:
            S = x.size(0)
            h = h + self.pe[:S]
        return self.dropout(h)  # type: ignore

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, d_model=64, temperature=10000, normalize=False, scale=None
    ) -> None:
        super().__init__()
        self.d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.reshape = ImageReshaper(d_model)

    def forward(self, input: LongTensor) -> Tensor:
        bs, c, h, w = input.size()  # [bs, h*w, c]
        y, x = torch.meshgrid(
            torch.arange(h).type_as(input),
            torch.arange(w).type_as(input),
            indexing="ij",
        )

        if self.normalize:
            y = y / (h - 1)  # Normalize y coordinates to [0, 1]
            x = x / (w - 1)  # Normalize x coordinates to [0, 1]
            y = y * self.scale
            x = x * self.scale
        dim_t = torch.arange(self.d_model).type_as(input)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.d_model
        )

        pos_x = x.flatten()[None, :, None] / dim_t
        pos_y = y.flatten()[None, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).repeat(bs, 1, 1)

        output = self.reshape(input) + pos

        return output