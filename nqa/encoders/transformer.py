"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from nqa.modules.util_class import LayerNorm
from nqa.modules.multi_head_attn import MultiHeadedAttention
from nqa.modules.position_ffn import PositionwiseFeedForward
from nqa.encoders.encoder import EncoderBase
from nqa.utils.misc import sequence_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, d_k, d_v, dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadedAttention(heads, d_model, d_k, d_v,
                                              dropout=dropout,
                                              max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.
        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        context, _, _ = self.attention(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    Returns:
        (`FloatTensor`, `FloatTensor`):
        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, max_relative_positions=0):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff,
                                     64, 64, dropout,
                                     max_relative_positions=max_relative_positions)
             for _ in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        out = src
        mask = None if lengths is None else \
            ~sequence_mask(lengths, out.shape[1]).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.layer[i](out, mask)
        out = self.layer_norm(out)

        return out
