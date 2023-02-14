# This code is based on the AKT implementation: https://github.com/arghosh/AKT/blob/master/akt.py
import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Parameter,
    Linear,
    GELU,
    ReLU,
    LayerNorm,
    Dropout,
    Softplus,
    Embedding,
)
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import torch.nn.functional as F
import numpy as np
from enum import IntEnum
from .rpe import RotaryPositionalEmbeddings, ALiBiPositionalEmbeddings
from IPython import embed 
import math 

from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import pandas as pd

class CL4KTTransformerLayer(Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, seq_len, de_type="none_0", bincounts=None):
        super(CL4KTTransformerLayer, self).__init__()
        """
            This is a Basic Block of Transformer paper.
            It contains one Multi-head attention object.
            Followed by layer norm and position-wise feed-forward net and dropotu layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttentionWithIndividualFeatures(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, seq_len=seq_len, de_type=de_type, bincounts=bincounts
        )

        # Two layer norm and two dropout layers
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        self.linear1 = Linear(d_model, d_ff)
        self.activation = GELU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)

        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, mask, query, key, values, diff=None, response=None, apply_pos=True):
        """
        Input:
            block: object of type BasicBlock(nn.Module). It contains maksed_attn_head objects which is of type MultiHeadAttnetion(nn.Module).
            mask: 0 means that it can peek (엿보다) only past values. 1 means that block can peek only current and past values
            query: Queries. In Transformer paper it is the input for both encoder and decoder
            key: Keys. In transformer paper it is the input for both encoder and decoder
            values: Values. In transformer paper it is the input for encoder and encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the alyer andr returned
        """

        batch_size, seqlen = query.size(0), query.size(1)
        """
        when mask==1
        >>> nopeek_mask (for question encoder, knoweldge encoder)
            array([[[[0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0]]]], dtype=uint8)

         >>> src_mask
            tensor([[[[ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]]]])

        when mask==0 (for knowledge retriever)
        >>> nopeek_mask
            array([[[[1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1]]]], dtype=uint8)

        >>> src_mask
            tensor([[[[False, False, False, False, False],
                    [ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False]]]])

        row: target, col: source
        """
        device = query.get_device()
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")

        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        bert_mask = torch.ones_like(src_mask).bool()

        if mask == 0:
            query2, attn = self.masked_attn_head(query, key, values, diff=diff, response=response, mask=src_mask)
        elif mask == 1:
            query2, attn = self.masked_attn_head(query, key, values, diff=diff, response=response, mask=src_mask)
        else:  # mask == 2
            query2, attn = self.masked_attn_head(query, key, values, diff=diff, response=response, mask=bert_mask)

        query = query + self.dropout1((query2))  # residual connection
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)

        return query, attn


class AKTTransformerLayer(Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, seq_len, de_type="none_0", bincounts=None):
        super(AKTTransformerLayer, self).__init__()
        """
            This is a Basic Block of Transformer paper.
            It contains one Multi-head attention object.
            Followed by layer norm and position-wise feed-forward net and dropotu layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttentionWithContextDistance(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, seq_len=seq_len, de_type=de_type, bincounts=bincounts)

        # Two layer norm and two dropout layers
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        self.linear1 = Linear(d_model, d_ff)
        self.activation = ReLU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)

        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, mask, query, key, values, diff=None, response=None, apply_pos=True):
        """
        Input:
            block: object of type BasicBlock(nn.Module). It contains maksed_attn_head objects which is of type MultiHeadAttnetion(nn.Module).
            mask: 0 means that it can peek (엿보다) only past values. 1 means that block can peek only current and past values
            query: Queries. In Transformer paper it is the input for both encoder and decoder
            key: Keys. In transformer paper it is the input for both encoder and decoder
            values: Values. In transformer paper it is the input for encoder and encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the alyer andr returned
        """

        batch_size, seqlen = query.size(0), query.size(1)
        """
        when mask==1
        >>> nopeek_mask (for question encoder, knoweldge encoder)
            array([[[[0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0]]]], dtype=uint8)

         >>> src_mask
            tensor([[[[ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]]]])

        when mask==0 (for knowledge retriever)
        >>> nopeek_mask
            array([[[[1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1]]]], dtype=uint8)

        >>> src_mask
            tensor([[[[False, False, False, False, False],
                    [ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False]]]])

        As a result, the upper triangular elements are masked
        row: target, col: source
        """
        device = query.get_device()
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")

        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        if mask == 0:
            query2, attn = self.masked_attn_head(query, key, values, diff=diff, response=response, mask=src_mask)
        elif mask == 1:
            query2, attn = self.masked_attn_head(query, key, values, diff=diff, response=response, mask=src_mask)
        else:  # mask == 2
            raise NotImplementedError

        query = query + self.dropout1((query2))  # residual connection
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)

        return query, attn


class MultiHeadAttentionWithIndividualFeatures(Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, seq_len, de_type="none_0", bias=True, bincounts=None):
        super(MultiHeadAttentionWithIndividualFeatures, self).__init__()
        """
        It has projection layer for getting keys, queries, and values. Followed by attention and a connected layer.
        """

        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = Linear(d_model, d_model, bias=bias)
        self.k_linear = Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = Linear(d_model, d_model, bias=bias)
        self.dropout = Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = Linear(d_model, d_model, bias=bias)
        self.gammas = Parameter(torch.zeros(n_heads, 1, 1))
        
        self.de_type = de_type
        if self.de_type.startswith("rotary"):
            self.rpe = RotaryPositionalEmbeddings(d_model // n_heads, max_len=seq_len)
        if self.de_type.startswith("alibi"):
            self.score = ALiBiPositionalEmbeddings(n_heads, de_type, bincounts=bincounts, max_len=seq_len)
            
        xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, diff=None, response=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions batch_size * num_heads * seqlen * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.de_type.startswith("rotary") and diff is not None:
            k = self.rpe(k, diff) # [batch_size, head, len_k,  head_dim]
            q = self.rpe(q, diff) # [batch_size, head, len_q,  head_dim]
            # if "v" in self.de_type :
            #     v = self.rpe(v, diff) # [batch_size, head, len_q,  head_dim]
            scores, attn_scores = attention(q, k, v, mask=mask, dropout=self.dropout)
            
        elif self.de_type.startswith("alibi") and diff is not None:
            score_mask = self.score.buffered_future_mask(q, diff, response)
            if "1" in self.de_type.split('_')[0]: #attention에 position alibi를 반영하는 경우 
                scores, attn_scores = attention(q, k, v, score_mask=score_mask,
                                        mask=mask, dropout=self.dropout)
            else:
                gammas = self.gammas
                scores, attn_scores = individual_attention(
                    q, k, v, self.d_k, mask, self.dropout, gammas, score_mask=score_mask
                )
        elif self.de_type.startswith("basic"):
            scores, attn_scores = attention(q, k, v, 
                                    mask=mask, dropout=self.dropout)
        else:
            # calculate attention using function we will define next
            gammas = self.gammas
            scores, attn_scores = individual_attention(
                q, k, v, self.d_k, mask, self.dropout, gammas
            )

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        # concat torch.Size([24, 200, 256])   [batch_size, seqlen, d_model]
        # print('concat', concat.shape)
        output = self.out_proj(concat)

        return output, attn_scores


class MultiHeadAttentionWithContextDistance(Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, seq_len, de_type="none_0", bincounts=None, bias=True):
        super(MultiHeadAttentionWithContextDistance, self).__init__()
        """
        It has projection layer for getting keys, queries, and values. Followed by attention and a connected layer.
        """

        # d_feature=d_model // n_heads
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = Linear(d_model, d_model, bias=bias)
        self.k_linear = Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = Linear(d_model, d_model, bias=bias)
        self.dropout = Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = Linear(d_model, d_model, bias=bias)
        self.gammas = Parameter(torch.zeros(n_heads, 1, 1))

        self.de_type = de_type
        if self.de_type.startswith("rotary"):
            self.rpe = RotaryPositionalEmbeddings(d_model // n_heads, max_len=seq_len)
        if self.de_type.startswith("alibi"):
            self.score = ALiBiPositionalEmbeddings(n_heads, de_type, bincounts=bincounts, max_len=seq_len)
            
        xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, diff=None, response=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions batch_size * num_heads * seqlen * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.de_type.startswith("rotary") and diff is not None:
            k = self.rpe(k, diff) # [batch_size, head, len_k,  head_dim]
            q = self.rpe(q, diff) # [batch_size, head, len_q,  head_dim]
            # if "v" in self.de_type :
            #     v = self.rpe(v, diff) # [batch_size, head, len_q,  head_dim]
            scores, attn = attention(q, k, v, mask=mask, dropout=self.dropout)
            
        elif self.de_type.startswith("alibi") and diff is not None:
            score_mask = self.score.buffered_future_mask(q, diff, response)
            if "1" in self.de_type.split('_')[0]: #attention에 position alibi를 반영하는 경우 
                scores, attn = attention(q, k, v, score_mask=score_mask,
                                        mask=mask, dropout=self.dropout)
            else:
                gammas = self.gammas
                scores, attn = monotonic_attention(
                    q, k, v, self.d_k, mask, self.dropout, gammas, score_mask=score_mask
                )
        elif self.de_type.startswith("basic"):
            scores, attn = attention(q, k, v, 
                                    mask=mask, dropout=self.dropout)
        else:
            # calculate attention using function we will define next
            gammas = self.gammas
            scores, attn = monotonic_attention(
                q, k, v, self.d_k, mask, self.dropout, gammas
            )
                
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        # concat torch.Size([24, 200, 256])   [batch_size, seqlen, d_model]
        # print('concat', concat.shape)
        output = self.out_proj(concat)

        return output, attn


def individual_attention(q, k, v, d_k, mask, dropout, gamma=None, score_mask=None):
    """
    This is called by MultiHeadAttention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(
        d_k
    )  # [batch_size, 8, seq_len, seq_len]
    if score_mask is not None:
        scores += score_mask
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float()

        distcum_scores = torch.cumsum(scores_, dim=-1)

        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)

        device = distcum_scores.get_device()
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor)
        position_effect = position_effect.to(device)

        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

        m = Softplus()
        gamma = -1.0 * m(gamma).unsqueeze(0)

        total_effect = torch.clamp(
            torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
        )

        scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    attn_scores = scores
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, attn_scores


def monotonic_attention(q, k, v, d_k, mask, dropout, gamma=None, score_mask=None):
    """
    This is called by MultiHeadAttention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(
        d_k
    )  # [batch_size, 8, seq_len, seq_len]
    if score_mask is not None:
        scores += score_mask
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # [batch_size, 8, seqlen, seqlen]
        scores_ = scores_ * mask.float()

        # [batch_size, 8, seqlen, seqlen]
        distcum_scores = torch.cumsum(scores_, dim=-1)
        # [batch_size, 8, seqlen, 1]
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        """
        >>> x1-x2
            tensor([[ 0,  1,  2,  3,  4],
                    [-1,  0,  1,  2,  3],
                    [-2, -1,  0,  1,  2],
                    [-3, -2, -1,  0,  1],
                    [-4, -3, -2, -1,  0]])

        >>> torch.abs(x1-x2)
            tensor([[0, 1, 2, 3, 4],
                    [1, 0, 1, 2, 3],
                    [2, 1, 0, 1, 2],
                    [3, 2, 1, 0, 1],
                    [4, 3, 2, 1, 0]])
        """
        device = distcum_scores.get_device()
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(
            torch.FloatTensor
        )  # [1, 1, seqlen, seqlen]
        position_effect = position_effect.to(device)
        # [batch_size, 8, seqlen, seqlen] positive distance
        # dist_score => d(t, tau)
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

        m = Softplus()
        # 1,8,1,1  gamma is \theta in the paper (learnable decay rate parameter)
        gamma = -1.0 * m(gamma).unsqueeze(0)
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e-5
        total_effect = torch.clamp(
            torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
        )

        scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # [batch_size, 8, seq_len, seq_len]
    attn = scores

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, attn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        self.postional_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.postional_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class CosinePositionalEmbedding(Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(torch.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class BERTEmbeddings(nn.Module):
    def __init__(self, num_skills, hidden_size, seq_len, dropout, padding_idx=0):
        super(BERTEmbeddings, self).__init__()
        self.item_embeddings = Embedding(
            num_skills, hidden_size, padding_idx=padding_idx
        )
        self.positional_embeddings = Embedding(seq_len, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        item_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.positional_embeddings(position_ids)
        embeddings = item_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
                Linear(self.emb_size, self.emb_size),
                ReLU(),
                Dropout(self.dropout),
                Linear(self.emb_size, self.emb_size),
                # Dropout(self.dropout),
            )
    def forward(self, in_fea):
        return self.FFN(in_fea)

def ut_mask(device, seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)

def lt_mask(device, seq_len):
    """ Upper Triangular Mask
    """
    return torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).to(dtype=torch.bool).to(device)

def pos_encode(device, seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0).to(device)

def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def attention(query, key, value, score_mask=None, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if score_mask is not None:
        scores += score_mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h, dropout, seq_len, de_type="none_0", bincounts=None):
        "Take in model size and number of heads."
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=False), 4) # Q, K, V, last
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.de_type = de_type
        if self.de_type.startswith("rotary"):
            self.rpe = RotaryPositionalEmbeddings(self.d_k, max_len=seq_len-1)
        if self.de_type.startswith("alibi"):
            self.score = ALiBiPositionalEmbeddings(h, de_type, max_len=seq_len-1, bincounts=bincounts)

    def forward(self, query, key, value, diff=None, response=None, mask=None):
        "Implements Figure 2"
        # if mask is not None:
        #     # Same mask applied to all h heads.
        #     mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
            
        if self.de_type.startswith("rotary") and diff is not None:
            query = self.rpe(query, diff[:, 1:]) # [batch_size, head, len_q,  head_dim]
            key = self.rpe(key, diff[:, :-1]) # [batch_size, head, len_k,  head_dim]
            # if "v" in self.de_type :
            #     value = self.rpe(value, diff) # [batch_size, head, len_q,  head_dim]

        elif self.de_type.startswith("alibi") and diff is not None:
            # 2) Apply attention on all the projected vectors in batch.
            score_mask = self.score.buffered_future_mask_sakt(query, diff, response)
            x, self.attn = attention(query, key, value, score_mask=score_mask,
                                    mask=mask, dropout=self.dropout)
        else:
            # 2) Apply attention on all the projected vectors in batch.
            x, self.attn = attention(query, key, value, mask=mask,
                                    dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn