import torch
import torch.nn as nn
from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout, CosineSimilarity
import torch.linalg as la
import numpy as np
from IPython import embed
from time import time 
from einops import rearrange, repeat
from torch.nn.init import xavier_uniform_, constant_
import math 

def SinusoidalPositionalEmbeddings(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]
    ran = np.arange(n_seq)
    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in ran])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table

#https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position, device):
        super().__init__()
        self.device = device
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, final_mat):
        embeddings = self.embeddings_table[final_mat]
        return embeddings.to(self.device)

class MultiheadAttention_Shaw(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, max_p, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.device = device
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = max_p

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position, device)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position, device)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, final_mat, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch_size, head, len_q,  head_dim]
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 3, 1) # [batch_size, head, head_dim, len_k]
        attn1 = torch.matmul(r_q1, r_k1) # [batch_size, head, len_q, len_k]

        r_q2 = query.view(batch_size, -1, self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(final_mat) # [batch_size, len_q, len_k, self.head_dim]
        attn2 = torch.matmul(r_q2, r_k2.transpose(-1, -2)).transpose(1, 2) # [batch_size, head, len_q, len_k]
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e32)
        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) #[batch_size, head, len_v, head_dim]
        weight1 = torch.matmul(attn, r_v1) #[batch_size, head, len_q, head_dim]
        r_v2 = self.relative_position_v(final_mat) # [batch_size, len_q, len_k, self.head_dim]
        weight2 = torch.matmul(attn.transpose(1, 2), r_v2).transpose(1, 2) # [batch_size, self.n_heads, len_q, self.head_dim]

        x = weight1 + weight2 #x = [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous() #x = [batch size, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim) #x = [batch size, query len, hid dim]
        x = self.fc_o(x) #x = [batch size, query len, hid dim]

        return x

#https://nn.labml.ai/transformers/rope/index.html
#https://github.com/JunnYu/RoFormer_pytorch/blob/roformer_v2/src/roformer/modeling_roformer.py
#https://github.com/lucidrains/rotary-embedding-torch/blob/517ee2cfeb10602032ef9d282c19851e19dd8943/rotary_embedding_torch/rotary_embedding_torch.py#L57

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d, base = 10_000, device = None):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = d
        self.freqs = None 
        self.device = device
        self._build_cache()

    def _build_cache(self):
        """
        x: [batch, head, seq_len, head_dim]
        Cache $\cos$ and $\sin$ values
        """
        # pos = torch.arange(seq_len).to(self.device)
        pos = torch.tensor([1]).to(self.device)

        # Return if cache is already built
        if self.freqs is not None and seq_len <= self.freqs.shape[0]:
            return
        # Get sequence length
        self.freqs = 1./  (self.base **(torch.arange(0, self.d, 2)[:(self.d//2)].float().to(self.device)/self.d))
        # self.freqs = self.freqs*10**4
        #pos @ self.freqs.T
        self.freqs = torch.einsum("..., f -> ... f", pos.type(self.freqs.dtype), self.freqs) # seq_len, dim//2 
        #seq_len, dim//2 -> seq_len, dim
        self.freqs = repeat(self.freqs, "... n -> ... (n r)", r=2)
        #unsqueeze
        # self.freqs = rearrange(self.freqs, "n d -> () () n d") # 1, 1, seq_len, dim 

    def rotate_half(self, x):
        x = rearrange(x, '... (d r) -> ... d r', r = 2)
        x1, x2 = x.unbind(dim = -1)
        x = torch.stack((-x2, x1), dim = -1)
        return rearrange(x, '... d r -> ... (d r)')

    def forward(self, t, diff, start_index = 0):
        b, head_num, s, head_dim = t.shape
        # t : [batch, head, seq_len, head_dim]
        # diff : [batch, seq_len]
        # self.freqs : [max_pos, head_dim]
        self.freqs = self.freqs.to(t) # device matching 
        diff_freqs = (diff*100).repeat(1,head_num*head_dim).view(b, head_num, s, head_dim)*self.freqs.squeeze() #[ batch, head, seq_len, head_dim ]
        rot_dim = self.freqs.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        # none, t, none
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * diff_freqs.cos()) + (self.rotate_half(t) * diff_freqs.sin())
        return torch.cat((t_left, t, t_right), dim = -1)

class MultiHeadAttention_Rotary(nn.Module):
    """
    ## Multi-head attention with rotary positional embeddings
    We override [multi-head attention from original transformer](../mha.html).
    """
    def __init__(self, d_model: int, heads: int, dropout_prob: float, max_p: int, bias=True, device=None):
        super().__init__()

        self.num_heads = heads
        self.head_dim = d_model // heads
        self.proj_bias = bias
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Rotary positional embedding layers
        self.rpe = RotaryPositionalEmbeddings(self.head_dim, max_p, device = device)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.query.weight)
        xavier_uniform_(self.key.weight)
        xavier_uniform_(self.value.weight)

        if self.proj_bias:
            constant_(self.query.bias, 0.)
            constant_(self.key.bias, 0.)
            constant_(self.value.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, diff, mask = None):
        batch_size = q.size(0)
        q = self.query(q).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.head_dim)

        q = self.rpe(q.transpose(1, 2), diff) # [batch_size, head, len_q,  head_dim]
        # q = q.transpose(1, 2) 
        k = self.rpe(k.transpose(1, 2), diff) # [batch_size, head, len_k,  head_dim]
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e32)
        attn = self.dropout(torch.softmax(attn, dim = -1)) # [batch_size, head, len_q,  len_k]

        v = v.transpose(1, 2) # [batch_size, head, len_v,  head_dim]
        # v = self.rpe(v.transpose(1, 2), diff) # [batch_size, head, len_k,  head_dim]
        output = torch.matmul(attn, v) # [batch_size, head, len_q,  head_dim]
        output = output.permute(0, 2, 1, 3).contiguous() #x = [batch size, query len, n heads, head dim]
        output = output.view(batch_size, -1, self.d_model) #x = [batch size, query len, hid dim]
        output = self.out_proj(output)

        return output