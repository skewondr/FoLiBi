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

        diff_freqs = diff.repeat(1, head_num*head_dim).view(b, head_num, head_dim, s).transpose(2,3)
        diff_freqs = diff_freqs*self.freqs.squeeze() #[ batch, head, seq_len, head_dim ]

        rot_dim = self.freqs.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        # none, t, none
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * diff_freqs.cos()) + (self.rotate_half(t) * diff_freqs.sin())
        return torch.cat((t_left, t, t_right), dim = -1)

class ALiBiPositionalEmbeddings(nn.Module):
    def __init__(self, attn_heads, max_len=100, bs=512):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()
        self.max_len = max_len
        self.attn_heads = attn_heads
        self.slopes = torch.Tensor(self.get_slopes(attn_heads))
        self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_len).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1) #attn_heads, 1, max_len 
        self.alibi = self.alibi.view(attn_heads, 1, max_len)
        self.alibi = self.alibi.repeat(bs, 1, 1)  # batch_size*attn_heads, 1, max_len
        self._future_mask = torch.empty(0)
        
    def get_slopes(self, n):
        """return list of lengnth n"""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
        else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
            return get_slopes_power_of_2(closest_power_of_2) + self.get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
        
        
    def buffered_future_mask(self, tensor):
        dim = tensor.size(2)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(1) < self.max_len
        ):
            self._future_mask = torch.triu(
                self.fill_with_neg_inf(torch.zeros([self.max_len, self.max_len])), 1
            )
            self._future_mask = self._future_mask.unsqueeze(0) + self.alibi #batch_size*attn_heads, max_len, max_len 
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:tensor.shape[0]*self.attn_heads, :dim, :dim]


    def fill_with_neg_inf(self, t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(float("-inf")).type_as(t)