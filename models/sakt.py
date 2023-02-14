import torch

from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout, BCELoss
from .modules import transformer_FFN, pos_encode, ut_mask, get_clones, MultiheadAttention, MultiHeadAttentionWithContextDistance

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from IPython import embed
from .rpe import SinusoidalPositionalEmbeddings 

class SAKT(Module):
    def __init__(self, device, num_skills,num_questions, seq_len, bincounts,
                 embedding_size, num_attn_heads, dropout, de_type="none_0",
                 choose_enc="g", num_blocks=2, emb_path="", pretrain_dim=768):
        super().__init__()
        self.device = device 

        self.num_questions = num_questions
        self.num_skills = num_skills
        self.seq_len = seq_len
        self.bincounts = bincounts
        self.embedding_size = embedding_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.loss_fn = BCELoss(reduction="mean")

        # num_questions, seq_len, embedding_size, num_attn_heads, dropout, emb_path="")
        self.interaction_emb = Embedding(num_skills * 2, embedding_size, padding_idx=0)
        self.exercise_emb = Embedding(num_skills, embedding_size, padding_idx=0)
        # self.P = Parameter(torch.Tensor(self.seq_len, self.embedding_size))
        self.position_emb = Embedding(seq_len + 1, embedding_size, padding_idx=0)

        self.de = de_type.split('_')[0]
        self.token_num = int(de_type.split('_')[1])
        self.choose_enc = choose_enc
        
        if self.de.startswith(("sde", "alibi-sde", "rotary-sde")):
            diff_vec = torch.from_numpy(SinusoidalPositionalEmbeddings(self.token_num+1, embedding_size)).to(device)
            self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=True)
        elif self.de.startswith("random"):
            self.diff_emb = Embedding(self.token_num+1, embedding_size)
            
        self.blocks = get_clones(Blocks(device, embedding_size, num_attn_heads, dropout, seq_len, de_type, bincounts=bincounts), self.num_blocks)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.embedding_size, 1)
        
    def base_emb(self, q, r, qry, pos, diff):
        masked_responses = r * (r > -1).long()
        x = q + self.num_skills * masked_responses
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
        if self.de.startswith(("sde", "random", "alibi-sde", "rotary-sde")):
            qshftemb += self.diff_emb(diff[:, 1:]).float()
            xemb += self.diff_emb(diff[:, :-1]).float()
        if self.de.startswith(("none", "sde", "random")):
            #alibi를 제외하면, position 정보가 들어가야 함. 
            posemb = self.position_emb(pos)
            xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, feed_dict):
        q = feed_dict["skills"][:, :-1]
        qry = feed_dict["skills"][:, 1:]
        r = feed_dict["responses"][:, :-1]
        pos = feed_dict["position"][:, :-1]
        diff = feed_dict["sdiff"]
        diff = (diff*(feed_dict["responses"]>-1).int()).long()    

        qshftemb, xemb = self.base_emb(q, r, qry, pos, diff)
        enc = None
        if self.de.startswith(("alibi", "rotary")):
            enc = diff
            
        for i in range(self.num_blocks):
            xemb = self.blocks[i](qshftemb, xemb, xemb, enc, r)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        out_dict = {
            "pred": p,
            "true": feed_dict["responses"][:, 1:].float(),
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss , len(pred[mask]), true[mask].sum().item()

class Blocks(Module):
    def __init__(self, device, embedding_size, num_attn_heads, dropout, seq_len, de_type="none_0", bincounts=None) -> None:
        super().__init__()
        self.device = device
        if de_type.startswith("monotonic"):
            kq_same = False
            self.attn = MultiHeadAttentionWithContextDistance(
            embedding_size, embedding_size//num_attn_heads, num_attn_heads, dropout, kq_same=kq_same, seq_len=seq_len, de_type=de_type, bincounts=bincounts)
        else:
            self.attn = MultiheadAttention(embedding_size, num_attn_heads, dropout=dropout, seq_len=seq_len, de_type=de_type, bincounts=bincounts)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(embedding_size)

        self.FFN = transformer_FFN(embedding_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(embedding_size)
        
    def forward(self, q=None, k=None, v=None, diff=None, r=None):
        causal_mask = ~ut_mask(self.device, seq_len = k.shape[1])
        attn_emb, _ = self.attn(q, k, v, diff=diff, response=r, mask=causal_mask)
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb