import torch

from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout, BCELoss
from .modules import transformer_FFN, pos_encode, ut_mask, get_clones, MultiheadAttention

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
        
        if self.de.startswith("sde"):
            diff_vec = torch.from_numpy(SinusoidalPositionalEmbeddings(2*(self.token_num+1), embedding_size)).to(device)
            self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=True)
            
        self.blocks = get_clones(Blocks(device, embedding_size, num_attn_heads, dropout, de_type, bincounts=bincounts), self.num_blocks)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.embedding_size, 1)
        
    def base_emb(self, q, r, qry, pos, diff):
        masked_responses = r * (r > -1).long()
        x = q + self.num_skills * masked_responses
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
        posemb = self.position_emb(pos)
        if not self.de.startswith("alibi"):
            xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, feed_dict):
        q = feed_dict["skills"][:, :-1]
        qry = feed_dict["skills"][:, 1:]
        r = feed_dict["responses"][:, :-1]
        pos = feed_dict["position"][:, :-1]
        diff = feed_dict["sdiff"]

        if self.token_num < 1000:
            boundaries = torch.linspace(0, 1, steps=self.token_num+1)                
            diff = torch.bucketize(diff, boundaries)
        else: 
            diff = None 
            
        qshftemb, xemb = self.base_emb(q, r, qry, pos, diff)
        enc = None
        if self.de.startswith("sde"):
            qshftemb += self.diff_emb(diff[:, 1:]).float()
            xemb += self.diff_emb(diff[:, :-1]).float()
        else:
            enc = diff
            
        for i in range(self.num_blocks):
            xemb = self.blocks[i](qshftemb, xemb, xemb, enc)

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
    def __init__(self, device, embedding_size, num_attn_heads, dropout, de_type="none_0", bincounts=None) -> None:
        super().__init__()
        self.device = device
        self.attn = MultiheadAttention(embedding_size, num_attn_heads, de_type=de_type, dropout=dropout, bincounts=bincounts)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(embedding_size)

        self.FFN = transformer_FFN(embedding_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(embedding_size)
        
    def forward(self, q=None, k=None, v=None, diff=None):
        causal_mask = ~ut_mask(self.device, seq_len = k.shape[1])
        attn_emb, _ = self.attn(q, k, v, diff=diff, mask=causal_mask)
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb