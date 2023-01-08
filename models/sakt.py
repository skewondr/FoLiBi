import torch

from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout, BCELoss
from .modules import transformer_FFN, pos_encode, ut_mask, get_clones, MultiheadAttention
from .rpe import SinusoidalPositionalEmbeddings 

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from IPython import embed

class SAKT(Module):
    def __init__(
        self, 
        device, 
        num_skills,
        num_questions, 
        seq_len, 
        embedding_size, 
        num_attn_heads, 
        dropout, 
        de_type="none",
        num_blocks=2, 
        emb_path="", 
        pretrain_dim=768
        ):
        super().__init__()
        self.device = device 

        self.num_questions = num_questions
        self.num_skills = num_skills
        self.seq_len = seq_len
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
        if self.de in ["sde", "lsde"]:
            diff_vec = torch.from_numpy(SinusoidalPositionalEmbeddings(2*self.token_num, embedding_size)).to(device)
            self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=True)
            rotary = "none"
        elif self.de == "rde":
            rotary = "qkv"
        else: 
            rotary = "none"

        self.blocks = get_clones(Blocks(device, embedding_size, num_attn_heads, dropout, rotary), self.num_blocks)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.embedding_size, 1)

    def base_emb(self, q, r, qry, pos, diff):
        masked_responses = r * (r > -1).long()
        x = q + self.num_skills * masked_responses
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
        posemb = self.position_emb(pos)
        xemb = xemb + posemb
        if self.de in ["sde", "lsde"]:
            diffx = self.token_num + diff * (r > -1).long()
            diffo = diff * (r > -1).int()
            diffox = torch.where(r == 0 ,diffo, diffx)
            demb = self.diff_emb(diffox).float()
            xemb += demb
            return qshftemb, xemb, demb
        else:
            return qshftemb, xemb, None

    def forward(self, feed_dict):
        q = feed_dict["skills"][:, :-1]
        r = feed_dict["responses"][:, :-1]
        qry = feed_dict["skills"][:, 1:]
        pos = feed_dict["position"][:, :-1]
        diff = feed_dict["sdiff"][:, :-1]
        
        if self.token_num < 1000 :  
            diff = torch.ceil(diff * (self.token_num-1)).long()
            diff_ox = torch.where(r == 0 , (diff - self.token_num) * (r > -1).int(), diff * (r > -1).int())
        else:
            diff = diff * 100
            diff_ox = torch.where(r == 0 , (diff - 100) * (r > -1).int(), diff * (r > -1).int())
            
        qshftemb, xemb, demb = self.base_emb(q, r, qry, pos, diff)
        
        for i in range(self.num_blocks): #sakt's num_blocks = 1
            if i>0 and self.de == "lsde": xemb += demb
            xemb = self.blocks[i](qshftemb, xemb, xemb, diff_ox)
            
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
    def __init__(self, device, embedding_size, num_attn_heads, dropout, rotary="none") -> None:
        super().__init__()
        self.device = device
        self.rotary  = rotary
        if self.rotary in ["qkv", "none"]:
            self.attn = MultiheadAttention(embedding_size, num_attn_heads, dropout=dropout, rotary=rotary)
        else:
            self.attn = MultiheadAttention(embedding_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(embedding_size)

        self.FFN = transformer_FFN(embedding_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(embedding_size)

    def forward(self, q=None, k=None, v=None, diff=None):
        causal_mask = ut_mask(self.device, seq_len = k.shape[1])
        attn_emb, _ = self.attn(q, k, v, diff=diff, mask=~causal_mask)
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb