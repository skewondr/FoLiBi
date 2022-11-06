import torch

from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout, BCELoss
from .modules import transformer_FFN, pos_encode, ut_mask, get_clones

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


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
        num_blocks=2, 
        emb_path="", 
        pretrain_dim=768
        ):
        super().__init__()
        self.device = device 

        self.num_questions = num_questions
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.loss_fn = BCELoss(reduction="mean")

        # num_questions, seq_len, embedding_size, num_attn_heads, dropout, emb_path="")
        self.interaction_emb = Embedding(num_questions * 2, embedding_size, padding_idx=0)
        self.exercise_emb = Embedding(num_questions, embedding_size, padding_idx=0)
        # self.P = Parameter(torch.Tensor(self.seq_len, self.embedding_size))
        self.position_emb = Embedding(seq_len + 1, embedding_size, padding_idx=0)

        self.blocks = get_clones(Blocks(device, embedding_size, num_attn_heads, dropout), self.num_blocks)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.embedding_size, 1)

    def base_emb(self, q, r, qry, pos):
        masked_responses = r * (r > -1).long()
        x = q + self.num_questions * masked_responses
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
        posemb = self.position_emb(pos)
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, feed_dict):
        q = feed_dict["skills"][:-1]
        r = feed_dict["responses"][:-1]
        qry = feed_dict["skills"][1:]
        pos = feed_dict["position"][:-1]
        qshftemb, xemb = self.base_emb(q, r, qry, pos)
        for i in range(self.num_blocks):
            xemb = self.blocks[i](qshftemb, xemb, xemb)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)

        out_dict = {
            "pred": p,
            "true": r.float(),
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        # from IPython import embed ; embed()
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss , len(pred[mask]), true[mask].sum().item()

class Blocks(Module):
    def __init__(self, device, embedding_size, num_attn_heads, dropout) -> None:
        super().__init__()
        self.device = device
        self.attn = MultiheadAttention(embedding_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(embedding_size)

        self.FFN = transformer_FFN(embedding_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(embedding_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm 
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(self.device, seq_len = k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb