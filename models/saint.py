import torch 
import torch.nn as nn
from torch.nn import Dropout, BCELoss
import pandas as pd
from .modules import transformer_FFN, get_clones, ut_mask, pos_encode, MultiheadAttention
from torch.nn import Embedding, Linear
from IPython import embed 
from .rpe import SinusoidalPositionalEmbeddings 

# device = "cpu" if not torch.cuda.is_available() else "cuda"

class SAINT(nn.Module):
    def __init__(self, device, num_skills, num_questions, seq_len, embedding_size, num_attn_heads, num_blocks, dropout, de_type="none"):
        super().__init__()
        # print(f"num_questions: {num_questions}, num_skills: {num_skills}")
        if num_questions == num_skills and num_questions == 0:
            assert num_questions != 0
        self.num_questions = num_questions
        self.num_skills = num_skills
        self.model_name = "saint"
        self.num_en = num_blocks
        self.num_de = num_blocks

        self.embd_pos = nn.Embedding(seq_len, embedding_dim = embedding_size) 
        # self.embd_pos = Parameter(torch.Tensor(seq_len-1, embedding_size))
        # kaiming_normal_(self.embd_pos)
        self.device = device
        self.encoder = get_clones(Encoder_block(device, embedding_size, num_attn_heads, num_questions, num_skills, seq_len, dropout), self.num_en)
        
        self.de = de_type.split('_')[0]
        self.token_num = int(de_type.split('_')[1])
        if self.de in ["sde", "lsde"]:
            diff_vec = torch.from_numpy(SinusoidalPositionalEmbeddings(2*(self.token_num+1), embedding_size)).to(device)
            self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=True)
            rotary = "none"
        elif self.de in ["rde", "lrde"]:
            rotary = "qkv"
        else: 
            rotary = "none"
            
        self.decoder = get_clones(Decoder_block(device, embedding_size, num_attn_heads, seq_len, dropout, rotary), self.num_de)
        self.embd_res    = nn.Embedding(2+1, embedding_dim = embedding_size)                  #response embedding, include a start token

        self.dropout = Dropout(dropout)
        self.out = nn.Linear(in_features=embedding_size, out_features=1)
        self.loss_fn = BCELoss(reduction="mean")
    
    def forward(self, feed_dict):
        in_ex = feed_dict["questions"]
        in_cat = feed_dict["skills"]
        in_res = feed_dict["responses"][:, :-1]
        diff = feed_dict["sdiff"][:, :-1]

        if self.num_questions > 0:
            in_pos = pos_encode(self.device, in_ex.shape[1])
        else:
            in_pos = pos_encode(self.device, in_cat.shape[1])
        in_pos = self.embd_pos(in_pos)
        # in_pos = self.embd_pos.unsqueeze(0)
        ## pass through each of the encoder blocks in sequence
        first_block = True
        for i in range(self.num_en):
            if i >= 1:
                first_block = False
            in_ex = self.encoder[i](in_ex, in_cat, in_pos, first_block=first_block)
            in_cat = in_ex
            
        ## pass through each decoder blocks in sequence
        start_token = torch.tensor([[2]]).repeat(in_res.shape[0], 1).to(self.device)
        in_res = in_res * (in_res > -1).long()
        in_res = torch.cat((start_token, in_res), dim=-1)
        r = in_res
        diff_token = torch.tensor([[0]]).repeat(in_res.shape[0], 1).to(self.device)
        diff = torch.cat((diff_token, diff), dim=-1)
        first_block = True
        
        if self.token_num < 1000:
            boundaries = torch.linspace(0, 1, steps=self.token_num+1)                
            diff = torch.bucketize(diff, boundaries)
            diff_ox = torch.where(r==0 , diff-(self.token_num+1), diff)  
        else:
            diff = diff * 100
            diff_ox = torch.where(r==0 , diff-(100+1), diff)
            
        ## todo create a positional encoding (two options numeric, sine)
        #combining the embedings
        out = self.embd_res(in_res) + in_pos                         # (b,n,d)
        if self.de in ["sde", "lsde"]:
            diffx = (self.token_num+1) + diff 
            diffo = diff
            diffox = torch.where(r == 0 ,diffo, diffx)
            demb = self.diff_emb(diffox).float()
            out += demb
        
        for i in range(self.num_de):
            if i>0 and self.de == "lsde": out += demb 
            out = self.decoder[i](out, en_out=in_ex, diff=diff_ox)
        
        ## Output layer

        res = self.out(self.dropout(out))
        res = torch.sigmoid(res).squeeze(-1)
        out_dict = {
            "pred": res[:, 1:],
            "true": feed_dict["responses"][:, 1:].float(),
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss , len(pred[mask]), true[mask].sum().item()

class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, device, dim_model, heads_en, total_ex, total_cat, seq_len, dropout, emb_path="", pretrain_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.emb_path = emb_path
        self.total_cat = total_cat
        self.total_ex = total_ex
        self.device = device
        if total_ex > 0:
            if emb_path == "":
                self.embd_ex = nn.Embedding(total_ex, embedding_dim = dim_model)                   # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
            else:
                embs = pd.read_pickle(emb_path)
                self.exercise_embed = Embedding.from_pretrained(embs)
                self.linear = Linear(pretrain_dim, dim_model)
        if total_cat > 0:
            self.emb_cat = nn.Embedding(total_cat, embedding_dim = dim_model)
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding

        self.multi_en = MultiheadAttention(d_model = dim_model, h = heads_en, dropout = dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = Dropout(dropout)

        self.ffn_en = transformer_FFN(dim_model, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, in_ex, in_cat, in_pos, first_block=True):

        ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            embs = []
            if self.total_ex > 0:
                if self.emb_path == "":
                    in_ex = self.embd_ex(in_ex)
                else:
                    in_ex = self.linear(self.exercise_embed(in_ex))
                embs.append(in_ex)
            if self.total_cat > 0:
                in_cat = self.emb_cat(in_cat)
                embs.append(in_cat)
            out = embs[0]
            for i in range(1, len(embs)):
                out += embs[i]
            out = out + in_pos
            # in_pos = self.embd_pos(in_pos)
        else:
            out = in_ex
        
        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        # out = out.permute(1,0,2)                                # (n,b,d)  # print('pre multi', out.shape)
        
        # norm -> attn -> drop -> skip corresponging to transformers' norm_first
        #Multihead attention                            
        _,n,_ = out.shape
        out = self.layer_norm1(out)                           # Layer norm
        skip_out = out 
        out, attn_wt = self.multi_en(out, out, out,
                                mask=~ut_mask(self.device, seq_len=n))  # attention mask upper triangular
        out = self.dropout1(out)
        out = out + skip_out                                    # skip connection

        #feed forward
        # out = out.permute(1,0,2)                                # (b,n,d)
        out = self.layer_norm2(out)                           # Layer norm 
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout2(out)
        out = out + skip_out                                    # skip connection

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, device, dim_model, heads_de, seq_len, dropout, rotary="none"):
        super().__init__()
        self.seq_len    = seq_len
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding
        self.rotary = rotary
        if self.rotary in ["qkv", "none"]:
            self.multi_de1 = MultiheadAttention(dim_model, heads_de, dropout=dropout, rotary=self.rotary)
        else:
            self.multi_de1  = MultiheadAttention(dim_model, heads_de, dropout=dropout)  # M1 multihead for interaction embedding as q k v
        self.multi_de2  = MultiheadAttention(dim_model, heads_de, dropout=dropout)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en     = transformer_FFN(dim_model, dropout)                                         # feed forward layer

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.device = device 


    def forward(self, out, en_out, diff=None):
        _,n,_ = out.shape
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out, diff=diff, mask=~ut_mask(self.device, seq_len=n))
        out = self.dropout1(out)
        out = skip_out + out                                        # skip connection

        #Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                    mask=~ut_mask(self.device, seq_len=n))  # attention mask upper triangular
        out = self.dropout2(out)
        out = out + skip_out

        #feed forward
        # out = out.permute(1,0,2)                                    # (b,n,d)
        out = self.layer_norm3(out)                               # Layer norm 
        skip_out = out
        out = self.ffn_en(out)                                    
        out = self.dropout3(out)
        out = out + skip_out                                        # skip connection

        return out