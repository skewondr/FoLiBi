# https://github.com/arghosh/AKT/blob/master/akt.py
import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.nn import Module, Embedding, Linear, ReLU, Dropout, ModuleList, Sequential
from .modules import AKTTransformerLayer
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
from .rpe import SinusoidalPositionalEmbeddings 

class AKT(Module):
    def __init__(self, device,  num_skills, num_questions, seq_len, bincounts,
                 embedding_size, num_blocks, kq_same, choose_enc="g",
                 de_type="none_0", model_type="akt", num_attn_heads=8,
                 final_fc_dim=512, d_ff=2048, reg_l=1e-5, dropout=0.2, separate_qr=False):
        super(AKT, self).__init__()

        """
        params:
            num_skills: # of skills
            num_questions: # of questions
            embedding_size: embedding dim
            num_blocks: # of attn blocks
            seq_len: max length of sequenc
            kq_same: key랑 query랑 같은지
            num_attn_heads: number of heads if multi-headed attention
            final_fc_dim: dimension of final fully connected net before prediction
            d_ff: dimension for fully connected net inside the basic block
            
        """
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks
        self.seq_len = seq_len
        self.bincounts = bincounts
        self.kq_same = kq_same
        print("kq_same", kq_same)
        self.model_type = model_type
        self.num_attn_heads = num_attn_heads
        self.final_fc_dim = final_fc_dim
        self.d_ff = d_ff
        self.reg_l = reg_l
        self.dropout = dropout
        self.separate_qr = separate_qr

        if self.num_questions > 0:
            self.difficult_param = Embedding(
                self.num_questions, 1, padding_idx=0
            )  # /mu_{q_t} parameter
            self.q_embed_diff = Embedding(
                self.num_skills, self.embedding_size, padding_idx=0
            )  # d_{c_t}
            self.qr_embed_diff = Embedding(
                2 * self.num_skills, self.embedding_size, padding_idx=0
            )  # f_{(c_t, r_t)} or h_{r_t}
        self.q_embed = Embedding(
            self.num_skills, self.embedding_size, padding_idx=0
        )  # c_{c_t}
        if self.separate_qr:
            self.qr_embed = Embedding(
                2 * self.num_skills, self.embedding_size, padding_idx=0
            )  # e_{(c_t, r_t)}
        else:
            self.r_embed = Embedding(
                2 + 1, self.embedding_size, padding_idx=0
            )  # e_{(c_t, r_t)} 

        self.de = de_type.split('_')[0]
        self.token_num = int(de_type.split('_')[1])
        self.choose_enc = choose_enc
        
        if self.de.startswith(("sde", "alibi-sde", "rotary-sde")):
            diff_vec = torch.from_numpy(SinusoidalPositionalEmbeddings(self.token_num+1, embedding_size)).to(device)
            self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=True)
        elif self.de.startswith("random"):
            self.diff_emb = Embedding(self.token_num+1, embedding_size)
            
        self.model = Architecture(
            n_question=self.num_skills,
            n_blocks=self.num_blocks,
            n_heads=self.num_attn_heads,
            dropout=self.dropout,
            d_model=self.embedding_size,
            d_feature=self.embedding_size / self.num_attn_heads,
            d_ff=self.d_ff,
            kq_same=self.kq_same,
            model_type=self.model_type,
            seq_len=seq_len,
            choose_enc=choose_enc,
            de_type=de_type,
            bincounts=self.bincounts,
        )

        self.out = Sequential(
            Linear(2 * self.embedding_size, self.final_fc_dim),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim, self.final_fc_dim // 2),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim // 2, 1),
        )
        self.reset()
        self.loss_fn = nn.BCELoss(reduction="mean")
        self.position_emb = Embedding(seq_len + 1, embedding_size, padding_idx=0)

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_questions + 1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.0)

    def forward(self, feed_dict):
        q = feed_dict["skills"]
        r = feed_dict["responses"]
        attention_mask = feed_dict["attention_mask"]
        masked_r = r * (r > -1).long()
        pid_data = feed_dict["questions"]
        diff = feed_dict["sdiff"]
        diff = (diff*(r>-1).int()).long()    
        pos = feed_dict["position"]
        
        f_embed=None
        q_embed_data = self.q_embed(q)  # c_{c_t}: [batch_size, seq_len, embedding_size]
        if self.separate_qr:
            qr = q + self.num_skills * masked_r
            qr_embed_data = self.qr_embed(qr)  # f_{(c_t, r_t)}: [batch_size, seq_len, d_model]
        else:
            qr = masked_r
            qr_embed_data = q_embed_data + self.r_embed(qr)
            
        if self.num_questions > 0:
            q_embed_diff_data = self.q_embed_diff(q)  # d_{c_t}: variation vector
            pid_embed_data = self.difficult_param(pid_data)  # \mu_{q_t}
            qr_embed_diff_data = self.qr_embed_diff(qr)  # f_{(c_t, r_t)} or h_{r_t}
            if self.de.startswith("none"):
                q_embed_data = (
                    q_embed_data + pid_embed_data * q_embed_diff_data
                )  # x_t = c_{c_t} + \mu_{q_t} + d_{c_t}
                qr_embed_data = qr_embed_data + pid_embed_data * (
                    qr_embed_diff_data + q_embed_diff_data
                )
            elif self.de.startswith(("sde", "random", "alibi-sde", "rotary-sde")):
                if "q" in self.choose_enc:
                    q_embed_data += self.diff_emb(diff).float()
                if "i" in self.choose_enc:
                    qr_embed_data += self.diff_emb(diff).float()
                if "f" in self.choose_enc:
                    f_embed = self.diff_emb(diff).float()
            # elif self.de.startswith("alibi") and not "1" in self.de:
            #     posemb = self.position_emb(pos)
            #     if "q" in self.choose_enc:
            #         q_embed_data += posemb
            #     if "i" in self.choose_enc:
            #         qr_embed_data += posemb

            c_reg_loss = torch.mean(pid_embed_data ** 2.0) * self.reg_l
        else:
            c_reg_loss = 0

        pooled_ques_score = (self.q_embed(q) * attention_mask.unsqueeze(-1)).sum(
            1
        ) / attention_mask.sum(-1).unsqueeze(-1)
        pooled_inter_score = (qr_embed_data * attention_mask.unsqueeze(-1)).sum(
            1
        ) / attention_mask.sum(-1).unsqueeze(-1)

        # [batch_size, seq_len, d_model]
        # pass to the decoder
        # output shape [batch_size, seq_len, d_model or d_model//2]
        # d_output is h_t
        d_output, attn = self.model(q_embed_data, qr_embed_data, diff, r, f_embed)  # 211x512

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)  # concat([h_t, x_t])
        output = torch.sigmoid(self.out(concat_q)).squeeze()

        if self.training:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "c_reg_loss": c_reg_loss,
            }
        else:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "c_reg_loss": c_reg_loss,
                "q_embed": pooled_ques_score,
                "qr_embed": pooled_inter_score,
            }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        c_reg_loss = out_dict["c_reg_loss"]
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss + c_reg_loss, len(pred[mask]), true[mask].sum().item()

    def alignment_and_uniformity(self, out_dict):
        return (
            out_dict["uniformity"],
            out_dict["uniformity"],
            out_dict["uniformity"],
            out_dict["uniformity"],
        )


class Architecture(Module):
    def __init__(
        self,
        n_question,
        n_blocks,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        model_type,
        seq_len,
        choose_enc="g",
        de_type="none_0",
        bincounts=None,
    ):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type
        print("model_type", model_type)
        self.de_type = de_type
        self.choose_enc = choose_enc
        if model_type == "akt":
            self.blocks_1 = ModuleList(
                [
                    AKTTransformerLayer(
                        d_model=d_model,
                        d_feature=d_model // n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        n_heads=n_heads,
                        kq_same=kq_same,
                        de_type=de_type,
                        bincounts=bincounts,
                        seq_len=seq_len,
                    )
                    for _ in range(n_blocks)
                ]
            )
            self.blocks_2 = ModuleList(
                [
                    AKTTransformerLayer(
                        d_model=d_model,
                        d_feature=d_model // n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        n_heads=n_heads,
                        kq_same=kq_same,
                        de_type=de_type,
                        bincounts=bincounts,
                        seq_len=seq_len,
                    )
                    for _ in range(n_blocks * 2)
                ]
            )

    def forward(self, q_embed_data, qa_embed_data, diff=None, r=None, f_embed=None):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        q_enc = None
        i_enc = None 
        f_enc = None 
        if self.de_type.startswith(("alibi", "rotary", "basic", "relative")):
            if "q" in self.choose_enc:
                q_enc = diff
            if "i" in self.choose_enc:
                i_enc = diff 
            if "f" in self.choose_enc:
                f_enc = diff 

        # encoder
        for block in self.blocks_1:  # knowledge encoder: encode (question, response)'s
            # knowledge encoder
            # y^{\hat}_{t-1} = f_{enc_2} (y_1, ..., y_{t-1})
            # y can see both current and past information
            """
            mask: 0 means that it can peek only past values.
            1 means that block can peek only current and past values
            """
            y, _ = block(mask=1, query=y, key=y, values=y, diff=i_enc, response=r)
        flag_first = True
        for idx, block in enumerate(self.blocks_2):
            if flag_first:  # peek current question
                # question encoder
                # x^{\hat}_{t} = f_{enc_1} (x_1, ..., x_t)
                # x can see both current and past information
                if idx == 1 and f_embed is not None:
                    x = x+f_embed
                x, _ = block(mask=1, query=x, key=x, values=x, diff=q_enc, response=r, apply_pos=False)
                flag_first = False
            else:  # dont peek current response
                # knoweldge retriever
                # h_t = f_{kr} (x^{\hat}_1, ..., x^{\hat}_t, y^{\hat}_1, ..., y^{\hat}_{t-1})
                # h can see past only
                if idx == 1 and f_embed is not None:
                    x = x+f_embed
                    y = y+f_embed
                x, attn = block(mask=0, query=x, key=x, values=y, diff=f_enc, response=r, apply_pos=True)
                flag_first = True
        return x, attn