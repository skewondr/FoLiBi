import math
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Embedding,
    Linear,
    ReLU,
    Dropout,
    ModuleList,
    Softplus,
    Sequential,
    Sigmoid,
    BCEWithLogitsLoss,
)
import torch.nn.functional as F
from torch.nn.modules.activation import GELU
from .modules import CL4KTTransformerLayer

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
from .rpe import SinusoidalPositionalEmbeddings 
from IPython import embed 

class CL4KT(Module):
    def __init__(self, device, num_skills, num_questions, seq_len, bincounts, **kwargs):
        super(CL4KT, self).__init__()
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.seq_len = seq_len
        self.bincounts = bincounts
        self.args = kwargs
        self.hidden_size = self.args["hidden_size"]
        self.num_blocks = self.args["num_blocks"]
        self.num_attn_heads = self.args["num_attn_heads"]
        self.kq_same = self.args["kq_same"]
        self.final_fc_dim = self.args["final_fc_dim"]
        self.d_ff = self.args["d_ff"]
        self.dropout = self.args["dropout"]
        self.reg_cl = self.args["reg_cl"]
        self.negative_prob = self.args["negative_prob"]
        self.hard_negative_weight = self.args["hard_negative_weight"]
        self.only_rp = self.args["only_rp"]
        self.choose_cl = self.args["choose_cl"]
        self.de = self.args["de_type"].split('_')[0]
        self.token_num = int(self.args["de_type"].split('_')[1])
        self.de_type = self.args["de_type"]
        self.choose_enc = self.args["choose_enc"]

        if self.de.startswith(("sde", "alibi-sde", "rotary-sde")):
            diff_vec = torch.from_numpy(SinusoidalPositionalEmbeddings(self.token_num+1, self.hidden_size)).to(device)
            self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=True)
        elif self.de.startswith("random"):
            self.diff_emb = Embedding(self.token_num+1, self.hidden_size)
            
        self.position_emb = Embedding(seq_len + 1, self.hidden_size, padding_idx=0)
        self.question_embed = Embedding(
            self.num_skills + 2, self.hidden_size, padding_idx=0
        )
        self.interaction_embed = Embedding(
            2 * (self.num_skills + 2), self.hidden_size, padding_idx=0
        )
        self.sim = Similarity(temp=self.args["temp"])

        self.question_encoder = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                    de_type=self.de_type,
                    bincounts=self.bincounts,
                    seq_len=seq_len,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.interaction_encoder = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                    de_type=self.de_type,
                    bincounts=self.bincounts,
                    seq_len=seq_len,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.knoweldge_retriever = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                    de_type=self.de_type,
                    bincounts=self.bincounts,
                    seq_len=seq_len,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out = Sequential(
            Linear(2 * self.hidden_size, self.final_fc_dim),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim, self.final_fc_dim // 2),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim // 2, 1),
        )

        self.cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.loss_fn = nn.BCELoss(reduction="mean")

    def forward(self, batch):
        if self.training:
            q_i, q_j, q = batch["skills"]  # augmented q_i, augmented q_j and original q
            r_i, r_j, r, neg_r = batch[
                "responses"
            ]  # augmented r_i, augmented r_j and original r
            attention_mask_i, attention_mask_j, attention_mask = batch["attention_mask"]
            diff_i, diff_j, diff = batch["sdiff"]
            diff_i, diff_j, diff = (diff_i*(r_i>-1).int()).long(), (diff_j*(r_j>-1).int()).long(), (diff*(r>-1).int()).long()    
            pos = batch["position"]
            
            if not self.only_rp:
                ques_i_embed = self.question_embed(q_i)
                ques_j_embed = self.question_embed(q_j)
                inter_i_embed = self.get_interaction_embed(q_i, r_i)
                inter_j_embed = self.get_interaction_embed(q_j, r_j)
                inter_k_embed = self.get_interaction_embed(q, neg_r)
                if self.de.startswith(("sde", "random", "alibi-sde", "rotary-sde")):
                    if "q" in self.choose_enc:
                        ques_i_embed += self.diff_emb(diff_i).float()
                        ques_j_embed += self.diff_emb(diff_j).float()
                    if "i" in self.choose_enc:
                        inter_i_embed += self.diff_emb(diff_i).float()
                        inter_j_embed += self.diff_emb(diff_j).float()
                        inter_k_embed += self.diff_emb(diff).float()
                # elif self.de.startswith("alibi") and not "1" in self.de:
                #     posemb = self.position_emb(pos)
                #     if "q" in self.choose_enc:
                #         ques_i_embed += posemb
                #         ques_j_embed += posemb
                #     if "i" in self.choose_enc:
                #         inter_i_embed += posemb
                #         inter_j_embed += posemb
                #         inter_k_embed += posemb
                    
                q_i_enc, q_j_enc = None, None
                i_i_enc, i_j_enc, i_k_enc = None, None, None   
                if self.de.startswith(("alibi", "rotary", "basic", "relative")):
                    if "q" in self.choose_enc:
                        q_i_enc = diff_i
                        q_j_enc = diff_j
                    if "i" in self.choose_enc:
                        i_i_enc = diff_i 
                        i_j_enc = diff_j 
                        i_k_enc = diff 

                # mask=2 means bidirectional attention of BERT
                ques_i_score, ques_j_score = ques_i_embed, ques_j_embed
                inter_i_score, inter_j_score = inter_i_embed, inter_j_embed

                # BERT
                if self.choose_cl in ["q_cl", "both"]:
                    for block in self.question_encoder:
                        ques_i_score, _ = block(
                            mask=2,
                            query=ques_i_score,
                            key=ques_i_score,
                            values=ques_i_score,
                            apply_pos=False,
                            diff=q_i_enc,
                            response=r_i,
                        )
                        ques_j_score, _ = block(
                            mask=2,
                            query=ques_j_score,
                            key=ques_j_score,
                            values=ques_j_score,
                            apply_pos=False,
                            diff=q_j_enc,
                            response=r_j,
                        )
                if self.choose_cl in ["s_cl", "both"]:
                    for block in self.interaction_encoder:
                        inter_i_score, _ = block(
                            mask=2,
                            query=inter_i_score,
                            key=inter_i_score,
                            values=inter_i_score,
                            apply_pos=False,
                            diff=i_i_enc,
                            response=r_i,
                        )
                        inter_j_score, _ = block(
                            mask=2,
                            query=inter_j_score,
                            key=inter_j_score,
                            values=inter_j_score,
                            apply_pos=False,
                            diff=i_j_enc,
                            response=r_j,
                        )
                        if self.negative_prob > 0:
                            inter_k_score, _ = block(
                                mask=2,
                                query=inter_k_embed,
                                key=inter_k_embed,
                                values=inter_k_embed,
                                apply_pos=False,
                                diff=i_k_enc,
                                response=neg_r,
                            )
                if self.choose_cl in ["q_cl", "both"]:
                    pooled_ques_i_score = (ques_i_score * attention_mask_i.unsqueeze(-1)).sum(
                        1
                    ) / attention_mask_i.sum(-1).unsqueeze(-1)
                    pooled_ques_j_score = (ques_j_score * attention_mask_j.unsqueeze(-1)).sum(
                        1
                    ) / attention_mask_j.sum(-1).unsqueeze(-1)

                    ques_cos_sim = self.sim(
                        pooled_ques_i_score.unsqueeze(1), pooled_ques_j_score.unsqueeze(0)
                    )
                    # Hard negative should be added

                    ques_labels = torch.arange(ques_cos_sim.size(0)).long().to(q_i.device)
                    question_cl_loss = self.cl_loss_fn(ques_cos_sim, ques_labels)
                    # question_cl_loss = torch.mean(question_cl_loss)
                else: 
                    question_cl_loss = 0
                if self.choose_cl in ["s_cl", "both"]:
                    pooled_inter_i_score = (inter_i_score * attention_mask_i.unsqueeze(-1)).sum(
                        1
                    ) / attention_mask_i.sum(-1).unsqueeze(-1)
                    pooled_inter_j_score = (inter_j_score * attention_mask_j.unsqueeze(-1)).sum(
                        1
                    ) / attention_mask_j.sum(-1).unsqueeze(-1)

                    inter_cos_sim = self.sim(
                        pooled_inter_i_score.unsqueeze(1), pooled_inter_j_score.unsqueeze(0)
                    )

                    if self.negative_prob > 0:
                        pooled_inter_k_score = (
                            inter_k_score * attention_mask.unsqueeze(-1)
                        ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                        neg_inter_cos_sim = self.sim(
                            pooled_inter_i_score.unsqueeze(1), pooled_inter_k_score.unsqueeze(0)
                        )
                        inter_cos_sim = torch.cat([inter_cos_sim, neg_inter_cos_sim], 1)

                    inter_labels = torch.arange(inter_cos_sim.size(0)).long().to(q_i.device)

                    if self.negative_prob > 0:
                        weights = torch.tensor(
                            [
                                [0.0] * (inter_cos_sim.size(-1) - neg_inter_cos_sim.size(-1))
                                + [0.0] * i
                                + [self.hard_negative_weight]
                                + [0.0] * (neg_inter_cos_sim.size(-1) - i - 1)
                                for i in range(neg_inter_cos_sim.size(-1))
                            ]
                        ).to(q_i.device)
                        inter_cos_sim = inter_cos_sim + weights

                    interaction_cl_loss = self.cl_loss_fn(inter_cos_sim, inter_labels)
                else: 
                    interaction_cl_loss = 0 
            else: 
                question_cl_loss, interaction_cl_loss = 0, 0
        else:
            q = batch["skills"]  # augmented q_i, augmented q_j and original q
            r = batch["responses"]  # augmented r_i, augmented r_j and original r

            attention_mask = batch["attention_mask"]
            diff = batch["sdiff"]
            diff = (diff*(r>-1).int()).long()    
            pos = batch["position"]
            question_cl_loss, interaction_cl_loss = 0, 0
            
        q_embed = self.question_embed(q)
        i_embed = self.get_interaction_embed(q, r)
        f_embed = None 

        if self.de.startswith(("sde", "random", "alibi-sde", "rotary-sde")):
            if "q" in self.choose_enc:
                q_embed += self.diff_emb(diff).float()
            if "i" in self.choose_enc:
                i_embed += self.diff_emb(diff).float()
            if "f" in self.choose_enc:
                f_embed = self.diff_emb(diff).float()
        # elif self.de.startswith("alibi") and not "1" in self.de:
        #     posemb = self.position_emb(pos)
        #     if "q" in self.choose_enc:
        #         q_embed += posemb
        #     if "i" in self.choose_enc:
        #         i_embed += posemb
            
        q_enc = None
        i_enc = None 
        f_enc = None 
        if self.de.startswith(("alibi", "rotary", "basic", "relative")):
            if "q" in self.choose_enc:
                q_enc = diff
            if "i" in self.choose_enc:
                i_enc = diff 
            if "f" in self.choose_enc:
                f_enc = diff 
                
        x, y = q_embed, i_embed
        for block in self.question_encoder:
            x, _ = block(mask=1, query=x, key=x, values=x, diff=q_enc, response=r, apply_pos=True)

        for block in self.interaction_encoder:
            y, _ = block(mask=1, query=y, key=y, values=y, diff=i_enc, response=r, apply_pos=True)

        for idx, block in enumerate(self.knoweldge_retriever):
            if f_embed is not None:
                x = x+f_embed
                y = y+f_embed
            x, attn = block(mask=0, query=x, key=x, values=y, diff=f_enc, response=r, apply_pos=True)

        retrieved_knowledge = torch.cat([x, q_embed], dim=-1)

        output = torch.sigmoid(self.out(retrieved_knowledge)).squeeze()
        total_cl_loss = question_cl_loss + interaction_cl_loss

        if self.training:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "cl_loss": total_cl_loss,
                "attn": attn,
            }
        else:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "attn": attn,
                "x": x,
            }

        return out_dict

    def alignment_and_uniformity(self, out_dict):
        return (
            out_dict["question_alignment"],
            out_dict["interaction_alignment"],
            out_dict["question_uniformity"],
            out_dict["interaction_uniformity"],
        )

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        if not self.only_rp:
            cl_loss = torch.mean(out_dict["cl_loss"])  # torch.mean() for multi-gpu FIXME
        else:
            cl_loss = 0
        mask = true > -1

        loss = self.loss_fn(pred[mask], true[mask]) + self.reg_cl * cl_loss

        return loss, len(pred[mask]), true[mask].sum().item()

    def get_interaction_embed(self, skills, responses):
        masked_responses = responses * (responses > -1).long()
        interactions = skills + self.num_skills * masked_responses
        output = self.interaction_embed(interactions)
        return output


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


# https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], ("unrecognized pooling type %s" % self.pooler_type)

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


# ref: https://github.com/SsnL/align_uniform
def align_loss(x, y, alpha=2):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    x = F.normalize(x, dim=1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()