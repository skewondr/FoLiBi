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
    def __init__(self, device, num_skills, num_questions, seq_len, **kwargs):
        super(CL4KT, self).__init__()
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.seq_len = seq_len
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

        self.question_embed = Embedding(
            self.num_skills + 2, self.hidden_size, padding_idx=0
        )
        self.interaction_embed = Embedding(
            2 * (self.num_skills + 2), self.hidden_size, padding_idx=0
        )
        self.sim = Similarity(temp=self.args["temp"])

        if self.de in ["sde", "lsde"]:
            diff_vec = torch.from_numpy(SinusoidalPositionalEmbeddings(2*self.token_num, self.hidden_size)).to(device)
            self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=True)
            rotary = "none"
        elif self.de == "rde":
            rotary = "qkv"
        else: 
            rotary = "none"

        self.question_encoder = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
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
                    rotary=rotary,
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
            diff_i, diff_j, diff = batch["sdiff"]
            if self.token_num < 1000 :  
                diff_i = torch.ceil(diff_i * (self.token_num-1)).long()
                diff_j = torch.ceil(diff_j * (self.token_num-1)).long()
                diff = torch.ceil(diff * (self.token_num-1)).long()
                s_diff_ox = torch.where(r == 1 ,  (diff - self.token_num) * (r > -1).int(), diff * (r > -1).int())
                si_diff_ox = torch.where(r_i == 1 , (diff_i - self.token_num) * (r_i > -1).int(), diff_i * (r_i > -1).int())
                sj_diff_ox = torch.where(r_j == 1 , (diff_j - self.token_num) * (r_j > -1).int(), diff_j * (r_j > -1).int())
                neg_diff = torch.where(neg_r == 1 , (diff - self.token_num) * (neg_r > -1).int(), diff * (neg_r > -1).int())
            else:
                diff_i, diff_j, diff = diff_i*100, diff_j*100, diff*100
                s_diff_ox = torch.where(r == 1 ,  (diff - 100) * (r > -1).int(), diff * (r > -1).int())
                si_diff_ox = torch.where(r_i == 1 , (diff_i - 100) * (r_i > -1).int(), diff_i * (r_i > -1).int())
                sj_diff_ox = torch.where(r_j == 1 , (diff_j - 100) * (r_j > -1).int(), diff_j * (r_j > -1).int())
                neg_diff = torch.where(neg_r == 1 , (diff - 100) * (neg_r > -1).int(), diff * (neg_r > -1).int())
            
            attention_mask_i, attention_mask_j, attention_mask = batch["attention_mask"]

            if not self.only_rp:
                ques_i_embed = self.question_embed(q_i)
                ques_j_embed = self.question_embed(q_j)
                inter_i_embed, _ = self.get_interaction_embed(q_i, r_i, diff_i)
                inter_j_embed, _ = self.get_interaction_embed(q_j, r_j, diff_j)
                if self.negative_prob > 0:
                    # inter_k_embed = self.get_negative_interaction_embed(q, r) # hard negative
                    inter_k_embed, _ = self.get_interaction_embed(q, neg_r, diff)

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
                        )
                        ques_j_score, _ = block(
                            mask=2,
                            query=ques_j_score,
                            key=ques_j_score,
                            values=ques_j_score,
                            apply_pos=False,
                        )
                if self.choose_cl in ["s_cl", "both"]:

                    for block in self.interaction_encoder:
                        inter_i_score, _ = block(
                            mask=2,
                            query=inter_i_score,
                            key=inter_i_score,
                            values=inter_i_score,
                            diff = si_diff_ox,
                            apply_pos=False,
                        )
                        inter_j_score, _ = block(
                            mask=2,
                            query=inter_j_score,
                            key=inter_j_score,
                            values=inter_j_score,
                            diff = sj_diff_ox,
                            apply_pos=False,
                        )
                        if self.negative_prob > 0:
                            inter_k_score, _ = block(
                                mask=2,
                                query=inter_k_embed,
                                key=inter_k_embed,
                                values=inter_k_embed,
                                diff = neg_diff,
                                apply_pos=False,
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
            
            if self.token_num < 1000 :  
                diff = torch.ceil(diff * (self.token_num-1)).long()
                s_diff_ox = torch.where(r == 1 ,  (diff - self.token_num) * (r > -1).int(), diff * (r > -1).int())
            else:
                diff = diff * 100
                s_diff_ox = torch.where(r == 1 ,  (diff - 100) * (r > -1).int(), diff * (r > -1).int())
            
            question_cl_loss, interaction_cl_loss = 0, 0


        q_embed = self.question_embed(q)
        i_embed, demb = self.get_interaction_embed(q, r, diff)

        x, y = q_embed, i_embed
        for block in self.question_encoder:
            x, _ = block(mask=1, query=x, key=x, values=x, apply_pos=True)

        for i, block in enumerate(self.interaction_encoder):
            if i>0 and self.de == "lsde": y += demb
            y, _ = block(mask=1, query=y, key=y, values=y, diff=s_diff_ox, apply_pos=True)

        for block in self.knoweldge_retriever:
            x, attn = block(mask=0, query=x, key=x, values=y, apply_pos=True)

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

    def get_interaction_embed(self, skills, responses, diff=None):
        masked_responses = responses * (responses > -1).long()
        interactions = skills + self.num_skills * masked_responses
        output = self.interaction_embed(interactions)
        if self.de in ["sde", "lsde"]:
            diffx = self.token_num + diff * (responses > -1).long()
            diffo = diff * (responses > -1).int()
            diffox = torch.where(responses == 1 ,diffo, diffx)
            demb = self.diff_emb(diffox).float()
            output += demb
            return output, demb
        else:
            return output, None


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
