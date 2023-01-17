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
from IPython import embed 

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class RDEMKT(Module):
    def __init__(self, device, num_skills, num_questions, seq_len, **kwargs):
        super(RDEMKT, self).__init__()
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
        self.only_rp = self.args["only_rp"]
        self.choose_cl = self.args["choose_cl"]
        self.q_reg = self.args["ques_lambda"]
        self.i_reg = self.args["inter_lambda"]
        
        self.question_embed = Embedding(
            self.num_skills + 2, self.hidden_size, padding_idx=0
        )
        self.answer_embed = Embedding(
            2 + 1, self.hidden_size, padding_idx=2
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
                    rotary="qk",
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
                    rotary="qkv",
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
        self.q_out = Linear(self.hidden_size, self.num_skills + 1) #except masked token 
        self.r_out = Linear(self.hidden_size, 1)

        self.cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.loss_fn = nn.BCELoss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, batch):
        if self.training:
            q_i, q = batch["skills"]  # augmented q_i, augmented q_j and original q
            r_i, r = batch["responses"]  # augmented r_i, augmented r_j and original r
            diff_i, diff = batch["sdiff"]
            attention_mask_i, attention_mask, attention_mask_n = batch["attention_mask"]

            if not self.only_rp:
                ques_i_embed = self.question_embed(q_i) #original
                inter_i_embed = self.get_interaction_embed(q, r_i) #masked

                # BERT
                if self.choose_cl in ["q_cl", "both"]:
                    for block in self.question_encoder:
                        ques_i_score, _ = block(
                            mask=2,
                            query=ques_i_embed,
                            key=ques_i_embed,
                            values=ques_i_embed,
                            diff = None,
                            apply_pos=False,
                        )
                if self.choose_cl in ["s_cl", "both"]:
                    si_diff_ox = torch.where(r == 1 , (diff - 1) * (r > -1).int(), diff * (r > -1).int())
                    for block in self.interaction_encoder:
                        inter_i_score, _ = block(
                            mask=2,
                            query=inter_i_embed,
                            key=inter_i_embed,
                            values=inter_i_embed,
                            diff = si_diff_ox,
                            apply_pos=False,
                        )
                if self.choose_cl in ["q_cl", "both"]:
                    q_pred = self.q_out(ques_i_score)
                    q_pred = q_pred.view(-1, q_pred.shape[-1]) # flatten
                    target = q.flatten()
                    mask = target > 0 
                    # mask = attention_mask_i.flatten()
                    question_mkm_loss = self.cl_loss_fn(q_pred[mask], target[mask]) * self.q_reg
                    # question_mkm_loss = torch.mean(question_mkm_loss)
                else: 
                    question_mkm_loss = 0
                if self.choose_cl in ["s_cl", "both"]:
                    r_pred = torch.sigmoid(self.r_out(inter_i_score))
                    r_pred = r_pred.squeeze().flatten()
                    target = r.float().flatten()
                    # mask = torch.logical_and(target > -1, attention_mask_i.flatten()) 
                    mask = target > -1 
                    interaction_mkm_loss = self.loss_fn(r_pred[mask], target[mask]) * self.i_reg
                else: 
                    interaction_mkm_loss = 0 
            else: 
                question_mkm_loss, interaction_mkm_loss = 0, 0
        else:
            q = batch["skills"]  # augmented q_i, augmented q_j and original q
            r = batch["responses"]  # augmented r_i, augmented r_j and original r
            diff = batch["sdiff"]

            attention_mask = batch["attention_mask"]
            question_mkm_loss, interaction_mkm_loss = 0, 0

        # if self.choose_cl == "none":
        #     q_rotary = None
        #     s_rotary = None
        # elif self.choose_cl == "q_rp":
        #     q_rotary = diff * (r > -1).int()
        #     s_rotary = None
        # elif self.choose_cl == "s_rp":
        diff_o = (diff - 1) * (r > -1).int()
        diff_x = diff * (r > -1).int()
        s_rotary = torch.where(r == 1 , diff_o, diff_x)
        # else:
        #     diff_o = (diff - 1) * (r > -1).int()
        #     diff_x = diff * (r > -1).int()
        #     q_rotary = diff * (r > -1).int()
        #     s_rotary = torch.where(r == 1 , diff_o, diff_x)

        q_embed = self.question_embed(q)
        i_embed = self.get_interaction_embed(q, r)

        x, y = q_embed, i_embed
        for block in self.question_encoder:
            x, _ = block(mask=1, query=x, key=x, values=x, diff=None, apply_pos=True)

        for block in self.interaction_encoder:
            y, _ = block(mask=1, query=y, key=y, values=y, diff=s_rotary, apply_pos=True)

        for block in self.knoweldge_retriever:
            x, attn = block(mask=0, query=x, key=x, values=y, diff=None, apply_pos=True)

        retrieved_knowledge = torch.cat([x, q_embed], dim=-1)

        output = torch.sigmoid(self.out(retrieved_knowledge)).squeeze()
        total_cl_loss = question_mkm_loss + interaction_mkm_loss

        if self.training:
            if not self.only_rp:
                if self.choose_cl == "q_cl":
                    total_cl_loss = question_mkm_loss
                elif self.choose_cl == "s_cl":
                    total_cl_loss = interaction_mkm_loss
                else: 
                    total_cl_loss = question_mkm_loss + interaction_mkm_loss
            else:
                total_cl_loss = 0
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

        loss = self.loss_fn(pred[mask], true[mask]) + cl_loss

        return loss, len(pred[mask]), true[mask].sum().item()

    def get_interaction_embed(self, skills, responses):
        masked_responses = torch.where(responses == -1, 2, responses)
        interactions = self.question_embed(skills) + self.answer_embed(masked_responses)
        return interactions

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