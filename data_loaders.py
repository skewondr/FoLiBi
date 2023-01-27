from torch.utils.data import Dataset
import os
from utils.augment_seq import (
    preprocess_qr,
    preprocess_qsr,
    augment_kt_seqs,
    replace_only,
    mask_kt_seqs,
)
import torch
from collections import defaultdict
from IPython import embed
import numpy as np
import random 
import math 
from itertools import chain

class SimCLRDatasetWrapper(Dataset):
    def __init__(
        self,
        ds: Dataset,
        seq_len: int,
        mask_prob: float,
        crop_prob: float,
        permute_prob: float,
        replace_prob: float,
        negative_prob: float,
        eval_mode=False,
    ):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.crop_prob = crop_prob
        self.permute_prob = permute_prob
        self.replace_prob = replace_prob
        self.negative_prob = negative_prob
        self.eval_mode = eval_mode

        self.num_questions = self.ds.num_questions
        self.num_skills = self.ds.num_skills
        self.q_mask_id = self.num_questions + 1
        self.s_mask_id = self.num_skills + 1
        self.easier_skills = self.ds.easier_skills
        self.harder_skills = self.ds.harder_skills
        self.diff_quantiles = self.ds.diff_quantiles

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        original_data = self.ds[index]
        q_seq = original_data["questions"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]
        attention_mask = original_data["attention_mask"]

        if self.eval_mode:
            return {
                "questions": q_seq,
                "skills": s_seq,
                "responses": r_seq,
                "attention_mask": attention_mask,
                "sdiff": original_data["sdiff"],
                "position": original_data["position"],
            }

        else:
            q_seq_list = original_data["questions"].tolist()
            s_seq_list = original_data["skills"].tolist()
            r_seq_list = original_data["responses"].tolist()
            qdiff_list = original_data["qdiff"].tolist()
            sdiff_list = original_data["sdiff"].tolist()
            qdiff_array = self.ds.qdiff_array
            sdiff_array = self.ds.sdiff_array

            t1 = augment_kt_seqs(
                q_seq_list,
                s_seq_list,
                r_seq_list,
                qdiff_list,
                sdiff_list,
                qdiff_array,
                sdiff_array,
                self.mask_prob,
                self.crop_prob,
                self.permute_prob,
                self.replace_prob,
                self.negative_prob,
                self.easier_skills,
                self.harder_skills,
                self.q_mask_id,
                self.s_mask_id,
                self.seq_len,
                seed=index,
            )

            t2 = augment_kt_seqs(
                q_seq_list,
                s_seq_list,
                r_seq_list,
                qdiff_list,
                sdiff_list,
                qdiff_array,
                sdiff_array,
                self.mask_prob,
                self.crop_prob,
                self.permute_prob,
                self.replace_prob,
                self.negative_prob,
                self.easier_skills,
                self.harder_skills,
                self.q_mask_id,
                self.s_mask_id,
                self.seq_len,
                seed=index + 1,
            )

            aug_q_seq_1, aug_s_seq_1, aug_r_seq_1, negative_r_seq, attention_mask_1, qdiff_1, sdiff_1 = t1
            aug_q_seq_2, aug_s_seq_2, aug_r_seq_2, _, attention_mask_2, qdiff_2, sdiff_2 = t2

            aug_q_seq_1 = torch.tensor(aug_q_seq_1, dtype=torch.long)
            aug_q_seq_2 = torch.tensor(aug_q_seq_2, dtype=torch.long)
            aug_s_seq_1 = torch.tensor(aug_s_seq_1, dtype=torch.long)
            aug_s_seq_2 = torch.tensor(aug_s_seq_2, dtype=torch.long)
            aug_r_seq_1 = torch.tensor(aug_r_seq_1, dtype=torch.long)
            aug_r_seq_2 = torch.tensor(aug_r_seq_2, dtype=torch.long)
            aug_qd_seq_1 = torch.tensor(qdiff_1, dtype=torch.float)
            aug_qd_seq_2 = torch.tensor(qdiff_2, dtype=torch.float)
            aug_sd_seq_1 = torch.tensor(sdiff_1, dtype=torch.float)
            aug_sd_seq_2 = torch.tensor(sdiff_2, dtype=torch.float)
            negative_r_seq = torch.tensor(negative_r_seq, dtype=torch.long)
            attention_mask_1 = torch.tensor(attention_mask_1, dtype=torch.long)
            attention_mask_2 = torch.tensor(attention_mask_2, dtype=torch.long)

            ret = {
                "questions": (aug_q_seq_1, aug_q_seq_2, q_seq),
                "skills": (aug_s_seq_1, aug_s_seq_2, s_seq),
                "responses": (aug_r_seq_1, aug_r_seq_2, r_seq, negative_r_seq),
                "attention_mask": (attention_mask_1, attention_mask_2, attention_mask),
                "qdiff": (aug_qd_seq_1, aug_qd_seq_2, original_data["qdiff"]),
                "sdiff": (aug_sd_seq_1, aug_sd_seq_2, original_data["sdiff"]),
                "position": original_data["position"],
            }
            try:
                assert torch.max(ret["sdiff"][0]) < 11 and torch.max(ret["sdiff"][1]) < 11 and torch.max(ret["sdiff"][2]) < 11 
            except: embed()
            return ret

    def __getitem__(self, index):
        return self.__getitem_internal__(index)

class MKMDatasetWrapper(Dataset):
    def __init__(
        self,
        diff_order: str,
        ds: Dataset,
        seq_len: int,
        mask_prob: float,
        eval_mode=False,
    ):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.eval_mode = eval_mode

        self.num_questions = self.ds.num_questions
        self.num_skills = self.ds.num_skills
        self.q_mask_id = self.num_questions + 1
        self.s_mask_id = self.num_skills + 1
        self.diff_order = diff_order
        self.diff_quantiles = self.ds.diff_quantiles

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        original_data = self.ds[index]
        q_seq = original_data["questions"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]
        attention_mask = original_data["attention_mask"]

        if self.eval_mode:
            return {
                "questions": q_seq,
                "skills": s_seq,
                "responses": r_seq,
                "attention_mask": attention_mask,
                "qdiff": original_data["qdiff"],
                "sdiff": original_data["sdiff"],
                "position": original_data["position"],
            }

        else:
            q_seq_list = original_data["questions"].tolist()
            s_seq_list = original_data["skills"].tolist()
            r_seq_list = original_data["responses"].tolist()
            qdiff_list = original_data["qdiff"].tolist()
            sdiff_list = original_data["sdiff"].tolist()
            
            rng = random.Random(index)
            
            true_seq_len = np.sum(np.asarray(q_seq_list) != 0)
            if self.diff_order.startswith("asc"):
                pos_index = sorted(range(len(sdiff_list)-true_seq_len, len(sdiff_list)), key=lambda k: sdiff_list[k])
            elif self.diff_order == "des":
                pos_index = sorted(range(len(sdiff_list)-true_seq_len, len(sdiff_list)), key=lambda k: sdiff_list[k], reverse=True)
            elif self.diff_order == "random":
                pos_index = list(range(len(sdiff_list)-true_seq_len, len(sdiff_list)))
                rng.shuffle(pos_index)
            elif self.diff_order == "chunk":
                pos_index = list(range(len(sdiff_list)-true_seq_len, len(sdiff_list)))

            crop_seq_len = max(1, math.floor(self.mask_prob * true_seq_len))
            if self.diff_order in ["chunk", "asc_chunk"]:
                start_idx = rng.randint(0, true_seq_len - crop_seq_len) #include index true_seq_len - crop_seq_len
            else: 
                start_idx = 0

            pos_index = pos_index[start_idx : start_idx + crop_seq_len]
            neg_index = list(set(range(len(sdiff_list)-true_seq_len, len(sdiff_list)))-set(pos_index))
            diff_index = (pos_index, neg_index)
            
            t1 = mask_kt_seqs(
                "mask",
                diff_index,
                q_seq_list,
                s_seq_list,
                r_seq_list,
                qdiff_list,
                sdiff_list,
                self.q_mask_id,
                self.s_mask_id,
                self.seq_len,
                seed=index,
            )
            aug_q_seq_1, aug_s_seq_1, aug_r_seq_1, attention_mask_1, attention_mask_neg, qdiff_1, sdiff_1 = t1
            aug_q_seq_1 = torch.tensor(aug_q_seq_1, dtype=torch.long)
            aug_s_seq_1 = torch.tensor(aug_s_seq_1, dtype=torch.long)
            aug_r_seq_1 = torch.tensor(aug_r_seq_1, dtype=torch.long)
            aug_qd_seq_1 = torch.tensor(qdiff_1, dtype=torch.float)
            aug_sd_seq_1 = torch.tensor(sdiff_1, dtype=torch.float)
            attention_mask_1 = torch.tensor(attention_mask_1, dtype=torch.long)
            attention_mask_neg = torch.tensor(attention_mask_neg, dtype=torch.long)

            ret = {
                "questions": (aug_q_seq_1, q_seq),
                "skills": (aug_s_seq_1, s_seq),
                "responses": (aug_r_seq_1, r_seq),
                "attention_mask": (attention_mask_1, attention_mask, attention_mask_neg),
                "qdiff": (aug_qd_seq_1, original_data["qdiff"]),
                "sdiff": (aug_sd_seq_1, original_data["sdiff"]),
                "position": original_data["position"],
            }
            try:
                assert torch.max(ret["sdiff"][0]) < 11 and torch.max(ret["sdiff"][1]) < 11 
            except: embed()
            return ret

    def __getitem__(self, index):
        return self.__getitem_internal__(index)

def get_diff_df(df, seq_len, num_skills, num_questions, total_cnt_init=1, diff_unk=0.0):
    q_total_cnt = np.ones((num_questions+1))
    q_crt_cnt = np.zeros((num_questions+1))

    c_total_cnt = np.ones((num_skills+1))
    c_crt_cnt = np.zeros((num_skills+1))
    
    if total_cnt_init == 0:
        q_total_cnt = np.zeros((num_questions+1))
        c_total_cnt = np.zeros((num_skills+1))

    for q, c, r in zip(df["item_id"], df["skill_id"], df["correct"]):
        c_total_cnt[c] += 1
        if r:
            c_crt_cnt[c] += 1
        q_total_cnt[q] += 1
        if r:
            q_crt_cnt[q] += 1

    if diff_unk != 0.0: ## else unk is zero
        q_crt_cnt[np.where(q_total_cnt == total_cnt_init)] = diff_unk
        c_crt_cnt[np.where(c_total_cnt == total_cnt_init)] = diff_unk

    if total_cnt_init == 0:
        q_total_cnt = np.where(q_total_cnt == 0, 1, q_total_cnt)
        c_total_cnt = np.where(c_total_cnt == 0, 1, c_total_cnt)

    q_diff = q_crt_cnt/q_total_cnt
    c_diff = c_crt_cnt/c_total_cnt
    df = df.assign(item_diff=q_diff[np.array(df["item_id"].values)])
    df = df.assign(skill_diff=c_diff[np.array(df["skill_id"].values)])
    
    print("-"*80)
    s_df = df.loc[:, ['skill_id', 'skill_diff']]
    s_df = s_df.drop_duplicates(subset=["skill_id"]).sort_values(by='skill_id')
    q_df = df.loc[:, ['item_id', 'item_diff']]
    q_df = q_df.drop_duplicates(subset=["item_id"]).sort_values(by='item_id')

    responses = [
        u_df["correct"].values[-seq_len :]
        for _, u_df in df.groupby("user_id")
    ]
    
    print(f"mean of total set, skill correct ratio:{np.mean(s_df['skill_diff']*100):.2f}")
    print(f"mean of total set, question correct ratio:{np.mean(q_df['item_diff']*100):.2f}")
    print(f"mean of total set, class 0 ratio:{sum(list(x).count(0) for x in responses)/sum(len(x) for x in responses):.2f}")
    print(f"mean of total set, class 1 ratio:{sum(list(x).count(1) for x in responses)/sum(len(x) for x in responses):.2f}")
    print("-"*80)

    return df

def get_balanced(seq_len, objects, rm_index=None): 
    if rm_index is not None : output = [i for idx, i in enumerate(objects) if idx not in rm_index]
    else: output = objects
    return output[-seq_len :]


class MostRecentQuestionSkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions, diff_df, diff_quantiles, name="train"):
        
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.diff_quantiles = diff_quantiles
        
        self.questions, self.skills, self.responses, self.skill_diff, self.question_diff = [], [], [], [], []
        for _, u_df in self.df.groupby("user_id"):
            r1 = get_balanced(self.seq_len, u_df["item_id"].values)
            r2 = get_balanced(self.seq_len, u_df["skill_id"].values)
            r3 = get_balanced(self.seq_len, u_df["correct"].values)
            r4 = get_balanced(self.seq_len, u_df["skill_diff"].values)
            r5 = get_balanced(self.seq_len, u_df["item_diff"].values)
            if len(r1) >= 5: 
                self.questions.append(r1)
                self.skills.append(r2)
                self.responses.append(r3)
                self.skill_diff.append(r4)
                self.question_diff.append(r5)
  
        s_df = diff_df.loc[:, ['skill_id', 'skill_diff']]
        s_df = s_df.drop_duplicates(subset=["skill_id"]).sort_values(by='skill_id')
        q_df = diff_df.loc[:, ['item_id', 'item_diff']]
        q_df = q_df.drop_duplicates(subset=["item_id"]).sort_values(by='item_id')
            
        # if name == "train":
        print(f"# of {name} set, {len(self.responses)}")
        print(f"mean of {name} set, skill correct ratio:{np.mean(s_df['skill_diff']*100):.2f}")
        print(f"mean of {name} set, question correct ratio:{np.mean(q_df['item_diff']*100):.2f}")
        print(f"mean of {name} set, class 0 ratio:{sum(list(x).count(0) for x in self.responses)/sum(len(x) for x in self.responses):.2f}")
        print(f"mean of {name} set, class 1 ratio:{sum(list(x).count(1) for x in self.responses)/sum(len(x) for x in self.responses):.2f}")
        print("-"*80)
        
        skill_difficulty = dict(zip(s_df.skill_id, s_df.skill_diff))
        self.ordered_skills = [
            item[0] for item in sorted(skill_difficulty.items(), key=lambda x: x[1])
        ]
        question_difficulty = dict(zip(q_df.item_id, q_df.item_diff))

        self.sdiff_array = np.zeros(self.num_skills+1)
        self.qdiff_array = np.zeros(self.num_questions+1)
        self.sdiff_array[list(skill_difficulty.keys())] = np.array(list(skill_difficulty.values()))
        self.qdiff_array[list(question_difficulty.keys())] = np.array(list(question_difficulty.values()))
        self.sdiff_array = torch.bucketize(torch.tensor(self.sdiff_array), self.diff_quantiles).cpu().detach().numpy()

        self.easier_skills = {}
        self.harder_skills = {}
        for i, s in enumerate(self.ordered_skills):
            if i == 0:  # the hardest
                self.easier_skills[s] = self.ordered_skills[i + 1]
                self.harder_skills[s] = s
            elif i == len(self.ordered_skills) - 1:  # the easiest
                self.easier_skills[s] = s
                self.harder_skills[s] = self.ordered_skills[i - 1]
            else:
                self.easier_skills[s] = self.ordered_skills[i + 1]
                self.harder_skills[s] = self.ordered_skills[i - 1]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )
        self.padded_sd = torch.full(
            (len(self.skills), self.seq_len), -1, dtype=torch.float
        )
        self.padded_qd = torch.full(
            (len(self.questions), self.seq_len), -1, dtype=torch.float
        )
        self.position = torch.full(
            (len(self.questions), self.seq_len), 0, dtype=torch.long
        )

    def __getitem__(self, index):
        try:
            q, s, r = self.questions[index], self.skills[index], self.responses[index]
            sd = self.sdiff_array[s]
            qd = self.qdiff_array[q]
            self.padded_q[index, -len(q) :] = torch.tensor(q, dtype=torch.long)
            self.padded_s[index, -len(s) :] = torch.tensor(s, dtype=torch.long)
            self.padded_r[index, -len(r) :] = torch.tensor(r, dtype=torch.long)
            self.attention_mask[index, -len(s) :] = torch.ones(len(s), dtype=torch.long)
            self.padded_sd[index, -len(s) :] = torch.tensor(sd, dtype=torch.float)
            self.padded_qd[index, -len(q) :] = torch.tensor(qd, dtype=torch.float)
            self.position[index, -len(s) :] = torch.arange(1, len(s)+1, dtype=torch.long)
        except: embed()

        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "attention_mask": self.attention_mask[index],
            "sdiff": self.padded_sd[index],
            "qdiff": self.padded_qd[index],
            "position": self.position[index],
        }


    def __len__(self):
        return self.len


class MostEarlyQuestionSkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["item_id"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.skills = [
            u_df["skill_id"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )

        for i, elem in enumerate(zip(self.questions, self.skills, self.responses)):
            q, s, r = elem
            self.padded_q[i, : len(q)] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            self.attention_mask[i, : len(r)] = torch.ones(len(s), dtype=torch.long)

    def __getitem__(self, index):
        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return self.len


class SkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["skill_id"].values for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.questions, self.responses = preprocess_qr(
            self.questions, self.responses, self.seq_len
        )
        self.len = len(self.questions)

    def __getitem__(self, index):
        return {"questions": self.questions[index], "responses": self.responses[index]}

    def __len__(self):
        return self.len
