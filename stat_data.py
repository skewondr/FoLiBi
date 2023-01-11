# This code is based on the following repositories:
#  1. https://github.com/theophilee/learner-performance-prediction/blob/master/prepare_data.py
#  2. https://github.com/THUwangcy/HawkesKT/blob/main/data/Preprocess.ipynb

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import os
import pickle
import time
from IPython import embed

# Please specify your dataset Path
BASE_PATH = "./dataset"

def get_stat(data_name, df=None):
    """
    This is forked from:
    https://github.com/THUwangcy/HawkesKT/blob/main/data/Preprocess.ipynb
    """
    if df is None : 
        df_path = os.path.join(os.path.join(BASE_PATH, data_name), "preprocessed_df.csv")
        df = pd.read_csv(df_path, sep="\t")

    print("skill_min", df["skill_id"].min())
    users = df["user_id"].unique()
    # df["skill_id"] += 1  # zero for padding
    # df["item_id"] += 1  # zero for padding
    num_skills, num_skills2 = df["skill_id"].max(), len(df["skill_id"].unique())
    num_questions, num_questions2 = df["item_id"].max(), len(df["item_id"].unique())

    print(f"# user: {len(users)}")
    print(f"# skill: {num_skills}, {num_skills2}")
    print(f"# question: {num_questions}, {num_questions2}")
    print(f"# interaction: {len(df)}")

    # # user_id	item_id	timestamp	correct	skill_id
    # diff_df = df.pivot_table(index=['correct'], columns='skill_id', aggfunc='size', fill_value=0)
    # diff_df.loc[len(diff_df.index)] = diff_df.loc[0] + diff_df.loc[1]
    # diff_df.loc[len(diff_df.index)] = diff_df.loc[1] / diff_df.loc[2] * 100
    # print(f"mean of correct ratio: {diff_df.loc[3].mean():.2f}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess DKT datasets")
    parser.add_argument("--data_name", type=str, default="assistments09")
    parser.add_argument("--min_user_inter_num", type=int, default=5)
    parser.add_argument("--remove_nan_skills", default=True, action="store_true")
    args = parser.parse_args()

    get_stat(
        data_name=args.data_name,
    )