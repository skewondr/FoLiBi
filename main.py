import os
import argparse
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import yaml
from data_loaders import (
    MostRecentQuestionSkillDataset,
    MostEarlyQuestionSkillDataset,
    SimCLRDatasetWrapper,
    MKMDatasetWrapper,
    get_diff_df,
)
from models.akt import AKT
from models.sakt import SAKT
from models.saint import SAINT
from models.cl4kt import CL4KT
from models.rdemkt import RDEMKT
from train import model_train
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from utils.config import ConfigNode as CN
from utils.file_io import PathManager
from stat_data import get_stat
import wandb 
import time 
from time import localtime 
import statistics 
import json
import random 

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    
def main(config):

    tm = localtime(time.time())
    params_str = f'{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}:{tm.tm_min}:{tm.tm_sec}'
    if config.use_wandb:
        wandb.init(project="SIGIR", entity="skewondr")
        wandb.run.name = params_str
        wandb.run.save()

    accelerator = Accelerator()
    device = accelerator.device

    model_name = config.model_name
    dataset_path = config.dataset_path
    data_name = config.data_name
    seed = config.seed

    np.random.seed(seed)
    torch.manual_seed(seed)

    df_path = os.path.join(os.path.join(dataset_path, data_name), "preprocessed_df.csv")

    train_config = config.train_config
    checkpoint_dir = config.checkpoint_dir
    
    seed = train_config.seed
    set_seed(seed)

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ckpt_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, data_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = train_config.batch_size
    eval_batch_size = train_config.eval_batch_size
    learning_rate = train_config.learning_rate
    optimizer = train_config.optimizer
    seq_len = train_config.seq_len
    diff_order = train_config.diff_order
    diff_as_loss_weight = train_config.diff_as_loss_weight
    uniform = train_config.uniform

    if train_config.sequence_option == "recent":  # the most recent N interactions
        dataset = MostRecentQuestionSkillDataset
    elif train_config.sequence_option == "early":  # the most early N interactions
        dataset = MostEarlyQuestionSkillDataset
    else:
        raise NotImplementedError("sequence option is not valid")

    # test_aucs, test_accs, test_rmses = [], [], []
    # test_aucs_balanced, test_accs_balanced, test_rmses_balanced = [], [], []
    # test_aucs_weighted, test_accs_weighted, test_rmses_weighted = [], [], []
    test_result = []
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    df = pd.read_csv(df_path, sep="\t")

    users = df["user_id"].unique()
    np.random.shuffle(users)
    get_stat(data_name, df)
    df["skill_id"] += 1  # zero for padding
    df["item_id"] += 1  # zero for padding
    num_skills = df["skill_id"].max() + 1
    num_questions = df["item_id"].max() + 1
        
    print("MODEL", model_name)
    print(dataset)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(users)):
        if fold >= 1 : break
        train_users = users[train_ids]
        np.random.shuffle(train_users)
        offset = int(len(train_ids) * 0.9)

        valid_users = train_users[offset:]
        train_users = train_users[:offset]
        test_users = users[test_ids]

        df = get_diff_df(df, seq_len, num_skills, num_questions, total_cnt_init=config.total_cnt_init, diff_unk=config.diff_unk)
        train_df = df[df["user_id"].isin(train_users)]

        train_quantiles = None
        train_bincounts = None
        token_num = int(args.de_type.split('_')[1])
        boundaries = np.linspace(0, 1, num=token_num+1)   
        train_quantiles = torch.Tensor([train_df['skill_diff'].quantile(i) for i in boundaries])            
        if uniform:
            boundaries = torch.Tensor(boundaries)
            train_diff_buckets = torch.bucketize(torch.Tensor(train_df['skill_diff'].to_numpy()), boundaries)
            diff_quantiles = boundaries
        else:
            train_diff_buckets = torch.bucketize(torch.Tensor(train_df['skill_diff'].to_numpy()), train_quantiles)
            diff_quantiles = train_quantiles
        train_bincounts = torch.bincount(train_diff_buckets)

        valid_df = df[df["user_id"].isin(valid_users)]
        test_df = df[df["user_id"].isin(test_users)]
        
        train_dataset = dataset(train_df, seq_len, num_skills, num_questions, diff_df= train_df, diff_quantiles=diff_quantiles, name="train")
        valid_dataset = dataset(valid_df, seq_len, num_skills, num_questions, diff_df= train_df, diff_quantiles=diff_quantiles, name="valid")
        test_dataset = dataset(test_df, seq_len, num_skills, num_questions, diff_df= train_df, diff_quantiles=diff_quantiles, name="test")

        print("train_ids", len(train_users))
        print("valid_ids", len(valid_users))
        print("test_ids", len(test_users))


        if model_name == "akt":
            model_config = config.akt_config
            if data_name in ["statics", "assistments15"]:
                num_questions = 0
            model = AKT(device, num_skills, num_questions, seq_len, train_bincounts, **model_config)
        elif model_name == "cl4kt":
            model_config = config.cl4kt_config
            model = CL4KT(device, num_skills, num_questions, seq_len, train_bincounts, **model_config)
            mask_prob = model_config.mask_prob
            crop_prob = model_config.crop_prob
            permute_prob = model_config.permute_prob
            replace_prob = model_config.replace_prob
            negative_prob = model_config.negative_prob
        elif args.model_name == "sakt":
            model_config = config.sakt_config
            model = SAKT(device, num_skills, num_questions, seq_len, train_bincounts, **model_config)
        elif args.model_name == "saint":
            model_config = config.saint_config
            model = SAINT(device, num_skills, num_questions, seq_len, **model_config)
        elif model_name == "rdemkt":
            model_config = config.rdemkt_config
            model = RDEMKT(device, num_skills, num_questions, seq_len, train_bincounts, **model_config)
            mask_prob = model_config.mask_prob

        dir_name = os.path.join("saved_model", model_name, data_name, params_str)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(os.path.join(dir_name, "configs.json"), 'w') as f:
            json.dump(model_config, f)
            json.dump(train_config, f)
        
        print(train_config)
        print(model_config)

        if model_name == "cl4kt":   
            train_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        train_dataset,
                        seq_len,
                        mask_prob,
                        crop_prob,
                        permute_prob,
                        replace_prob,
                        negative_prob,
                        eval_mode=False,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        valid_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        test_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )

        elif model_name == "rdemkt":   
            train_loader = accelerator.prepare(
                DataLoader(
                    MKMDatasetWrapper(
                        diff_order,
                        train_dataset,
                        seq_len,
                        mask_prob,
                        eval_mode=False,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    MKMDatasetWrapper(
                        diff_order, valid_dataset, seq_len, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    MKMDatasetWrapper(
                        diff_order, test_dataset, seq_len, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )

        else:
            train_loader = accelerator.prepare(
                DataLoader(train_dataset, batch_size=batch_size)
            )

            valid_loader = accelerator.prepare(
                DataLoader(valid_dataset, batch_size=eval_batch_size)
            )

            test_loader = accelerator.prepare(
                DataLoader(test_dataset, batch_size=eval_batch_size)
            )

        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate, weight_decay=train_config.l2)

        model, opt = accelerator.prepare(model, opt)

        t1 = model_train(
            dir_name,
            fold,
            model,
            accelerator,
            opt,
            train_loader,
            valid_loader,
            test_loader,
            config,
            n_gpu
        ) #t1 = [test_auc, test_acc, test_rmse]

        test_result.append(t1) # fold, 9

    print_args = dict()
    metric_type = ['d', 'b', 'w']
    metric = ['auc', 'acc', 'rmse']
    #총 9개의 n-fold 평균 결과가 나와야 함. 
    # from IPython import embed ; embed()
    for index in range(len(test_result[0])):
        fold_total = []
        for fold in range(len(test_result)):
            fold_total.append(test_result[fold][index])
        print_args[f'{metric[index%3]}_{metric_type[index//3]}'] = np.mean(fold_total)

    if config.use_wandb:
        print_args['Model'] = model_name 
        print_args['Dataset'] = data_name 
        print_args.update(train_config)
        print_args.update(model_config)
        wandb.log(print_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="cl4kt",
        help="The name of the model to train. \
            The possible models are in [akt, cl4kt]. \
            The default model is cl4kt.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="algebra05",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--reg_cl",
        type=float,
        default=0.1,
        help="regularization parameter contrastive learning loss",
    )
    parser.add_argument(
        "--reg_l",
        type=float,
        default=0.1,
        help="regularization parameter learning loss",
    )
    parser.add_argument("--mask_prob", type=float, default=0.2, help="mask probability")
    parser.add_argument("--crop_prob", type=float, default=0.3, help="crop probability")
    parser.add_argument(
        "--permute_prob", type=float, default=0.3, help="permute probability"
    )
    parser.add_argument(
        "--replace_prob", type=float, default=0.3, help="replace probability"
    )
    parser.add_argument(
        "--negative_prob",
        type=float,
        default=1.0,
        help="reverse responses probability for hard negative pairs",
    )
    parser.add_argument(
        "--inter_lambda", type=float, default=1, help="loss lambda ratio for regularization"
    )
    parser.add_argument(
        "--ques_lambda", type=float, default=1, help="loss lambda ratio for regularization"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout probability"
    )
    parser.add_argument(
        "--batch_size", type=float, default=512, help="train batch size"
    )
    parser.add_argument(
        "--only_rp", type=int, default=0, help="train with only rp model"
    )
    parser.add_argument(
        "--choose_cl", type=str, default="both", help="choose between q_cl and s_cl"
    )
    parser.add_argument(
        "--describe", type=str, default="default", help="description of the training"
    )
    parser.add_argument(
        "--diff_order", type=str, default="random", help="random/des/asc/chunk"
    )
    parser.add_argument(
        "--use_wandb", type=int, default=1
    )
    parser.add_argument("--l2", type=float, default=0.0, help="l2 regularization param")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    
    parser.add_argument("--total_cnt_init", type=int, default=0, help="total_cnt_init")
    parser.add_argument("--diff_unk", type=float, default=0.5, help="diff_unk")
    
    parser.add_argument("--gpu_num", type=int, required=True, help="gpu number")
    parser.add_argument("--server_num", type=str, required=True, help="server number")

    parser.add_argument("--diff_as_loss_weight", action="store_true", default=False, help="diff_as_loss_weight")
    parser.add_argument("--valid_balanced", action="store_true", default=False, help="valid_balanced")
    parser.add_argument("--exponential", action="store_true", default=True, help="exponential function for forgetting behavior")
    parser.add_argument("--uniform", action="store_true", default=False, help="uniform or quantiles for difficulty")
    parser.add_argument("--seed",  type=int, default=12405, help="seed")
    
    parser.add_argument("--de_type", type=str, default="none_0", help="difficulty encoding")
    parser.add_argument("--choose_enc", type=str, default="f", help="choose encoder")
    
    parser.add_argument("--seq_len",  type=int, default=100, help="max sequence length")
    args = parser.parse_args()

    base_cfg_file = PathManager.open("configs/example_opt.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.use_wandb = args.use_wandb
    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.learning_rate = args.lr
    cfg.train_config.optimizer = args.optimizer
    cfg.train_config.describe = args.describe
    cfg.train_config.gpu_num = args.gpu_num
    cfg.train_config.server_num = args.server_num
    cfg.train_config.diff_as_loss_weight = args.diff_as_loss_weight
    cfg.train_config.valid_balanced = args.valid_balanced
    cfg.train_config.uniform = args.uniform
    cfg.train_config.seed = args.seed
    cfg.train_config.seq_len = args.seq_len
    
    cfg.total_cnt_init = args.total_cnt_init
    cfg.diff_unk = args.diff_unk
    
    if args.model_name == "cl4kt":
        cfg.cl4kt_config = cfg.cl4kt_config[cfg.data_name]
        cfg.cl4kt_config.only_rp = 0
        cfg.cl4kt_config.choose_cl = args.choose_cl
        # cfg.cl4kt_config.reg_cl = args.reg_cl
        # cfg.cl4kt_config.mask_prob = args.mask_prob
        # cfg.cl4kt_config.crop_prob = args.crop_prob
        # cfg.cl4kt_config.permute_prob = args.permute_prob
        # cfg.cl4kt_config.replace_prob = args.replace_prob
        # cfg.cl4kt_config.negative_prob = args.negative_prob
        # cfg.cl4kt_config.dropout = args.dropout
        # cfg.cl4kt_config.l2 = args.l2
    elif args.model_name == "akt":
        cfg.akt_config = cfg.akt_config[cfg.data_name]
    #     cfg.akt_config.l2 = args.l2
    #     cfg.akt_config.dropout = args.dropout
    elif args.model_name == "rdemkt":
        cfg.rdemkt_config = cfg.rdemkt_config[cfg.data_name]
        cfg.rdemkt_config.only_rp = 1
        # cfg.mkt_config.choose_cl = args.choose_cl
        # cfg.mkt_config.inter_lambda = args.inter_lambda
        # cfg.mkt_config.ques_lambda = args.ques_lambda
        # cfg.mkt_config.mask_prob = args.mask_prob
    
    cfg[f"{args.model_name}_config"].de_type =  args.de_type
    cfg[f"{args.model_name}_config"].choose_enc =  args.choose_enc
    
    cfg.freeze()

    main(cfg)
