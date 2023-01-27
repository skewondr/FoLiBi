import pandas as pd
import numpy as np
import torch
import os
import glob
import csv
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def model_train(
    dir_name,
    fold,
    model,
    accelerator,
    opt,
    train_loader,
    valid_loader,
    test_loader,
    config,
    n_gpu,
    early_stop=True,
):
    train_losses = []
    avg_train_losses = []
    best_valid_auc = 0

    logs_df = pd.DataFrame()
    num_epochs = config["train_config"]["num_epochs"]
    model_name = config["model_name"]
    data_name = config["data_name"]
    train_config = config["train_config"]
    valid_balanced = train_config["valid_balanced"]

    token_cnts = 0
    label_sums = 0
    for i in range(1, num_epochs + 1):
        for batch in tqdm(train_loader):
            opt.zero_grad()

            model.train()
            out_dict = model(batch)
            if n_gpu > 1:
                loss, token_cnt, label_sum = model.module.loss(batch, out_dict)
            else:
                loss, token_cnt, label_sum = model.loss(batch, out_dict)
            
            accelerator.backward(loss)

            token_cnts += token_cnt
            label_sums += label_sum

            if train_config["max_grad_norm"] > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_config["max_grad_norm"]
                )

            opt.step()
            train_losses.append(loss.item())

        print("token_cnts", token_cnts, "label_sums", label_sums)

        total_preds, total_trues = [], []
        total_preds_balanced, total_trues_balanced = [], []

        with torch.no_grad():
            for batch in valid_loader:
                model.eval()

                out_dict = model(batch)
                pred = out_dict["pred"].flatten()
                true = out_dict["true"].flatten()
                mask = true > -1
                pred = pred[mask]
                true = true[mask]

                total_preds.append(pred)
                total_trues.append(true)

                ## For balanced evaluation
                padding_mask = out_dict["true"] > -1 ## mask tensor for padding
                n_samples = padding_mask.sum(1)
                correct_samples_per_user = (batch['responses'][:, 1:] * padding_mask).sum(1)
                incorrect_samples_per_user = n_samples - correct_samples_per_user
                differences = correct_samples_per_user - incorrect_samples_per_user
                for u in range(differences.shape[0]):
                    difference_user = differences[u]
                    n_samples_user = n_samples[u]
                    start_idx = len(out_dict['true'][u]) - n_samples_user
                    true = out_dict["true"][u][start_idx:]
                    if difference_user > 0: ## Correct samples are more than incorrect samples, random under sampling on correct samples
                        indices = torch.nonzero(true).flatten()
                    elif difference_user < 0: ## Incorrect samples are more than correct samples, random under sampling on incorrect samples
                        indices = torch.nonzero(1 - true).flatten()

                    additional_mask_indices = start_idx + indices[np.random.choice(len(indices), size=torch.abs(difference_user).item(), replace=False)]
                    padding_mask[u][additional_mask_indices] = False

                pred = out_dict["pred"].flatten()
                true = out_dict["true"].flatten()
                padding_mask = padding_mask.flatten()
                pred_balanced = pred[padding_mask]
                true_balanced = true[padding_mask]            
                total_preds_balanced.append(pred_balanced)
                total_trues_balanced.append(true_balanced)
                    
            total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
            total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

            total_preds_balanced = torch.cat(total_preds_balanced).squeeze(-1).detach().cpu().numpy()
            total_trues_balanced = torch.cat(total_trues_balanced).squeeze(-1).detach().cpu().numpy()
            

        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        valid_auc_balanced = roc_auc_score(y_true=total_trues_balanced, y_score=total_preds_balanced)
        valid_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

        if valid_balanced:
            early_stop_valid = valid_auc_balanced
        else:
            early_stop_valid = valid_auc

        path = os.path.join("saved_model", model_name, data_name)
        if not os.path.isdir(path):
            os.makedirs(path)

        if early_stop_valid > best_valid_auc:

            path = os.path.join(
                dir_name, f"{fold}_params_*"
            )
            for _path in glob.glob(path):
                os.remove(_path)
            best_valid_auc = early_stop_valid
            best_epoch = i
            torch.save(
                {"epoch": i, "model_state_dict": model.state_dict()},
                os.path.join(dir_name, f"{fold}_params_best.pt")
                )
        
        if i - best_epoch > 10:
            break

        # clear lists to track next epochs
        train_losses = []
        valid_losses = []

        total_preds, total_trues = [], []

        print(f"Fold {fold}:\t Epoch {i}\tTRAIN LOSS: {train_loss:.4f}\tVALID AUC: {valid_auc:.4f}\tVALID AUC(Balanced): {valid_auc_balanced:.4f}")
        
    checkpoint = torch.load(os.path.join(dir_name, f"{fold}_params_best.pt"))

    model.load_state_dict(checkpoint["model_state_dict"])

    total_preds, total_trues = [], []
    total_preds_balanced, total_trues_balanced = [], []


    # evaluation on test dataset
    with torch.no_grad():
        for batch in test_loader:

            model.eval()

            out_dict = model(batch)

            pred = out_dict["pred"].flatten()
            true = out_dict["true"].flatten()
            mask = true > -1
            pred = pred[mask]
            true = true[mask]
            total_preds.append(pred)
            total_trues.append(true)
            
            ## For balanced evaluation
            padding_mask = out_dict["true"] > -1 ## mask tensor for padding
            n_samples = padding_mask.sum(1)
            correct_samples_per_user = (batch['responses'][:, 1:] * padding_mask).sum(1)
            incorrect_samples_per_user = n_samples - correct_samples_per_user
            differences = correct_samples_per_user - incorrect_samples_per_user
            for u in range(differences.shape[0]):
                difference_user = differences[u]
                n_samples_user = n_samples[u]
                start_idx = len(out_dict['true'][u]) - n_samples_user
                true = out_dict["true"][u][start_idx:]

                if difference_user > 0: ## Correct samples are more than incorrect samples, random under sampling on correct samples
                    indices = torch.nonzero(true).flatten()
                elif difference_user < 0: ## Incorrect samples are more than correct samples, random under sampling on incorrect samples
                    indices = torch.nonzero(1 - true).flatten()

                additional_mask_indices = start_idx + indices[np.random.choice(len(indices), size=torch.abs(difference_user).item(), replace=False)]
                padding_mask[u][additional_mask_indices] = False

            pred = out_dict["pred"].flatten()
            true = out_dict["true"].flatten()
            padding_mask = padding_mask.flatten()
            pred_balanced = pred[padding_mask]
            true_balanced = true[padding_mask]            
            total_preds_balanced.append(pred_balanced)
            total_trues_balanced.append(true_balanced)

        total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
        total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

        total_preds_balanced = torch.cat(total_preds_balanced).squeeze(-1).detach().cpu().numpy()
        total_trues_balanced = torch.cat(total_trues_balanced).squeeze(-1).detach().cpu().numpy()

    auc = roc_auc_score(y_true=total_trues, y_score=total_preds)
    acc = accuracy_score(y_true=total_trues >= 0.5, y_pred=total_preds >= 0.5)
    rmse = np.sqrt(mean_squared_error(y_true=total_trues, y_pred=total_preds))

    auc_balanced = roc_auc_score(y_true=total_trues_balanced, y_score=total_preds_balanced)
    acc_balanced = accuracy_score(y_true=total_trues_balanced >= 0.5, y_pred=total_preds_balanced >= 0.5)
    rmse_balanced = np.sqrt(mean_squared_error(y_true=total_trues_balanced, y_pred=total_preds_balanced))

    print(f"[ORIGINAL] Best Model\tTEST AUC: {auc:.4f}\tTEST ACC: {acc:.4f}\tTEST RMSE: {rmse:.4f}")
    print(f"[BALANCED] Best Model\tTEST AUC: {auc_balanced:.4f}\tTEST ACC: {acc_balanced:.4f}\tTEST RMSE: {rmse_balanced:.4f}")
    print(f"Under sampling ratio: {100*(total_preds_balanced.shape[0]/total_preds.shape[0]):.2f}%")
    
    return [auc, acc, rmse, auc_balanced, acc_balanced, rmse_balanced]
        

    # logs_df = logs_df.append(
    #     pd.DataFrame(
    #         {"EarlyStopEpoch": best_epoch, "auc": auc, "acc": acc, "rmse": rmse},
    #         index=[0],
    #     ),
    #     sort=False,
    # )

    # log_out_path = os.path.join(log_path, data_name)
    # os.makedirs(log_out_path, exist_ok=True)
    # logs_df.to_csv(
    #     os.path.join(log_out_path, "{}_{}.csv".format(model_name, now)), index=False
    # )

