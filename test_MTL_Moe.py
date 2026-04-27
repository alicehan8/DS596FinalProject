import wandb
import torch
import os
from transformers import  AdamW
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from einops import rearrange
import sys
from einops.layers.torch import Rearrange
# from sei import *
# from pretrain_multihead import *
from Pretrain_Moe import *
# from DeepHistone import NetDeepHistone
from utils import *
import pandas as pd
from transformers import  AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForMaskedLM
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAUPRC
import pickle
# from ablution_Study import CNN_BLSTM

        
def main(args):

    Histone = args.histone # takes the histone as arg and uses that to find model
    # Histone =  "H3K27me3"
    # Histone = "H3K27ac"

    model_name = "LLM_Moe"
    # ad
    task_dict = {"task1":6,"task2":5,"task3":4,"task4":4,"task5":3}
    # for aml??
    # task_dict = {"AML": 1}
    # ad
    model_path = "./models/%s_%s.pt" %(Histone,model_name)
    # aml
    # model_path = "./models/%s_%s_aml.pt" %(Histone,model_name)

    batch_size = 8
    seq_length = 4096

    print("load datasets...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ad
    data_module = ADDataModule("./Dataset/%s_all_data.csv" %Histone,["chr10"], ["chr8","chr9"],seq_length, batch_size,pretrain=True)
    # try AML data
    # data_module = AMLDataModule("./Dataset/%s_AML_generated.csv" %Histone,["chr10"], ["chr8","chr9"],seq_length, batch_size,pretrain=True)
    test_loader = data_module.test_dataloader()    
    model = Pretrain_Moe(task_dict).to(device)

    
    checkpoint = torch.load(model_path)
    # BECAUSE MODEL IS PARALLEL BUT TEST IS NOT
    # model.load_state_dict(checkpoint['model_state_dict'])
    state_dict = checkpoint['model_state_dict']


    print("MODEL KEYS:")
    for i, k in enumerate(model.state_dict().keys()):
        print(k)
        if i > 20:
            break

    print("\nCHECKPOINT KEYS:")
    for i, k in enumerate(state_dict.keys()):
        print(k)
        if i > 20:
            break


    # remove "module." prefix
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     new_key = k.replace("module.", "")  # ONLY this
    #     new_state_dict[new_key] = v
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")           # strip DDP prefix
        k = k.replace("mixer.submamba_fwd", "mixer.submodule.mamba_fwd")   # fix renamed attr
        k = k.replace("mixer.submamba_rev", "mixer.submodule.mamba_rev")   # fix renamed attr
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    result = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys (in model but not checkpoint):", result.missing_keys)
    print("Unexpected keys (in checkpoint but not model):", result.unexpected_keys)



    print("MODEL KEYS 2:")
    for i, k in enumerate(model.state_dict().keys()):
        print(k)
        if i > 20:
            break

    print("\nCHECKPOINT KEYS 2:")
    for i, k in enumerate(state_dict.keys()):
        print(k)
        if i > 20:
            break

    prediction_all = []
    all_labels = []
    model.eval()

    # metrics
    from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAUPRC
    num_tasks = 22
    # num_tasks = 57

    # ad
    acc_metric = [BinaryAccuracy() for _ in range(num_tasks)]
    auc_metric = [BinaryAUROC() for _ in range(num_tasks)]
    f1_metric = [BinaryF1Score() for _ in range(num_tasks)]
    prc_metric = [BinaryAUPRC() for _ in range(num_tasks)]
    # aml
    # auc_metric = BinaryAUROC()
    # acc_metric = BinaryAccuracy()
    # f1_metric = BinaryF1Score()
    # prc_metric = BinaryAUPRC()

    criterion = nn.BCELoss()
    total_loss = 0
    num_batches = 0

    criterion = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        counter = 0
        for batch in test_loader:
            counter += 1
            data, labels = batch['sequence'], batch['label'] #.view(-1)
            data, labels = data.to(device), labels.to(device)
            outputs, _ = model(data)

            print("labels shape ", counter, ":", labels.shape)

            outputs = torch.concat([v for k, v in outputs.items()], dim=1)
            logits = torch.sigmoid(outputs)
            prediction_all.append(logits.cpu().detach())  # keep as tensor, not numpy

            all_labels.append(labels.cpu())

            # update emtrics
            probs = torch.sigmoid(outputs)
            loss = criterion(probs, labels)
            total_loss += loss.item()
            num_batches += 1

            num_tasks = probs.shape[1]

            # ad
            for t in range(num_tasks):
                acc_metric[t].update(probs[:, t], labels[:, t])
                auc_metric[t].update(probs[:, t], labels[:, t])
                f1_metric[t].update(probs[:, t], labels[:, t])
                prc_metric[t].update(probs[:, t], labels[:, t])
            # aml
            # preds = probs.squeeze()
            # truth = labels.squeeze()

            # auc_metric.update(preds, truth)
            # acc_metric.update(preds, truth)
            # f1_metric.update(preds, truth)
            # prc_metric.update(preds, truth)


    # Convert all at once after the loop
    prediction_all = torch.cat(prediction_all, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
            
    # with open("./test_results/%s_%s_test_result" %(Histone,model_name),"wb") as f:
    #     pickle.dump(prediction_all,f)
    # ad
    np.save("./test_results/%s_%s_test_result.npy" % (Histone, model_name), prediction_all)
    np.save("./test_results/%s_%s_labels.npy" % (Histone, model_name), labels)

    # aml
    # np.save("./test_results/%s_%s_AML_test_result.npy" % (Histone, model_name), prediction_all)
    # np.save("./test_results/%s_%s_AML_labels.npy" % (Histone, model_name), labels)

    test_loss = total_loss / num_batches

    # ad
    acc_per_task = [m.compute().item() for m in acc_metric]
    auc_per_task = [m.compute().item() for m in auc_metric]
    f1_per_task  = [m.compute().item() for m in f1_metric]
    prc_per_task = [m.compute().item() for m in prc_metric]

    results = {
        "accuracy": float(np.mean(acc_per_task)),
        "AUC": float(np.mean(auc_per_task)),
        "F1": float(np.mean(f1_per_task)),
        "PRC": float(np.mean(prc_per_task)),
    }

    #aml
    # results = {
    #     "AUC": auc_metric.compute().item(),
    #     "accuracy": acc_metric.compute().item(),
    #     "F1": f1_metric.compute().item(),
    #     "PRC": prc_metric.compute().item(),
    # }   

    print(f"test_loss={test_loss}", results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--histone", type=str,default=False, help="histone type")
    parser.add_argument('--save_model', type=bool, default=False,
                        help='For Saving the current Model')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    main(args)
