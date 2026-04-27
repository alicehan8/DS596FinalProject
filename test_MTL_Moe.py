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
from pretrain_multihead import *
from Pretrain_Moe import *
from DeepHistone import NetDeepHistone
from utils import *
import pandas as pd
from transformers import  AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForMaskedLM
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAUPRC
import pickle
from ablution_Study import CNN_BLSTM

        
def main(args):

    Histone = args.histone
    # Histone =  "H3K27me3"
    # Histone = "H3K27ac"

    model_name = "LLM_Moe"
    task_dict = {"task1":6,"task2":5,"task3":4,"task4":4,"task5":3}
    model_path = "./models/%s_%s.pt" %(Histone,model_name)
    batch_size = 8
    seq_length = 4096

    print("load datasets...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = ADDataModule("./Datasets/%s_all_data.csv" %Histone,["chr10"], ["chr8","chr9"],seq_length, batch_size,pretrain=True)
    test_loader = data_module.test_dataloader()    
    model = Pretrain_Moe(task_dict).to(device)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])


    prediction_all = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            data, labels = batch['sequence'], batch['label'].view(-1)
            data, labels = data.to(device),labels.to(device)
            outputs = model(data).squeeze()       
            logits = torch.sigmoid(outputs)
            prediction_all.append(logits.cpu().detach().numpy())
            
    with open("./test_results/%s_%s_test_result" %(Histone,model_name),"wb") as f:
        pickle.dump(prediction_all,f)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-h", "--histone", type=bool,default=False, help="histone type")
    parser.add_argument('--save_model', type=bool, default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    main(args)
