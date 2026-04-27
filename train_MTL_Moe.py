import wandb
import torch
import os
from torch.optim import  AdamW
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForMaskedLM
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAUPRC



        
def main(args):


    # histone = "H3K27me3"
    # histone = "H3K27ac"
    histone = "H3K4me3"
    model_name = "LLM_Moe"
    # model_name = "CNN_Moe"
    print("load parameters")
    model_path = "./models/%s_%s.pt" %(histone,model_name)
    early_stop = 0
    max_early_stop = 5
    batch_size = 8 # 8 
    best_loss = 100
    seq_length = 4096
    # seq_length = 5120
    task_dict = {"task1":6,"task2":5,"task3":4,"task4":4,"task5":3}


    print("load datasets...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "CNN_Moe":
        data_module = ADDataModule("./Dataset/%s_all_data.csv" %histone,["chr10"], ["chr8","chr9"],seq_length, batch_size,pretrain=False)
    else:
        data_module = ADDataModule("./Dataset/%s_all_data.csv" %histone,["chr10"], ["chr8","chr9"],seq_length, batch_size,pretrain=True)
    train_loader = data_module.train_dataloader()
    vali_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()    


    # model = Sei(n_genomic_features = 22).to(device)
    # model = PretrainModel().to(device)
    # model = Pretrain_BLSTM().to(device)
    # model = NetDeepHistone().to(device)
    if model_name == "CNN_Moe":
        model = CNN_Moe(task_dict).to(device)
    else:
        model = Pretrain_Moe(task_dict).to(device)
    AWL = AutomaticWeightedLoss(5)
    AWL.to(device)
    # param_groups = param_groups_lrd(model,AWL=AWL)
    optim = AdamW([{'params': model.parameters()},
                {'params': AWL.parameters()}], lr=5e-5)

    if args.reload:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    if args.wandb_report:
        if args.reload:
            wandb.init(project="AD", resume="must", id=args.wandbId)
            print("wandb resumed")
        else:
            os.environ["WANDB_PROJECT"] = "AD"
            print("init wandb")
            wandb.init(
                project='AD',
                config={
                    "architecture": model_name,
                    "saved_model":model_path,
                    "batch_size":batch_size,
                    "structure": str(model)
                }
            )

    def evaluate_model(model, data_loader, device):
        loss_all = {"loss_sum":0,"loss_count":0}
        metries = multiperformance()
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                data, labels = batch['sequence'],batch['label']
                data, labels = data.to(device),labels.to(device)
                outputs, aux_loss = model(data)
                outputs = torch.concat([v for k,v in outputs.items()], dim =1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)
                loss_all["loss_sum"]+=loss.item()
                loss_all["loss_count"]+=1  
                logits = torch.sigmoid(outputs)
                # print(outputs.shape,labels.shape)
                metries.update(logits.view(-1),labels.view(-1))
                # break


        return loss_all["loss_sum"]/loss_all["loss_count"], metries.compute()


    print("training....")

    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        AWL.train(True)
        loss_all = {"loss_sum":0,"loss_count":0}
        print("epoch"+str(epoch))
        for batch in train_loader:
            optim.zero_grad()
            data, labels = batch['sequence'].to(device),batch['label'].to(device)
            start_index = 0
            task_label = []
            loss_list = []
            outputs, aux_loss = model(data)
            for index ,(task_name, task_num)in enumerate(task_dict.items()):        
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs[task_name], labels[:,start_index:start_index+task_num])
                start_index+=task_num
                loss_list.append(loss)

            sum_loss = AWL(loss_list)
            sum_loss = sum_loss+0.5*aux_loss
            sum_loss.backward()
            optim.step()
        
            global_step+=1
            loss_all["loss_sum"]+=sum_loss.item()
            loss_all["loss_count"]+=1 

            # global_step = 30000
            if global_step%100 == 0:
                mean_loss = loss_all["loss_sum"]/loss_all["loss_count"]
                if args.wandb_report:
                    wandb.log({"loss":mean_loss})   
                    
            if global_step%30000 == 0:
            # if global_step%30 == 0:
                model_path_last = "./models/%s_%s_last.pt" %(histone,model_name)
                if args.save_model:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict()
                    },model_path_last)

                vali_loss, metrics_vali = evaluate_model(model, vali_loader, device)

                if torch.distributed.get_rank() == 0:
                    print(vali_loss)
                    print(metrics_vali)

                    if args.wandb_report:
                        wandb.log({"vali_loss": vali_loss})
                        wandb.log(metrics_vali)
        
                if best_loss> vali_loss:
                    best_loss=vali_loss
                    early_stop = 0
                    if args.save_model:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optim.state_dict()
                        },
                            model_path)
                        
                    test_loss, metrics_test = evaluate_model(model, test_loader, device)
                    test_result_dict = {k+"_test": v.item() for k,v in metrics_test.items()}
                    print(test_result_dict)
        
                    if args.wandb_report:
                        wandb.log({"test_loss": test_loss})
                        wandb.log(test_result_dict)
        
                else:
                    early_stop+=1
                model.train()

# trainer.train()
    if args.wandb_report:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-w", "--wandb_report", type=bool,default=False, help="wandb or not")
    parser.add_argument("-r", "--reload", type=bool,default=False, help="reload or not")
    parser.add_argument("-d", "--wandbId", type=str, help="wandb or not")
    parser.add_argument('--freeze_layer', type=bool, default=False,
                        help='freeze_layer or not')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_model', type=bool, default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    main(args)
