"""
Train CNN_BLSTM on the original 22-patient AD dataset.

Architecture comparison
              | CNN_Moe / LLM_Moe          | CNN_BLSTM
  Encoder     | one-hot / Caduceus LLM     | one-hot (same CNN backbone)
  Aggregator  | MoE transformer (5-task)   | BiLSTM → mean pool
  Head        | 5 task-specific linears    | single linear (22 outputs)
  Loss        | per-task BCE + AWL + aux   | single BCE over all 22 outputs
  Optimizer   | AdamW 5e-5                 | AdamW 5e-5


Usage
─────
  python train_blstm.py -t H3K4me3 --save_model
  python train_blstm.py -t H3K27ac --save_model --wandb_report
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import torch
import torch.nn.functional as F
import argparse
from transformers import AdamW

from models import CNN_BLSTM
from utils.utils import multiperformance, ADDataModule

# EDIT FOR SCC
# DATA_ROOT : absolute path to the folder containing {histone}_all_data.csv
DATA_ROOT = "/projectnb/ds596/projects/Team 5/Datasets"

# MODELS_DIR  : where to save .pt checkpoints
# RESULTS_DIR : where to save prediction pickles (consumed by plot_results.py)
MODELS_DIR  = "/projectnb/ds596/projects/Team 5/code_testrun_lsj/AML_benchmarking/models"
RESULTS_DIR = "/projectnb/ds596/projects/Team 5/code_testrun_lsj/AML_benchmarking/results"
# END EDIT

TASK_DICT = {"task1": 6, "task2": 5, "task3": 4, "task4": 4, "task5": 3}
NUM_TASKS  = sum(TASK_DICT.values())   # 22

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def evaluate(model, loader, device, save_preds=False):
    model.eval()
    total_loss, count = 0.0, 0
    metrics = multiperformance()
    preds_all = []

    with torch.no_grad():
        for batch in loader:
            seq, labels = batch["sequence"].to(device), batch["label"].to(device)
            logits = model(seq)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()
            count += 1
            probs = torch.sigmoid(logits)
            metrics.update(probs.view(-1), labels.view(-1))
            if save_preds:
                preds_all.append(probs.cpu().detach().numpy())

    if save_preds:
        return total_loss / max(count, 1), metrics.compute(), preds_all
    return total_loss / max(count, 1), metrics.compute()


def main(args):
    histone    = args.histone
    model_name = "CNN_BLSTM"
    batch_size = args.batch_size
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path  = os.path.join(DATA_ROOT, f"{histone}_all_data.csv")
    model_path = os.path.join(MODELS_DIR, f"{histone}_{model_name}.pt")
    print(f"Data   : {data_path}")
    print(f"Model  : {model_path}")

    data = ADDataModule(
        data_dir=data_path,
        vali_set=["chr10"],
        test_set=["chr8", "chr9"],
        seq_length=4096,
        batch_size=batch_size,
        pretrain=False,
    )
    train_loader = data.train_dataloader()
    vali_loader  = data.val_dataloader()
    test_loader  = data.test_dataloader()

    model = CNN_BLSTM(num_tasks=NUM_TASKS).to(device)
    optim = AdamW(model.parameters(), lr=5e-5)

    if args.reload and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from {model_path}")

    if args.wandb_report and WANDB_AVAILABLE:
        os.environ["WANDB_PROJECT"] = "AD"
        wandb.init(project="AD", config={
            "architecture": model_name,
            "histone":      histone,
            "num_tasks":    NUM_TASKS,
            "batch_size":   batch_size,
        })

    best_loss  = float("inf")
    early_stop = 0

    print("Training ...")
    for epoch in range(args.epochs):
        model.train()
        print(f"Epoch {epoch}")

        for batch in train_loader:
            optim.zero_grad()
            seq, labels = batch["sequence"].to(device), batch["label"].to(device)

            logits = model(seq)
            loss   = F.binary_cross_entropy_with_logits(logits, labels)

            loss.backward()
            optim.step()

        # Validate at end of every epoch
        vali_loss, vali_metrics = evaluate(model, vali_loader, device)
        print(f"  epoch {epoch}  vali_loss={vali_loss:.4f}  {vali_metrics}")

        if args.wandb_report and WANDB_AVAILABLE:
            wandb.log({"epoch": epoch, "vali_loss": vali_loss, **vali_metrics})

        if vali_loss < best_loss:
            best_loss  = vali_loss
            early_stop = 0
            if args.save_model:
                os.makedirs(MODELS_DIR, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                }, model_path)

            test_loss, test_metrics, test_preds = evaluate(
                model, test_loader, device, save_preds=True)
            print(f"  [best] test_loss={test_loss:.4f}  {test_metrics}")

            os.makedirs(RESULTS_DIR, exist_ok=True)
            pred_path = os.path.join(RESULTS_DIR, f"{histone}_{model_name}_test_result")
            with open(pred_path, "wb") as f:
                pickle.dump(test_preds, f)
            print(f"  Predictions saved: {pred_path}")

            if args.wandb_report and WANDB_AVAILABLE:
                wandb.log({"test_loss": test_loss,
                           **{k + "_test": v for k, v in test_metrics.items()}})
        else:
            early_stop += 1
            if early_stop >= args.patience:
                print("Early stopping.")
                break

    if args.wandb_report and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--histone", choices=["H3K27ac", "H3K4me3", "H3K27me3"],
                        default="H3K4me3")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--patience",   type=int, default=5)
    parser.add_argument("--wandb_report", action="store_true")
    parser.add_argument("--reload",     action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--seed",       type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
