import numpy as np
import pandas as pd
import torch
from torcheval.metrics import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryAUPRC
)
from utils import AMLDataModule

# -----------------------
# Config
# -----------------------
Histone = "H3K4me3"
model_name = "LLM_Moe"

pred_path = f"./test_results/{Histone}_{model_name}_AML_test_result.npy"
label_path = f"./test_results/{Histone}_{model_name}_AML_labels.npy"

# -----------------------
# Load data
# -----------------------
data = np.load(pred_path).astype(np.float32)
labels = np.load(label_path).astype(np.float32)

data = np.asarray(data, dtype=np.float32)
labels = np.asarray(labels, dtype=np.float32)

print("Prediction shape:", data.shape)
print("Label shape:", labels.shape)

# -----------------------
# Save predictions (simple + safe)
# -----------------------
df_preds = pd.DataFrame(data.squeeze(), columns=["AML_probability"])
df_preds.to_csv(f"{Histone}_AML_Predictions.csv", index=False)

# -----------------------
# GLOBAL METRICS (ONLY THING NEEDED)
# -----------------------
preds = torch.tensor(data, dtype=torch.float32).squeeze()
truth = torch.tensor(labels, dtype=torch.float32).squeeze()

auc = BinaryAUROC()
# f1 = BinaryF1Score()
f1 = BinaryF1Score(threshold=0.2)
acc = BinaryAccuracy()
prc = BinaryAUPRC()

auc.update(preds, truth)
f1.update(preds, truth)
acc.update(preds, truth)
prc.update(preds, truth)

global_metrics = {
    "AUC": auc.compute().item(),
    "F1": f1.compute().item(),
    "Accuracy": acc.compute().item(),
    "PRC": prc.compute().item(),
}

print("\nGlobal AML metrics:")
print(global_metrics)

pd.DataFrame([global_metrics]).to_csv(
    f"{Histone}_AML_Global_metrics.csv",
    index=False
)

print("pred min/max:", preds.min().item(), preds.max().item())
print("label distribution:", torch.mean(truth))
print("pred mean:", torch.mean(preds))