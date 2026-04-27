# import pickle
import pandas as pd
# import numpy as np
# import torch
# from torcheval.metrics import (
#     BinaryAccuracy,
#     BinaryAUROC,
#     BinaryF1Score,
#     BinaryAUPRC
# )


# # with open("./test_results/H3K4me3_LLM_Moe_test_result", "rb") as f:
# #     data = pickle.load(f)
# # data = np.load('./test_results/H3K4me3_LLM_Moe_test_result.npy')
# # labels = np.load('./test_results/H3K4me3_LLM_Moe_labels.npy')

# data = np.array(np.load('./test_results/H3K4me3_LLM_Moe_test_result.npy'), dtype=np.float32)
# labels = np.array(np.load('./test_results/H3K4me3_LLM_Moe_labels.npy'), dtype=np.float32)


# arr = np.asarray(data, dtype=np.float32).copy()
# labels = np.asarray(labels, dtype=np.float32).copy()
# print("shape", arr.shape)

# task_dict = {"task1":6, "task2":5, "task3":4, "task4":4, "task5":3}

# # Create column names
# columns = []
# for task, count in task_dict.items():
#     for i in range(count):
#         columns.append(f"{task}_label{i}")

# # Create DataFrame with probabilities
# df_probs = pd.DataFrame(arr, columns=columns)


# # Convert to binary predictions (threshold = 0.5)
# binary_preds = (arr > 0.5).astype(int)
# df_binary = pd.DataFrame(binary_preds, columns=[c + "_pred" for c in columns])

# # labels
# df_labels = pd.DataFrame(labels, columns=[c + "_true" for c in columns])


# # Combine both
# # df = pd.concat([df_probs, df_binary], axis=1)
# df = pd.concat([df_probs, df_binary, df_labels], axis=1)

# # Save
# df.to_csv("predictions.csv", index=False)


# # Debug prints
# print("labels:\n", df_labels.iloc[0])
# print("first 5 rows:\n", df.head())
# print("sample row (probs):\n", df_probs.iloc[0])
# print("sample row (binary):\n", df_binary.iloc[0])

# print("first 5 predictions", arr[:5])
# print(type(data))
# print(len(data))
# print(data[0])   # first batch predictions

# results = []
# start = 0

# for task, size in task_dict.items():
#     end = start + size

#     # p = torch.from_numpy(arr[:, start:end]).float()
#     # l = torch.from_numpy(labels[:, start:end]).float()
#     p = torch.tensor(arr[:, start:end], dtype=torch.float32)
#     l = torch.tensor(labels[:, start:end], dtype=torch.float32)

#     auc = BinaryAUROC()
#     acc = BinaryAccuracy()
#     f1 = BinaryF1Score()
#     prc = BinaryAUPRC()

#     auc.update(p, l)
#     acc.update(p, l)
#     f1.update(p, l)
#     prc.update(p, l)

#     results.append({
#         "task": task,
#         "AUC": auc.compute().item(),
#         "accuracy": acc.compute().item(),
#         "F1": f1.compute().item(),
#         "PRC": prc.compute().item()
#     })

#     start = end

# df_tasks = pd.DataFrame(results)
# df_tasks.to_csv("task_metrics.csv", index=False)

# print("\nPer-task metrics:")
# print(df_tasks)

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
# Load data
# -----------------------

# data = np.load("./test_results/H3K4me3_LLM_Moe_test_result.npy")
# labels = np.load("./test_results/H3K4me3_LLM_Moe_labels.npy")

# make it applicabel to any file name
# Histone = "H3K27ac"
# Histone = "H3K4me3"
Histone = "H3K27me3"
model_name = "LLM_Moe"
data = np.load("./test_results/%s_%s_test_result.npy" % (Histone, model_name))
labels = np.load("./test_results/%s_%s_labels.npy" % (Histone, model_name))
# data = np.load("./test_results/%s_%s_AML_test_result.npy" % (Histone, model_name))
# labels = np.load("./test_results/%s_%s_AML_labels.npy" % (Histone, model_name))

# force clean format (VERY IMPORTANT for stability)
data = np.asarray(data, dtype=np.float32)
labels = np.asarray(labels, dtype=np.float32)

# ad
patient_ids = [
    "NCI_ENCDO203ASI","NCI_ENCDO218FFZ","NCI_ENCDO250PFZ","NCI_ENCDO592ZWW","NCI_ENCDO609ZOG","NCI_ENCDO623FPG",
    "MCI_ENCDO471EKG","MCI_ENCDO672KST","MCI_ENCDO697SWU","MCI_ENCDO832DBZ","MCI_ENCDO877NVF",
    "CI_ENCDO077CCP","CI_ENCDO359XWR","CI_ENCDO448YMQ","CI_ENCDO845GYA",
    "AD_ENCDO201EUI","AD_ENCDO258GJF","AD_ENCDO853VGZ","AD_ENCDO997SGX",
    "ADCI_ENCDO754RFQ","ADCI_ENCDO776JEI","ADCI_ENCDO940VNT"
]

batch_size = 8
seq_length = 4096
# aml
# data_module = AMLDataModule("./Dataset/%s_AML_generated.csv" %Histone,["chr10"], ["chr8","chr9"],seq_length, batch_size,pretrain=True)
# patient_ids = data_module.test_data.label_cols

df = pd.DataFrame(data, columns=patient_ids)
df.to_csv("%s_predictions.csv" % (Histone), index=False)
# df.to_csv("%s_AML_predictions.csv" % (Histone), index=False)


print("shape:", data.shape)

# -----------------------
# Task definition
# -----------------------
task_dict = {
    "task1": 6,
    "task2": 5,
    "task3": 4,
    "task4": 4,
    "task5": 3
}

# -----------------------
# Per-task evaluation
# -----------------------
results = []
start = 0

for task, size in task_dict.items():
    end = start + size

    p_task = data[:, start:end]
    l_task = labels[:, start:end]

    aucs, f1s, accs, prcs = [], [], [], []

    for i in range(size):
        preds = torch.tensor(p_task[:, i], dtype=torch.float32)
        truth = torch.tensor(l_task[:, i], dtype=torch.float32)

        auc = BinaryAUROC()
        f1 = BinaryF1Score()
        acc = BinaryAccuracy()
        prc = BinaryAUPRC()

        auc.update(preds, truth)
        f1.update(preds, truth)
        acc.update(preds, truth)
        prc.update(preds, truth)

        aucs.append(auc.compute().item())
        f1s.append(f1.compute().item())
        accs.append(acc.compute().item())
        prcs.append(prc.compute().item())

    results.append({
        "task": task,
        "AUC": np.mean(aucs),
        "F1": np.mean(f1s),
        "Accuracy": np.mean(accs),
        "PRC": np.mean(prcs),
    })

    start = end

# -----------------------
# Save per-task metrics
# -----------------------
df_tasks = pd.DataFrame(results)
df_tasks.to_csv("%s_task_metrics.csv" % Histone, index=False)
# df_tasks.to_csv("%s_AML_task_metrics.csv" % Histone, index=False)


print("\nPer-task metrics:")
print(df_tasks)

# -----------------------
# Optional: global metrics
# -----------------------
all_preds = torch.tensor(data, dtype=torch.float32)
all_labels = torch.tensor(labels, dtype=torch.float32)

global_auc = BinaryAUROC()
global_f1 = BinaryF1Score()
global_acc = BinaryAccuracy()
global_prc = BinaryAUPRC()

for i in range(all_preds.shape[1]):
    global_auc.update(all_preds[:, i], all_labels[:, i])
    global_f1.update(all_preds[:, i], all_labels[:, i])
    global_acc.update(all_preds[:, i], all_labels[:, i])
    global_prc.update(all_preds[:, i], all_labels[:, i])

global_metrics = {
    "AUC": global_auc.compute().item(),
    "F1": global_f1.compute().item(),
    "Accuracy": global_acc.compute().item(),
    "PRC": global_prc.compute().item(),
}

print("\nGlobal metrics:")
print(global_metrics)

# Save global metrics too
pd.DataFrame([global_metrics]).to_csv("%s_global_metrics.csv" % Histone, index=False)
# pd.DataFrame([global_metrics]).to_csv("%s_AML_global_metrics.csv" % Histone, index=False)



# per patient
patient_ids = [
    "NCI_ENCDO203ASI","NCI_ENCDO218FFZ","NCI_ENCDO250PFZ","NCI_ENCDO592ZWW","NCI_ENCDO609ZOG","NCI_ENCDO623FPG",
    "MCI_ENCDO471EKG","MCI_ENCDO672KST","MCI_ENCDO697SWU","MCI_ENCDO832DBZ","MCI_ENCDO877NVF",
    "CI_ENCDO077CCP","CI_ENCDO359XWR","CI_ENCDO448YMQ","CI_ENCDO845GYA",
    "AD_ENCDO201EUI","AD_ENCDO258GJF","AD_ENCDO853VGZ","AD_ENCDO997SGX",
    "ADCI_ENCDO754RFQ","ADCI_ENCDO776JEI","ADCI_ENCDO940VNT"
]

# -----------------------
# Per-patient evaluation
# -----------------------
patient_results = []

for i, patient in enumerate(patient_ids):
    preds = torch.tensor(data[:, i], dtype=torch.float32)
    truth = torch.tensor(labels[:, i], dtype=torch.float32)

    auc = BinaryAUROC()
    f1 = BinaryF1Score()
    acc = BinaryAccuracy()
    prc = BinaryAUPRC()

    auc.update(preds, truth)
    f1.update(preds, truth)
    acc.update(preds, truth)
    prc.update(preds, truth)

    patient_results.append({
        "patient": patient,
        "AUC": auc.compute().item(),
        "F1": f1.compute().item(),
        "Accuracy": acc.compute().item(),
        "PRC": prc.compute().item(),
    })

# Save
df_patients = pd.DataFrame(patient_results)
df_patients.to_csv("%s_per_patient_metrics.csv" % Histone, index=False)
# df_patients.to_csv("%s_AML_per_patient_metrics.csv" % Histone, index=False)


print("\nPer-patient metrics:")
print(df_patients)