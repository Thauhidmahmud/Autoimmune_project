
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score, f1_score)
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from sklearn.manifold import TSNE


# ==========================================
# 0. REPRODUCIBILITY (Fixing Random Seeds)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU support
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# DataLoader-এর Shuffling নিয়ন্ত্রণে PyTorch Generator
g = torch.Generator()
g.manual_seed(42)


# ==========================================
# 1. Data Loading & Cleaning
# ==========================================
df = pd.read_csv("normalized.csv", encoding='utf-8-sig')
df.columns = df.columns.str.strip()

if 'Disease_name' not in df.columns:
    df.rename(columns={df.columns[0]: 'Disease_name'}, inplace=True)

counts = df['Disease_name'].value_counts()
df = df[df['Disease_name'].isin(counts[counts >= 5].index)].reset_index(drop=True)

# Label encoding & Base Features creation
le = LabelEncoder()
df["disease_id"] = le.fit_transform(df["Disease_name"])
num_diseases = len(le.classes_)

# ----------------------------------------------------
# Feature Formations for Feature Ablation Study
# ----------------------------------------------------
df["text_full"] = df["miRNA"].astype(str) + " [SEP] " + df["sequence"].astype(str) + " [SEP] " + df["genes"].astype(str)
df["text_mirna_only"] = df["miRNA"].astype(str)
df["text_no_genes"] = df["miRNA"].astype(str) + " [SEP] " + df["sequence"].astype(str)
df["text_no_sequence"] = df["miRNA"].astype(str) + " [SEP] " + df["genes"].astype(str)


# ==========================================
# 2. STRATEGIC SPLIT (Preventing Data Leakage)
# ==========================================
train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["disease_id"])
train_df_raw, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df["disease_id"])


# ==========================================
# 3. OVERSAMPLING (Only on Train Data)
# ==========================================
max_size = train_df_raw['Disease_name'].value_counts().max()
lst = [train_df_raw[train_df_raw['Disease_name'] == i] for i in train_df_raw['Disease_name'].unique()]
train_df_balanced = pd.concat([resample(l, replace=True, n_samples=max_size, random_state=42) for l in lst])

train_df = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced Train set: {len(train_df)} | Raw Unbalanced Train: {len(train_df_raw)} | Val set: {len(val_df)} | Test set (Unseen): {len(test_df)}")


# ==========================================
# 4. Tokenization & Dataset Pipeline Setup
# ==========================================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_data(dataframe, text_col="text_full"):
    encodings = tokenizer(dataframe[text_col].tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')
    return encodings["input_ids"], encodings["attention_mask"], \
           torch.tensor(dataframe["label"].values, dtype=torch.float), \
           torch.tensor(dataframe["disease_id"].values, dtype=torch.long)

class MiRNADataset(Dataset):
    def __init__(self, ids, masks, bin_labels, dis_labels):
        self.ids, self.masks, self.bin_labels, self.dis_labels = ids, masks, bin_labels, dis_labels
    def __len__(self): return len(self.bin_labels)
    def __getitem__(self, idx):
        return {"input_ids": self.ids[idx], "attention_mask": self.masks[idx],
                "binary": self.bin_labels[idx], "disease": self.dis_labels[idx]}


# ==========================================
# 5. Model Architecture
# ==========================================
class AutoimmuneTransMiR(nn.Module):
    def __init__(self, num_disease):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(0.3)
        self.out_bin = nn.Linear(768, 1)
        self.out_dis = nn.Linear(768, num_disease)
    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask).pooler_output
        hidden_feats = out
        out = self.drop(out)
        return self.out_bin(out), self.out_dis(out), hidden_feats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 6. Universal Training & Evaluation Function
# ==========================================
def train_and_eval_model(mode="full", text_col="text_full", use_raw_train=False, epochs=7):
    set_seed(42)  # Reset seed for fair comparison

    # Selecting Dataset Based on Feature Ablation & Raw/Balanced Training
    target_train_df = train_df_raw if use_raw_train else train_df

    tr_ids, tr_masks, tr_bin, tr_dis = encode_data(target_train_df, text_col=text_col)
    v_ids, v_masks, v_bin, v_dis = encode_data(val_df, text_col=text_col)
    t_ids, t_masks, t_bin, t_dis = encode_data(test_df, text_col=text_col)

    current_train_loader = DataLoader(MiRNADataset(tr_ids, tr_masks, tr_bin, tr_dis), batch_size=16, shuffle=True, generator=g)
    current_val_loader = DataLoader(MiRNADataset(v_ids, v_masks, v_bin, v_dis), batch_size=16)
    current_test_loader = DataLoader(MiRNADataset(t_ids, t_masks, t_bin, t_dis), batch_size=16)

    model = AutoimmuneTransMiR(num_diseases).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(current_train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion_bin = nn.BCEWithLogitsLoss()
    criterion_dis = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in current_train_loader:
            optimizer.zero_grad()
            b_out, d_out, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))

            # Loss Calculation based on Mode
            if mode == "binary_only":
                loss = criterion_bin(b_out.squeeze(), batch["binary"].to(device))
            elif mode == "disease_only":
                loss = criterion_dis(d_out, batch["disease"].to(device))
            else:  # Multi-task
                loss = criterion_bin(b_out.squeeze(), batch["binary"].to(device)) + \
                       criterion_dis(d_out, batch["disease"].to(device))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(current_train_loader)
        train_losses.append(avg_train_loss)

        # Validation Step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in current_val_loader:
                b_out, d_out, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                if mode == "binary_only":
                    v_loss = criterion_bin(b_out.squeeze(), batch["binary"].to(device))
                elif mode == "disease_only":
                    v_loss = criterion_dis(d_out, batch["disease"].to(device))
                else:
                    v_loss = criterion_bin(b_out.squeeze(), batch["binary"].to(device)) + \
                             criterion_dis(d_out, batch["disease"].to(device))
                total_val_loss += v_loss.item()

        avg_val_loss = total_val_loss / len(current_val_loader)
        val_losses.append(avg_val_loss)

    # Testing Evaluation
    model.eval()
    bin_true, bin_preds, dis_true, dis_preds = [], [], [], []
    all_embeddings, bin_probs = [], []

    with torch.no_grad():
        for batch in current_test_loader:
            b_out, d_out, feats = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))

            prob_bin = torch.sigmoid(b_out).cpu().numpy()
            bin_probs.extend(prob_bin)
            bin_preds.extend((prob_bin > 0.5).astype(int))
            bin_true.extend(batch["binary"].numpy())

            dis_preds.extend(torch.argmax(d_out, dim=1).cpu().numpy())
            dis_true.extend(batch["disease"].numpy())
            all_embeddings.append(feats.cpu().numpy())

    bin_acc = accuracy_score(bin_true, bin_preds) if mode != "disease_only" else 0.0
    dis_acc = accuracy_score(dis_true, dis_preds) if mode != "binary_only" else 0.0
    dis_f1 = f1_score(dis_true, dis_preds, average='macro') if mode != "binary_only" else 0.0

    return {
        "model": model,
        "bin_acc": bin_acc,
        "dis_acc": dis_acc,
        "dis_f1": dis_f1,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "bin_true": bin_true,
        "bin_preds": bin_preds,
        "bin_probs": bin_probs,
        "dis_true": dis_true,
        "dis_preds": dis_preds,
        "all_embeddings": all_embeddings
    }


# ==========================================
# 7. RUNNING FULL ABLATION SUITE
# ==========================================
print("\n" + "="*50)
print("EXECUTING COMPREHENSIVE ABLATION EXPERIMENTS")
print("="*50)

# Proposed Base Full Model
print("\n[1/6] Proposed Model (Full Features + Multi-Task + Oversampling)...")
full_res = train_and_eval_model(mode="full", text_col="text_full")

# Architecture & Strategy Ablation
print("[2/6] Ablation (Task): Single-Task (Binary Only)...")
bin_res = train_and_eval_model(mode="binary_only", text_col="text_full")

print("[3/6] Ablation (Task): Single-Task (Disease Only)...")
dis_res = train_and_eval_model(mode="disease_only", text_col="text_full")

print("[4/6] Ablation (Sampling): Without Oversampling...")
no_over_res = train_and_eval_model(mode="full", text_col="text_full", use_raw_train=True)

# Feature Fusion Ablation
print("[5/6] Ablation (Feature): Only miRNA Name (No Sequence & Genes)...")
f_mirna_res = train_and_eval_model(mode="full", text_col="text_mirna_only")

print("[6/6] Ablation (Feature): miRNA + Sequence (Without Genes)...")
f_nogene_res = train_and_eval_model(mode="full", text_col="text_no_genes")


# ==========================================
# 8. RESEARCH PAPER ABLATION TABLE
# ==========================================
ablation_data = {
    "Experiment Category": [
        "Proposed Pipeline",
        "Task Architecture",
        "Task Architecture",
        "Data Balancing",
        "Feature Input Fusion",
        "Feature Input Fusion"
    ],
    "Model Variant / Setup": [
        "Full Proposed Model",
        "Single-Task (Binary Only)",
        "Single-Task (Disease Only)",
        "No Oversampling (Imbalanced)",
        "Only miRNA (No Seq & Genes)",
        "miRNA + Sequence (No Genes)"
    ],
    "Binary Acc": [
        f"{full_res['bin_acc']:.4f}",
        f"{bin_res['bin_acc']:.4f}",
        "N/A",
        f"{no_over_res['bin_acc']:.4f}",
        f"{f_mirna_res['bin_acc']:.4f}",
        f"{f_nogene_res['bin_acc']:.4f}"
    ],
    "Disease Acc": [
        f"{full_res['dis_acc']:.4f}",
        "N/A",
        f"{dis_res['dis_acc']:.4f}",
        f"{no_over_res['dis_acc']:.4f}",
        f"{f_mirna_res['dis_acc']:.4f}",
        f"{f_nogene_res['dis_acc']:.4f}"
    ],
    "Disease Macro F1": [
        f"{full_res['dis_f1']:.4f}",
        "N/A",
        f"{dis_res['dis_f1']:.4f}",
        f"{no_over_res['dis_f1']:.4f}",
        f"{f_mirna_res['dis_f1']:.4f}",
        f"{f_nogene_res['dis_f1']:.4f}"
    ]
}

ablation_df = pd.DataFrame(ablation_data)

print("\n" + "="*80)
print(" COMPREHENSIVE ABLATION STUDY RESULTS (FOR PAPER) ")
print("="*80)
print(ablation_df.to_string(index=False))
