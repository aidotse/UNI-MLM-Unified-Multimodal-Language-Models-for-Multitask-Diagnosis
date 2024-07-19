import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn as nn
import torch.optim as optim
from focal_loss.focal_loss import FocalLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler, RandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils_asym import *



# Filepath to embeddings
fname = None



"""
checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
# checkpoint_path = "microsoft/Phi-3-mini-128k-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
phi3 = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

phi3.eval()
"""


# GEMMA-2B init
quantization_config = BitsAndBytesConfig(load_in_4bit=True, 
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", quantization_config=quantization_config)
gemma.eval()


# Read data & extract labels and features
df = pd.read_csv(fname)


# Load train/val sets and create data loaders
batch_size = 32

Data = DataSplit(df)
Data.split_data('all')
X, V = Data.get_data()


Data.y_train = Data.y_train.apply(lambda lst: [2 if x == -1 else x for x in lst])
Data.y_val = Data.y_val.apply(lambda lst: [2 if x == -1 else x for x in lst])

train_set = CustomDataset(X.values.tolist(), Data.y_train.tolist())
val_set = CustomDataset(V.values.tolist(), Data.y_val.tolist())

transposed_Y = list(map(list, zip(*Data.y_train.tolist())))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=5)


folder = "projectors"
# Setting model and hyperparameters
vd_model = torch.load(f'{folder}/vd.pth').to('cuda')
vmd_model = torch.load(f'{folder}/vmd.pth').to('cuda')

ts_pe_model = torch.load(f'{folder}/ts_pe.pth').to('cuda')
ts_ce_model = torch.load(f'{folder}/ts_ce.pth').to('cuda')
ts_le_model = torch.load(f'{folder}/ts_le.pth').to('cuda')

n_rad_model = torch.load(f'{folder}/n_rad.pth').to('cuda')
models = [vd_model, vmd_model, ts_pe_model, ts_ce_model, ts_le_model, n_rad_model]


def output_to_label(logits, labels):

    probs_tensor_pos = F.sigmoid(logits)

    pred = torch.round(probs_tensor_pos)

    probs_tensor_neg = 1-probs_tensor_pos

    prob = torch.stack((probs_tensor_neg, probs_tensor_pos), dim=1)


    return prob, pred, labels


def predict(models, val_loader, llm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_class_probs = [[] for _ in range(12)]
    per_class_preds = [[] for _ in range(12)]
    per_class_labels = [[] for _ in range(12)]

    for model in models:
        model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x, y.to(device)

            vd_inputs = x['vd'].to(device)
            vmd_inputs = x['vmd'].to(device)
            ts_pe_inputs = x['ts_pe'].to(device)
            ts_ce_inputs = x['ts_ce'].to(device)
            ts_le_inputs = x['ts_le'].to(device)
            n_rad_inputs = x['n_rad'].to(device)

            encoded_vd = models[0].encoder(vd_inputs)
            encoded_vmd = models[1].encoder(vmd_inputs)
            encoded_ts_pe = models[2].encoder(ts_pe_inputs)
            encoded_ts_ce = models[3].encoder(ts_ce_inputs)
            encoded_ts_le = models[4].encoder(ts_le_inputs)
            encoded_n_rad = models[5].encoder(n_rad_inputs)

            decoded_vd = models[0].decoder(encoded_vd)
            decoded_vmd = models[1].decoder(encoded_vmd)
            decoded_ts_pe = models[2].decoder(encoded_ts_pe)
            decoded_ts_ce = models[3].decoder(encoded_ts_ce)
            decoded_ts_le = models[4].decoder(encoded_ts_le)
            decoded_n_rad = models[5].decoder(encoded_n_rad)

            inputs = [vd_inputs, vmd_inputs, ts_pe_inputs, ts_ce_inputs, ts_le_inputs, n_rad_inputs]
            decoded = [decoded_vd, decoded_vmd, decoded_ts_pe, decoded_ts_ce, decoded_ts_le, decoded_n_rad]

            concat_emb = torch.cat((encoded_vd.view(-1,1,2048).to(torch.float16), encoded_vmd.view(-1,1,2048).to(torch.float16), 
                                encoded_ts_pe.view(-1,1,2048).to(torch.float16), encoded_ts_ce.view(-1,1,2048).to(torch.float16), 
                                encoded_ts_le.view(-1,1,2048).to(torch.float16), encoded_n_rad.view(-1,1,2048).to(torch.float16)),
                                  dim=1).to(device)

            logits = logit_extractor(concat_emb, llm)

            probabilities, hard_preds, labels = output_to_label(logits, labels)


            for i in range(probabilities.size(2)):
                class_prob = probabilities[:, :, i]  # Select probabilities for class i
                per_class_probs[i].append(probabilities[:, :, i])
                per_class_preds[i].append(hard_preds[:, i])
                per_class_labels[i].append(labels[:, i])


    return per_class_preds, per_class_labels, per_class_probs


per_class_preds, per_class_labels, per_class_probs = predict(models, val_loader, gemma)

f1_scores = []
auc_scores = []
precisions = []
recalls = []
accuracies = []
matrices = []
fpr_tpr = []
roc_aucs = []

for i,class_preds in enumerate(per_class_preds):
    preds = []
    preds = [t.cpu().numpy() for t in class_preds]
    preds = np.concatenate(preds)
    labels = []
    labels = [t.cpu().numpy() for t in per_class_labels[i]]
    labels = np.concatenate(labels)
    stuff = len(per_class_probs[i][0])
    probs = []
    probs = [t.cpu().numpy() for t in per_class_probs[i]]
    probs = np.concatenate(probs, axis=0)
    mask = ~np.isnan(labels) & (labels != 2)
    masked_labels = labels[mask]
    masked_preds = preds[mask]
    masked_probs = probs[mask]
    positive_probs = masked_probs[:, 1]

    f1_scores.append(metrics.f1_score(masked_labels, masked_preds, average='macro'))
    auc_scores.append(metrics.roc_auc_score(masked_labels, positive_probs))
    precisions.append(metrics.precision_score(masked_labels, masked_preds))
    recalls.append(metrics.recall_score(masked_labels, masked_preds))
    accuracies.append(metrics.accuracy_score(masked_labels, masked_preds))
    matrices.append(metrics.confusion_matrix(masked_labels, masked_preds))
    fpr, tpr, thresholds = metrics.roc_curve(masked_labels, positive_probs)
    fpr_tpr.append((fpr,tpr))
    roc_aucs.append(metrics.auc(fpr, tpr))


data_to_save = {
    'f1_scores': f1_scores,
    'auc_scores': auc_scores,
    'precisions': precisions,
    'recalls': recalls,
    'accuracies': accuracies,
    'matrices': matrices,
    'fpr_tpr': fpr_tpr,
    'roc_aucs': roc_aucs
}

with open('results.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)
