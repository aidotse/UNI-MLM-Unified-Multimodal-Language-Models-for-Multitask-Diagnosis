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
from utils_iso import *

# Filepath to embeddings
fname = None


# GEMMA-2B init
quantization_config = BitsAndBytesConfig(load_in_4bit=True, 
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", quantization_config=quantization_config)
gemma.eval()


"""
#PHI-3 init
checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
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



# Read data & extract labels and features
df = pd.read_csv(fname)
Data = DataSplit(df)
Data.split_data('all')
X, V = Data.get_data()



# Load train/val sets and create data loaders
batch_size = 32

Data.y_train = Data.y_train.apply(lambda lst: [2 if x == -1 else x for x in lst])
Data.y_val = Data.y_val.apply(lambda lst: [2 if x == -1 else x for x in lst])

train_set = CustomDataset(X.values.tolist(), Data.y_train.tolist())
val_set = CustomDataset(V.values.tolist(), Data.y_val.tolist())

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=5)


# Setting model and hyperparameters
torch.manual_seed(42)

vd_model = AutoEncoder(1024,2048)
vmd_model = AutoEncoder(1024,2048)

ts_pe_model = AutoEncoder(110,2048)
ts_ce_model = AutoEncoder(99,2048)
ts_le_model = AutoEncoder(242,2048)

n_rad_model = AutoEncoder(768,2048)
projectors = [vd_model, vmd_model, ts_pe_model, ts_ce_model, ts_le_model, n_rad_model]


vd_optimizer = optim.Adam(vd_model.parameters(), lr=1e-4, weight_decay=1e-5)
vmd_optimizer = optim.Adam(vmd_model.parameters(), lr=1e-4, weight_decay=1e-5)

ts_pe_optimizer = optim.Adam(ts_pe_model.parameters(), lr=1e-4, weight_decay=1e-5)
ts_ce_optimizer = optim.Adam(ts_ce_model.parameters(), lr=1e-4, weight_decay=1e-5)
ts_le_optimizer = optim.Adam(ts_le_model.parameters(), lr=1e-4, weight_decay=1e-5)

n_rad_optimizer = optim.Adam(n_rad_model.parameters(), lr=1e-4, weight_decay=1e-5)
optimizers = [vd_optimizer, vmd_optimizer, ts_pe_optimizer, ts_ce_optimizer, ts_le_optimizer, n_rad_optimizer]

mse_loss = nn.MSELoss()

num_epochs = 50
beta = 0.1

#Run training
training_loop(projectors, optimizers, mse_loss, train_loader, val_loader, num_epochs, gemma, beta)
