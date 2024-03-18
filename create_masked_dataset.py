from transformers import (AdamW,T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup) 
from transformers import (get_linear_schedule_with_warmup)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import pytorch_lightning as pl
from torch.optim import AdamW
from pprint import pprint
from tqdm import tqdm
from sacrebleu.metrics import bleu
from sacrebleu import corpus_bleu
from copy import deepcopy
from rouge import Rouge
import pandas as pd
import argparse
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import copy


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# if torch.cuda.is_available():
#   for i in range(torch.cuda.device_count()):
#     print(torch.cuda.get_device_name(i))
# else:
#   print("You are running on CPU")

def load_and_process_quac(dataset_df, output_file):

    dataset = dataset_df.to_dict('records')
    
    dialogues_x = []
    dialogues_y = []

    for item in dataset:
        questions = item['questions']
        answers = item['answers']
        
        mask_idx = random.randint(0, len(questions) - 1)

        # 1 = Question (userul) and 0 = Answer(serverul)
        dialogue_x = []
        for idx, (q, a) in enumerate(zip(questions, answers)):
            if idx == mask_idx:
                dialogue_x.append("1: <extra_id_0> 0: " + a)
                dialogue_y = q  # y = the masked question
            else:
                dialogue_x.append(f"1: {q} 0: {a}")

        dialogues_x.append(" ".join(dialogue_x))
        dialogues_y.append(dialogue_y)

    # (x, y) -> x=partial dialog, y=masked question
    df = pd.DataFrame({'x': dialogues_x, 'y': dialogues_y})

    #parquet_file_path = 'masked_train_quac.parquet'  
    df.to_parquet(output_file, index=False)
    
    return output_file

train_dataset = pd.read_parquet('quac_full_train.parquet')
validation_dataset = pd.read_parquet('quac_full_validation.parquet')

load_and_process_quac(train_dataset, 'masked_train_quac.parquet')
load_and_process_quac(validation_dataset, 'masked_validation_quac.parquet')

masked_train_dataset = pd.read_parquet('masked_train_quac.parquet' )
masked_validation_dataset = pd.read_parquet('masked_validation_quac.parquet')

dataset_dicts = masked_train_dataset.to_dict('records')
train_data, test_data = train_test_split(dataset_dicts, test_size=0.2, random_state=42)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df.to_parquet('masked_train.parquet', index=False)
test_df.to_parquet('masked_test.parquet', index=False)

train_dataset = pd.read_parquet('masked_train.parquet')
test_dataset = pd.read_parquet('masked_test.parquet')

print("#################Train############\n")
for i in range(3):
    print(train_dataset['x'][i] + train_dataset['y'][i] + '\n')
    
print("#################Validation############\n")
for i in range(3):
    print(masked_validation_dataset['x'][i] + masked_validation_dataset['y'][i] + '\n')
    
print("#################Test############\n")
for i in range(3):
    print(test_dataset['x'][i] + test_dataset['y'][i] + '\n')
