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

t5_tokenizer = T5Tokenizer.from_pretrained('t5-large', model_max_length=512)
t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')

class DialogQuestionGenerationDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len_inp=512, max_len_out=96):
        self.tokenizer = tokenizer
        self.path = file_path

        self.data = pd.read_parquet(self.path)

        self.inputs = []
        self.targets = []
        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100

        return {
            "source_ids": source_ids, 
            "source_mask": src_mask, 
            "target_ids": target_ids, 
            "target_mask": target_mask,
            "labels": labels
        }

    def _build(self):
        for _, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            dialog, masked_question = row['x'], row['y']

# !! TO DO: sa adaug si contextul neaparat!!
            input_ = f"dialog: {dialog}"  #  dialog with <extra_id_0> 
            target = masked_question  #  masked question is the target

            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len_input, padding='max_length',
                truncation=True, return_tensors="pt"
            )

            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len_output, padding='max_length',
                truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

train_path = 'masked_train.parquet'
validation_path = 'masked_validation_quac.parquet'
test_path = 'masked_test.parquet'

train_dataset = DialogQuestionGenerationDataset(t5_tokenizer, train_path)
validation_dataset = DialogQuestionGenerationDataset(t5_tokenizer, validation_path)

class T5FineTuner(pl.LightningModule):
    def __init__(self, model_name='t5-large', tokenizer=None, learning_rate=1e-3, adam_epsilon=1e-8):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["labels"]
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]
    
train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4)
val_dataloader = DataLoader(validation_dataset, batch_size=16, num_workers=4)

model = T5FineTuner(model_name='t5-large', tokenizer=t5_tokenizer)
trainer = pl.Trainer(
    max_epochs=10,
    accelerator='cuda',  
    devices=1,  # 3 NVIDIA A40
)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
model.model.save_pretrained('fine_tuned_T5')
model.tokenizer.save_pretrained('fine_tuned_T5_tokenizer')