from transformers import (AdamW,T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup) 
from transformers import (get_linear_schedule_with_warmup)
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pytorch_lightning as pl
from torch.optim import AdamW
from pprint import pprint
from tqdm import tqdm
import pandas as pd
import argparse
import torch
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# if torch.cuda.is_available():
#   for i in range(torch.cuda.device_count()):
#     print(torch.cuda.get_device_name(i))
# else:
#   print("You are running on CPU")

######################################## Create dataset ##############################################3
  
pd.options.display.max_rows , pd.options.display.max_columns  = 100, 100  

# def create_pandas_dataset(data, answer_threshold=7, verbose = False):

#   # answer_threshold: Only consider those Question Answer pairs where the Answer is short(if answerable)

#   count_long, count_short, count_unanswerable = 0, 0, 0 
#   rows = []
      
#   for val in tqdm(data):
#     passage = val['context']
#     question = val['question']

#     if val['answers']['text']: # check if 'answer' is not empty and if has text
#         answer = val['answers']['text'][0]
#         no_of_words = len(answer.split())

#         if no_of_words >= answer_threshold:
#             count_long += 1
#         else:
#             rows.append({'context': passage, 'answer': answer, 'question': question, 'is_answerable': True})
#            # result_df = result_df.append({'context': passage, 'answer': answer, 'question': question, 'is_answerable': True}, ignore_index=True)
#             count_short += 1    
#     else:
#         # result_df = result_df.append({'context': passage, 'answer': '', 'question': question, 'is_answerable': False}, ignore_index=True)
#         rows.append({'context': passage, 'answer': '', 'question': question, 'is_answerable': False})
#         count_unanswerable += 1
            
#   result_df = pd.DataFrame(rows, columns=['context', 'answer', 'question', 'is_answerable']) 
#   if verbose:
#     return result_df, count_long, count_short, count_unanswerable
#   else:
#     return result_df

# train_dataset = load_dataset('squad_v2', split='train') # 87599 for v1, 130319 for v2
# validation_dataset = load_dataset('squad_v2', split='validation') # 10570 for v1, 11873 for v2
# print(f"Total Train Samples:{len(train_dataset)} , Total Validation Samples:{len(validation_dataset)}")

# # Create dataFrames
# df_train , df_validation = create_pandas_dataset(train_dataset) , create_pandas_dataset(validation_dataset)
# print(f"\n Total Train Samples:{df_train.shape} , Total Validation Samples:{df_validation.shape}")

# # Saving data in parquest files
# df_train.to_parquet('train_squad_v2.parquet')
# df_validation.to_parquet('validation_squad_v2.parquet')


################################# Create PyTorch DataSet for T5 Training and Validation
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=512)
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

class QuestionGenerationDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len_inp=512, max_len_out=96):
        self.path = file_path
        self.passage_column = "context"
        self.answer = "answer"
        self.question = "question"
        self.is_answerable = "is_answerable"

        self.data = pd.read_parquet(self.path)
        self.data = self.data[self.data[self.is_answerable] == True].reset_index(drop=True).iloc[:2000, :] # get only answerable questions

        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # squeeze to get rid of the batch dimension
        target_mask = self.targets[index]["attention_mask"].squeeze()  # convert [batch,dim] to [dim] 

        labels = copy.deepcopy(target_ids)
        labels [labels==0] = -100

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,"labels":labels}

    def _build(self):
        for _, val in tqdm(self.data.iterrows()):
            passage, answer, target = val[self.passage_column], val[self.answer], val[self.question]

            input_ = f"context: {passage}  answer: {answer}" # T5 Input format for question answering tasks 
            target = f"question: {str(target)}" # Output format we require

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len_input,padding='max_length',
                truncation = True,return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len_output,padding='max_length',
                truncation = True,
                return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
            
train_path = 'train_squad_v2.parquet' # change this accordingly
validation_path = 'validation_squad_v2.parquet'
train_dataset = QuestionGenerationDataset(t5_tokenizer, train_path)
validation_dataset = QuestionGenerationDataset(t5_tokenizer, validation_path)

# Data Sample
train_sample = train_dataset[100] # thanks to __getitem__
decoded_train_input = t5_tokenizer.decode(train_sample['source_ids'])
decoded_train_output = t5_tokenizer.decode(train_sample['target_ids'])

# print(decoded_train_input)
# print(decoded_train_output)


# # #################################### Fine Tunning T5 ##################################
class T5Tuner(pl.LightningModule):

    def __init__(self, t5model, t5tokenizer, batchsize=4):
        super().__init__()
        self.model = t5model
        self.tokenizer = t5tokenizer
        self.batch_size = batchsize

    def forward(self, input_ids, attention_mask=None, decoder_attention_mask=None, lm_labels=None):
         outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
         
         return outputs

    def training_step(self, batch):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log('train_loss',loss)
        return loss

    def validation_step(self, batch):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log("val_loss",loss)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(validation_dataset, batch_size=self.batch_size, num_workers=2)

    def configure_optimizers(self):
        # optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        # return optimizer
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,  # Adjust as per requirement
                                                num_training_steps=self.trainer.estimated_stepping_batches)  # PyTorch Lightning provides this
        scheduler_config = {
        'scheduler': scheduler,
        'interval': 'step',
        'frequency': 1
        }
        return [optimizer], [scheduler_config]
    
model = T5Tuner(t5_model, t5_tokenizer)
trainer = pl.Trainer(max_epochs=4, accelerator='cuda')
trainer.fit(model)

# saving the model
model.model.save_pretrained('t5_trained_model2')
t5_tokenizer.save_pretrained('t5_tokenizer2')


# #################### Inference/Predictions ##########################3
trained_model_path = 't5_trained_model2'
trained_tokenizer = 't5_tokenizer2'
device = 'cuda' 

model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)

model = model.to(device)

# Text sample
context ="President Donald Trump said and predicted that some states would reopen this month."
answer = "Donald Trump"
text = "context: "+context + " " + "answer: " + answer
print(text)

context ="Since its topping out in 2013, One World Trade Center in New York City has been the tallest skyscraper in the United States."
answer = "World Trade Center"
text = "context: " + context + " " + "answer: " + answer
print(text)

encoding = tokenizer.encode_plus(text,max_length =512,padding='max_length', truncation = True,return_tensors="pt")
print (encoding.keys())

input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)

model.eval()
beam_outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=72, # How long the generated questions should be
    early_stopping=True,
    num_beams=5,
    num_return_sequences=2
)

# Decoding and printing out the generated sequences
for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(sent)
