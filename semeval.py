# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import transformers
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, TrainingArguments, Trainer
import logging
logging.basicConfig(level=logging.ERROR)
from torch import nn

# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


# Parameters

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
# EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)



# Data Preprocessing

train = pd.read_csv('data/public_data/train/track_a/eng.csv')

train = train.dropna()
print(train.head())
train = train.reset_index(drop=True)

training_set, validation_set = train_test_split(train, test_size=0.1, random_state=123)
training_set = training_set.reset_index(drop=True)
validation_set = validation_set.reset_index(drop=True)

train_targets = torch.tensor(
    list(zip(
        training_set["Anger"],
        training_set["Fear"],
        training_set["Joy"],
        training_set["Sadness"],
        training_set["Surprise"]
    )),
    dtype=torch.float
)

val_targets = torch.tensor(
    list(zip(
        validation_set["Anger"],
        validation_set["Fear"],
        validation_set["Joy"],
        validation_set["Sadness"],
        validation_set["Surprise"]
    )),
    dtype=torch.float
)


def prepare_data(text, label):
    tokenized = tokenizer(text, truncation=True, add_special_tokens=True,padding='max_length', max_length=MAX_LEN,return_token_type_ids=True, return_tensors="pt")
    tokenized['labels'] = label
    tokenized['input_ids'] = tokenized['input_ids'].squeeze(0)
    tokenized['attention_mask'] = tokenized['attention_mask'].squeeze(0)
    tokenized['token_type_ids'] = tokenized['token_type_ids'].squeeze(0)
    return tokenized

tokenized_train = [
    prepare_data(text, label) for text, label in zip(training_set["text"], train_targets)
]

tokenized_val = [
    prepare_data(text, label) for text, label in zip(validation_set["text"], val_targets)
]



data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)


# Model

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 5)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.GELU()(pooler)
        pooler = self.dropout(pooler)
        classifier = self.classifier(pooler)
        return classifier
    
model:RobertaClass = RobertaClass()
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy='steps',  # Evaluate at the end of each epoch
    eval_steps=100,
    eval_on_start=True,
    logging_steps=10,
    label_names=["labels"],
    dataloader_drop_last=True,
 ## ---
    report_to="tensorboard",
    
## best model
    metric_for_best_model="accuracy",
    greater_is_better=True,
    load_best_model_at_end=True
 )

def compute_metrics(p):
        pred, labels = p
        pred = torch.tensor(pred)
        labels = torch.tensor(labels)
        pred = torch.sigmoid(pred)
        pred = torch.round(pred)
        accuracy = ((pred == labels).sum(axis=1) == 5).sum() / len(labels)
        #print(f"{accuracy=}")
        return {
             'accuracy': accuracy
        }

def loss_fn(y_pred, y_true):
   print(y_pred, y_true,sep=" | ")
   y_pred = torch.sigmoid(y_pred)
   y_pred = torch.round(y_pred)
   loss = nn.BCEWithLogitsLoss()  # Define the loss function
   result = loss(y_pred, y_true)  # Compute the loss
   print(result)  # Use `.item()` to convert the tensor scalar to a Python float for printing
   return result


class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        labels = inputs.pop("labels")
        #print(inputs["input_ids"].shape)
        outputs = model(**inputs)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        outputs  = torch.cat((torch.zeros(1,outputs.size(1),device=device), outputs), dim=0) ## Add a column of zeros to the beginning of the tensor, Necessary because of a weird implementation of the Trainer class
        #print(f"{outputs.shape=}")
        return (loss, outputs) if return_outputs else loss
    
      

trainer = CustomTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_val,
   data_collator=data_collator,
   compute_metrics=compute_metrics, 
)



trainer.train()
results = trainer.evaluate(eval_dataset=tokenized_val)
print(results)

torch.save(model, 'model.pth')



