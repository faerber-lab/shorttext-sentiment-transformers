import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import RobertaModel, RobertaTokenizer, TrainingArguments, Trainer, RobertaConfig


import os
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np 

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix


if torch.backends.mps.is_available():
    device = torch.device('mps') # for m series mac 
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def load_and_split_dataset(dataset_path, split_ratio=0.8):
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)
    print(dataset.head())
    
    if (split_ratio == 1.0):
        training_set = dataset.reset_index(drop=True)
        return training_set , []
    
    training_set, validation_set = train_test_split(dataset, test_size=float(1.0-split_ratio), random_state=123)
    
    training_set = training_set.reset_index(drop=True)
    validation_set = validation_set.reset_index(drop=True)
    
    return training_set, validation_set

def tokenize_dataset_for_pretraining(dataset, tokenizer): 

    tokenized = [tokenizer(text) for text in dataset["text"]]
    
    return tokenized 

def tokenize_dataset(dataset, tokenizer, max_len):
    targets = torch.tensor(
        list(zip(
            dataset["Anger"],
            dataset["Fear"],
            dataset["Joy"],
            dataset["Sadness"],
            dataset["Surprise"]
        )),
        dtype=torch.float
    )

    print(dataset["text"])
    
    tokenized = [
        _prepare_data(text, label, tokenizer,max_len) for text, label in zip(dataset["text"], targets)
    ]
    
    return tokenized

def _prepare_data(text, label,tokenizer,max_len):
    tokenized = tokenizer(text, truncation=True, add_special_tokens=True,padding='max_length',
                          max_length=max_len,return_token_type_ids=True, return_tensors="pt")
    tokenized['labels'] = label
    tokenized['input_ids'] = tokenized['input_ids'].squeeze(0)
    tokenized['attention_mask'] = tokenized['attention_mask'].squeeze(0)
    tokenized['token_type_ids'] = tokenized['token_type_ids'].squeeze(0)
    return tokenized


class CustomClassifier(torch.nn.Module):
    def __init__(self,model_name,model_type, classifier_size, dropout_rate=0.3, num_classes=5):
        super(CustomClassifier, self).__init__()
        self.config = RobertaConfig.from_pretrained(model_name)
        #self.l1 = RobertaModel.from_pretrained("roberta-base")
        #self.l1 = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
        #self.l1 = AutoModelForSeq2SeqLM.from_pretrained(model_name)#"google-t5/t5-small")
        self.l1 = model_type.from_pretrained(model_name)

        self.pre_classifier = torch.nn.LazyLinear(classifier_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(classifier_size, num_classes)
        self.sigmoid = torch.nn.Sigmoid()   
        
    def forward(
        self, 
        labels=None, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        inputs_embeds=None, 
        output_attentions=None, 
        output_hidden_states=None, 
        return_dict=None
        ):
        # Pass inputs to the base model
        if inputs_embeds is not None:
            output_1 = self.l1(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:
            output_1 = self.l1(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        # Extract hidden states and pooler output
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]  # CLS token representation
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        # Compute loss if labels are provided
        if labels is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(output, labels)
            return loss, output

        return output



class T5Classifier(CustomClassifier):
    
    def forward(self, labels, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        loss = nn.BCEWithLogitsLoss()(output, labels)
        #print(f"{loss.shape=} {output.shape=}")
        return loss, output
    
class CustomTrainer(Trainer):
    
    #def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
    #    #labels = inputs.pop("labels")
    #    labels = inputs.get("labels")
    #    #print(inputs["input_ids"].shape)
    #    outputs = model(**inputs)
    #    loss = nn.BCEWithLogitsLoss()(outputs, labels)
    #    outputs  = torch.cat((torch.zeros(1,outputs.size(1),device=device), outputs), dim=0) ## Add a column of zeros to the beginning of the tensor, Necessary because of a weird implementation of the Trainer class
    #    #print(f"{outputs.shape=}")
    #    return (loss, outputs) if return_outputs else loss
    #
    
    def save_model(self, output_dir: str = None, _internal_call=False):
        """
        Override the default `save_model` method to ensure proper saving of shared memory tensors.
        """
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"{output_dir=}")
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        if self.tokenizer is not None:
            torch.save(self.tokenizer, output_dir)
    
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

def compute_metrics_f1(p): 
    predictions, labels = p
    #print(f"{predictions=} {labels=}")
    labels = torch.tensor(labels)
    predictions = torch.tensor(predictions)
    predictions = torch.sigmoid(predictions)
    predictions = torch.round(predictions)

    # can raise a warning when there are no positive labels or predictions for one 
    # emotion in the passed data 
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    labels = np.array(labels)
    predictions = np.array(predictions)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
        
        
def get_save_file_path(model_name):
    from datetime import datetime

    # Get the current date and time
    now = datetime.now()

    # Format the date and time
    formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    return f"./results/{model_name}_{formatted_date_time}"


def remove_all_files_and_folders_except_best_model(folder_path):
    import os
    import shutil
    best_model = None
    for file in os.listdir(folder_path):
        if file == "best_model.pth":
            best_model = file
        else:
            try:
                shutil.rmtree(f"{folder_path}/{file}")
            except:
                os.remove(f"{folder_path}/{file}")
    return best_model

def plot_confusion_matrix(predictions, save_path = None, file_name = None, show = True): 

    preds = predictions.predictions
    preds = torch.tensor(preds,device=device)
    preds = torch.sigmoid(preds)
    preds = torch.round(preds)
    ids = torch.tensor(predictions.label_ids,device=device)

    label_map = {
    'LABEL_0': 'Anger',
    'LABEL_1': 'Fear',
    'LABEL_2': 'Joy',
    'LABEL_3': 'Sadness',
    'LABEL_4': 'Surprise'
    }

    cm = multilabel_confusion_matrix(ids, preds)

    # label_map to labels
    labels = [label_map[f'LABEL_{i}'] for i in range(len(label_map))]

    # Confusion Matrix
    plt.figure(figsize=(20, 10))
    
    for i, label in enumerate(labels): 
        plt.subplot(2, 3, i + 1)
        sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues', xticklabels=["False", "True"], yticklabels=["False", "True"])
        plt.xlabel('Predicted value')
        plt.ylabel('True value')
        plt.title(f"Confusion Matrix for '{label}' class")


    if save_path is not None and file_name is not None: 
        plt.savefig(f"{save_path}/{file_name}.png", bbox_inches='tight')

    if show: 
        plt.show()
    