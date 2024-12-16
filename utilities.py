import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import RobertaModel, RobertaTokenizer, TrainingArguments, Trainer

import os



from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


def load_and_split_dataset(dataset_path, split_ratio=0.8):
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)
    print(dataset.head())
    
    training_set, validation_set = train_test_split(dataset, test_size=1-split_ratio, random_state=123)
    
    training_set = training_set.reset_index(drop=True)
    validation_set = validation_set.reset_index(drop=True)
    
    return training_set, validation_set

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
        #self.l1 = RobertaModel.from_pretrained("roberta-base")
        #self.l1 = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
        #self.l1 = AutoModelForSeq2SeqLM.from_pretrained(model_name)#"google-t5/t5-small")
        self.l1 = model_type.from_pretrained(model_name)
        self.pre_classifier = torch.nn.LazyLinear(classifier_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(classifier_size, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        #output = self.sigmoid(output)
        return output
    
class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        labels = inputs.pop("labels")
        #print(inputs["input_ids"].shape)
        outputs = model(**inputs)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        outputs  = torch.cat((torch.zeros(1,outputs.size(1),device=device), outputs), dim=0) ## Add a column of zeros to the beginning of the tensor, Necessary because of a weird implementation of the Trainer class
        #print(f"{outputs.shape=}")
        return (loss, outputs) if return_outputs else loss
    
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
        #torch.save(self.args, os.path.join(output_dir, self.TRAINING_ARGS_NAME))
    
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
    