import transformers

from utilities import *
from peft import LoraConfig, TaskType, get_peft_model
from upycli import command
import yaml

#def train_t5_model_and_save_best(model_name,dataset_path, freeze_layers = False, freeze_to_layer = 12, loRa = False):


def pretrain_model(config, save_folder_path, save_folder_name): 

    # load config 
    model_name = config["model_to_pretrain"]
    dataset_path = config["dataset_path"]
    train_set_split = config["training_set_split"]
    extended_dataset = config["extended_dataset"]

    # load pretraining args
    pretraining_args = config["training"]

    train_epochs = pretraining_args["train_epochs"]
    learning_rate = pretraining_args["learning_rate"]
    weight_decay = pretraining_args["weight_decay"]
    


    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_set, validation_set = load_and_split_dataset(dataset_path, train_set_split)

    if extended_dataset: 
        extended_split = config["extended_split"]
        training_set_extension, _ = load_and_split_dataset("data/track_a/train/extended_split_data.csv",extended_split)
        print(training_set_extension)
        training_set = pd.concat([training_set, training_set_extension], axis=0, ignore_index=True)

    training_set = tokenize_dataset_for_pretraining(dataset = training_set, tokenizer = tokenizer) 
    validation_set = tokenize_dataset_for_pretraining(dataset = validation_set, tokenizer = tokenizer) 

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm_probability = 0.15)

    model = AutoModelForMaskedLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir = save_folder_path,
        eval_strategy = "steps",
        eval_steps = 100,
        learning_rate = learning_rate,
        num_train_epochs = train_epochs,
        weight_decay = weight_decay,
        
        # logging 
        logging_dir=f"./logs/pretraining/{save_folder_name}",

        ## best model
        load_best_model_at_end=True,
        metric_for_best_model = "loss",
        save_strategy="steps",
        save_steps=100
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = training_set,
        eval_dataset = validation_set,
        data_collator = data_collator,
        tokenizer = tokenizer,
    )

    trainer.train()

    results = trainer.evaluate()

    trainer.save_model(f"{save_folder_path}/best_model")

    # save results 
    config["results"] = results 

    with open(f"{save_folder_path}/results.yaml", "w") as file:
        yaml.dump(config, file)

    remove_all_files_and_folders_except_best_model(save_folder_path)


def train_with_pretrained_model_and_save_best(config_path): 

    # load config from yaml
    with open(config_path, 'r') as file: 
        config = yaml.safe_load(file)

    # load classification config 
    classification_config = config["classification"]

    dataset_path = classification_config["dataset_path"]
    train_set_split = classification_config["training_set_split"]
    custom = classification_config["custom"]
    extend_dataset = classification_config["extended_dataset"]
    freeze_layers = classification_config["freeze_layers"]
    freeze_to_layer = classification_config["freeze_to_layer"]
    loRa = classification_config["loRa"]
    extended_split = classification_config["extended_split"]

    # load training args
    classification_args = classification_config["training"]

    train_epochs = classification_args["train_epochs"]
    learning_rate = classification_args["learning_rate"]
    per_dev_tr_bch_sz = classification_args["per_device_train_batch_size"]
    per_dev_evl_bch_sz = classification_args["per_device_eval_batch_size"]
    warmup_steps = classification_args["warmup_steps"]
    weight_decay = classification_args["weight_decay"]

    # look if a model should be pretrained or a pretrained model should be loaded 
    if "pretraining" in config.keys(): 

        pretrained_model = None
        model_to_pretrain = config["pretraining"]["model_to_pretrain"]

    else: 

        model_to_pretrain = None 
        pretrained_model = classification_config["pretrained_model"]


    if model_to_pretrain is not None and pretrained_model is None: 
        
        save_file_path, pretrained_model_name = get_save_file_path(model_name = model_to_pretrain, category = 1)

        config["pretraining"]["extended_dataset"] = classification_config["extended_dataset"]
        config["pretraining"]["extended_split"] = classification_config["extended_split"]
        config["pretraining"]["training_set_split"] = classification_config["training_set_split"]
        # further pretrain model with MaskedLanguageModeling objective 
        pretrain_model(config["pretraining"], save_file_path, pretrained_model_name)

        # load tokenizer from further pretrained model 
        tokenizer = AutoTokenizer.from_pretrained(f"{save_file_path}/best_model")

        # load pretrained model with classification head 
        if custom: 
            
            # load the classifier size 
            classifier_size = classification_config["classifier_size"]
            

            model = CustomClassifier(model_name = f"{save_file_path}/best_model", model_type = transformers.AutoModelForMaskedLM, classifier_size = classifier_size)
            
            # to initialize LazyLinear layer to fit model output dimension to classification head input dimension
            dummys = torch.zeros((2, 512), dtype=torch.long)
            label_dummys = torch.zeros((2, 5), dtype=torch.float)
            model(label_dummys,dummys, dummys, dummys)    

        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(f"{save_file_path}/best_model", num_labels = 5)

    elif model_to_pretrain is None and pretrained_model is not None: 

        pretrained_model_name = pretrained_model

        # load tokenizer from further pretrained model 
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        # load pretrained model with classification head 
        if custom:

            # load the classifier size 
            classifier_size = classification_config["classifier_size"]
            dropout_rate = classification_config["dropout_rate"]
            head_type = classification_config["head_type"] # "attention" or "fc"
            attention_dim = classification_config["attention_dim"]
            classification_layer_count = classification_config["classification_layer_count"]
            num_attention_heads = classification_config["num_attention_heads"]
            

            model = CustomClassifier(model_name = pretrained_model, model_type = transformers.AutoModelForMaskedLM, classifier_size = classifier_size,
                                     dropout_rate=dropout_rate,
                                     head_type=head_type,
                                     attention_dim=attention_dim,
                                     classification_layers_cnt=classification_layer_count,
                                     num_attention_heads=num_attention_heads) # maybe try bigger classigier_size? 

            # to initialize LazyLinear layer to fit model output dimension to classification head input dimension
            dummys = torch.zeros((2, 512), dtype=torch.long)
            label_dummys = torch.zeros((2, 5), dtype=torch.float)
            model(label_dummys,dummys, dummys, dummys) 

        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels = 5)

    else: 
        print(f"Parameter combination of model_to_pretrain = {model_to_pretrain} and pretrained_model = {pretrained_model} is not valid.")

    # debug 
    print(model)

    # load dataset 
    training_set, validation_set = load_and_split_dataset(dataset_path, train_set_split)

    # extend dataset with synthetic training data from ChatGPT
    if extend_dataset:
        training_set_extension, _ = load_and_split_dataset("data/track_a/train/extended_split_data.csv",extended_split)
        training_set = pd.concat([training_set, training_set_extension], axis=0, ignore_index=True)

    # tokenize dataset 
    training_set = tokenize_dataset(training_set, tokenizer, max_len = 512)
    validation_set = tokenize_dataset(validation_set, tokenizer, max_len = 512)

    # apply loRa or freeze some layers of the pretrained model 
    if loRa: 
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=1, 
            lora_alpha=1, 
            lora_dropout=0.1, 
            target_modules = ["query", "value"],
            modules_to_save = ["pre_classifier", "classifier"] # keep custom classification head trainable 
        )

        model = get_peft_model(model, lora_config)

        # LoRa info 
        print(model.print_trainable_parameters())

    if custom: 
        if freeze_layers: 
            for param in model.l1.roberta.encoder.layer[:freeze_to_layer-1].parameters(): 
                param.requires_grad = False
    else:    
        if freeze_layers: 
            for param in model.roberta.encoder.layer[:freeze_to_layer-1].parameters(): 
                param.requires_grad = False

    # training arguments 
    save_file_path, fine_tuned_model_name = get_save_file_path(model_name = pretrained_model_name, category = 2)

    training_args = TrainingArguments(
        output_dir = save_file_path,
        num_train_epochs = train_epochs,
        per_device_train_batch_size=  per_dev_tr_bch_sz,
        per_device_eval_batch_size = per_dev_evl_bch_sz,
        learning_rate = learning_rate,
        warmup_steps = warmup_steps,
        weight_decay = weight_decay,
        logging_dir = f"./logs/classification/{fine_tuned_model_name}",
        eval_strategy = 'steps',  # Evaluate at the end of each epoch
        eval_steps = 100,
        eval_on_start = True,
        logging_steps = 10,
        label_names = ["labels"],
        dataloader_drop_last = True,
        ## ---
        report_to = "tensorboard",
    
        ## best model
        metric_for_best_model = "f1",
        greater_is_better = True,
        load_best_model_at_end = True,
        save_strategy = "steps",
        save_steps = 100
    )

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = CustomTrainer(model=model, args=training_args,
                            train_dataset=training_set,
                            eval_dataset=validation_set,
                            data_collator=data_collator, 
                            compute_metrics=compute_metrics_f1)
    
    # train and evaluate result 
    trainer.train()

    results = trainer.evaluate()
    print(results)

    # save the results 
    config["results"] = results

    with open(f"{save_file_path}/results.yaml", "w") as file:
        yaml.dump(config, file)

    # save best model as .pth file 
    model.load_state_dict(torch.load(f"{trainer.state.best_model_checkpoint}/pytorch_model.bin",weights_only=True))
    torch.save(model, f"{save_file_path}/best_model.pth")

    # clean up 
    remove_all_files_and_folders_except_best_model(save_file_path)

    return 


def train_auto_model_and_save_best(model_name, dataset_path, freeze_layers = False, freeze_to_layer = 12, loRa = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    training_set, validation_set = load_and_split_dataset(dataset_path, 0.95)
    training_set_extension, _ = load_and_split_dataset("data/public_data/train/track_a/extended_eng.csv",1.0)
    print(training_set,training_set_extension)
    training_set = pd.concat([training_set, training_set_extension], axis=0, ignore_index=True)
    training_set = tokenize_dataset(training_set, tokenizer, 512)
    validation_set = tokenize_dataset(validation_set, tokenizer, 512)
    
    model = CustomClassifier(model_name=model_name, model_type=AutoModelForMaskedLM,classifier_size=768)
    
    # print name and type of all modules the model contains 
    #print([(n, type(m)) for n, m in model.named_modules()])

    # forward dummy batch: for lazy initialization linear classifier of model
    dummys = torch.zeros((2, 512), dtype=torch.long)
    label_dummys = torch.zeros((2, 5), dtype=torch.float)
    res = model(label_dummys,dummys, dummys, dummys) # to initialize LazyLinear layer to fit model output dimension to classification head input dimension   
    #print(res)
    if loRa: 
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=1, 
            lora_alpha=1, 
            lora_dropout=0.1, 
            target_modules = ["query", "value"],
            modules_to_save = ["pre_classifier", "classifier"] # keep custom classification head trainable 
        )

        model = get_peft_model(model, lora_config)

        # LoRa info 
        print(model.print_trainable_parameters())

    # freeze layers of pretrained base model 
    if freeze_layers: 
        for param in model.l1.roberta.encoder.layer[:freeze_to_layer-1].parameters(): 
            param.requires_grad = False
    
    save_file_path, _ = get_save_file_path(model_name, category = 2)
    
    training_args = TrainingArguments(
        output_dir=save_file_path,
        num_train_epochs=10,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
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
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=100
    )

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = CustomTrainer(model=model, args=training_args,
                            train_dataset=training_set,
                            eval_dataset=validation_set,
                            data_collator=data_collator, 
                            compute_metrics=compute_metrics_f1)
    # torch.save(model, f"{save_file_path}/base_model.pth")
    trainer.train()
    print(f"Best model saved at {trainer.state.best_model_checkpoint}")
    trainer._load_best_model()
    results = trainer.evaluate()
    print(results)
    model = CustomClassifier(model_name, AutoModelForMaskedLM, 768)
    model.load_state_dict(torch.load(f"{trainer.state.best_model_checkpoint}/pytorch_model.bin",weights_only=True))
    torch.save(model, f"{save_file_path}/best_model.pth")
    
    trainer = CustomTrainer(model=model, args=training_args,
                            train_dataset=training_set,
                            eval_dataset=validation_set,
                            data_collator=data_collator, 
                            compute_metrics=compute_metrics_f1)
    print("Best model loaded")
    print(trainer.evaluate(eval_dataset=validation_set))
    
    remove_all_files_and_folders_except_best_model(save_file_path)
    
    return
    

@command
def train_Roberta_model_and_save_best(model_name,dataset_path, freeze_layers = False, freeze_to_layer = 12, loRa = False, classification_head_size = 768,head_type="fc",save_as="",extended="yes",classification_layers=2,attention_dim=128,num_attention_heads=1,extended_split=1.0):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    training_set, validation_set = load_and_split_dataset(dataset_path, 0.80)
    if extended=="yes":
        training_set_extension, _ = load_and_split_dataset("Semeval_Task/data/track_a/train/extended_split_data.csv",extended_split)
        print(training_set,training_set_extension)
        training_set = pd.concat([training_set, training_set_extension], axis=0, ignore_index=True)
    training_set = training_set[training_set['text'].apply(lambda x: isinstance(x, str))]
    assert all(isinstance(text, str) for text in training_set['text']), "Invalid text type in training_set"
    
    training_set = tokenize_dataset(training_set, tokenizer, 512)
    validation_set = tokenize_dataset(validation_set, tokenizer, 512)

    

    model = CustomClassifier(model_name = model_name, model_type = RobertaModel, classifier_size = classification_head_size ,dropout_rate=0, head_type=head_type, attention_dim=attention_dim,classification_layers_cnt=classification_layers,num_attention_heads=num_attention_heads)

    print(model)
    # print name and type of all modules the model contains 
    #print([(n, type(m)) for n, m in model.named_modules()])

    # forward dummy batch: for lazy initialization linear classifier of model
    dummys = torch.zeros((2, 512), dtype=torch.long)
    label_dummys = torch.zeros((2, 5), dtype=torch.float)
    res = model(label_dummys,dummys, dummys, dummys) # to initialize LazyLinear layer to fit model output dimension to classification head input dimension   
    #print(res)
    if loRa: 
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=1, 
            lora_alpha=1, 
            lora_dropout=0.1, 
            target_modules = ["query", "value"],
            modules_to_save = ["pre_classifier", "classifier"] # keep custom classification head trainable 
        )

        model = get_peft_model(model, lora_config)

        # LoRa info 
        print(model.print_trainable_parameters())

    # freeze layers of pretrained base model 
    if freeze_layers: 
        for param in model.l1.encoder.layer[:freeze_to_layer-1].parameters(): 
            param.requires_grad = False
    
    save_file_path, fine_tuned_model_name  = get_save_file_path(model_name + save_as, category = 2)
    
    training_args = TrainingArguments(
        output_dir=save_file_path,
        num_train_epochs=30,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs/{fine_tuned_model_name}',
        eval_strategy='steps',  # Evaluate at the end of each epoch
        eval_steps=100,
        eval_on_start=True,
        logging_steps=10,
        label_names=["labels"],
        dataloader_drop_last=True,
        ## ---
        report_to="tensorboard",
    
        ## best model
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1
    )

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = CustomTrainer(model=model, args=training_args,
                            train_dataset=training_set,
                            eval_dataset=validation_set,
                            data_collator=data_collator, 
                            compute_metrics=compute_metrics_f1)
    # torch.save(model, f"{save_file_path}/base_model.pth")
    trainer.train()
    print(f"Best model saved at {trainer.state.best_model_checkpoint}")
    trainer._load_best_model()
    results = trainer.evaluate()
    print(results)
    model = CustomClassifier(model_name = model_name, model_type = RobertaModel, classifier_size = classification_head_size,dropout_rate=0, head_type=head_type, attention_dim=attention_dim,classification_layers_cnt = classification_layers,num_attention_heads=num_attention_heads)

    if loRa: 
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=1, 
            lora_alpha=1, 
            lora_dropout=0.1, 
            target_modules = ["query", "value"],
            modules_to_save = ["pre_classifier", "classifier"] # keep custom classification head trainable 
        )
        dummys = torch.zeros((2, 512), dtype=torch.long)
        label_dummys = torch.zeros((2, 5), dtype=torch.float)
        res = model(label_dummys,dummys, dummys, dummys) # to initialize LazyLinear layer to fit model output dimension to classification head input dimension   
    
        model = get_peft_model(model, lora_config)
    model.load_state_dict(torch.load(f"{trainer.state.best_model_checkpoint}/pytorch_model.bin",weights_only=True))
    torch.save(model, f"{save_file_path}/best_model.pth")
    
    trainer = CustomTrainer(model=model, args=training_args,
                            train_dataset=training_set,
                            eval_dataset=validation_set,
                            data_collator=data_collator, 
                            compute_metrics=compute_metrics_f1)
    print("Best model loaded")
    result = trainer.evaluate(eval_dataset=validation_set)
    print(result)
    
    with open(f"{save_file_path}/results.yaml", "w") as file:
        yaml.dump(result, file)

    
    remove_all_files_and_folders_except_best_model(save_file_path)
    
    
    
    return

def load_and_validate_Roberta_model_on_dev_set(model_name, model_path, dataset_path):
    """
    Classifies the texts in a development dataset using a RoBERTa-based classifier,
    fills in sentiment labels, and saves only the ID and labels into a new file.

    Args:
        dev_file (str): Path to the development CSV file.
        output_file (str): Path to save the classified CSV file.
        model_name (str): Pretrained RoBERTa model name or path.
        tokenizer_name (str): Tokenizer name corresponding to the RoBERTa model.
        emotion_columns (list): List of sentiment labels (e.g., ["Anger", "Fear", "Joy", "Sadness", "Surprise"]).
    """
    import torch
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    import pandas as pd
    from sklearn.preprocessing import MultiLabelBinarizer

    # Load the development dataset
    dev_set = load_and_split_dataset(dataset_path,split_ratio=1.0)

    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = torch.load(f"./results/{model_path}/best_model.pth")# should be .results/.../best_model.pth or similar
    model.eval()  # Set model to evaluation mode

    emotion_columns = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=emotion_columns)
    mlb.fit([emotion_columns])  # Ensure consistent encoding

    # Prepare output containers
    classified_results = []
    ids = []

    for _, row in dev_set.iterrows():
        text = row["text"]
        id_ = row["id"]

        # Tokenize the input text
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        # Perform inference with the model
        with torch.no_grad():
            outputs = model(**inputs)
            #print(outputs)
            probabilities = torch.sigmoid(outputs)
            predicted_labels = (probabilities > 0.5).int().tolist()[0]

        # Convert predictions to one-hot encoding
        one_hot_encoded = mlb.transform([[
            emotion_columns[i] for i, label in enumerate(predicted_labels) if label == 1
        ]])

        # Store the results
        classified_results.append(one_hot_encoded.flatten())
        ids.append(id_)

    # Combine IDs and classified results into a DataFrame
    classified_df = pd.DataFrame(classified_results, columns=emotion_columns)
    classified_df.insert(0, "id", ids)

    # Save the classified dataset
    classified_df.to_csv(f"results/{model_path}/dev_set.csv", index=False)
    print(f"Classified development dataset saved")


@command
def load_and_validate_Roberta_model(model_name, model_path, dataset_path, plot_conf_mat = "none", dev_set=False):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    train_set, validation_set = load_and_split_dataset(dataset_path,0.95)
    validation_set = tokenize_dataset(validation_set, tokenizer, 512)

    # TODO remove, only for testing
    # validation_set = validation_set[:35]

    #model = CustomClassifier(model_name, RobertaModel, 768)
    model = torch.load(f"./results/{model_path}/best_model.pth")# should be .results/.../best_model.pth or similar
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(output_dir=".",
                                        per_device_eval_batch_size=6,
                                        label_names=["labels"],
                                        dataloader_drop_last=True)
    trainer = CustomTrainer(model=model,
                            args=training_args,
                            eval_dataset=validation_set,
                            data_collator=data_collator, 
                            compute_metrics=compute_metrics_f1)
    
        
    
    predictions = trainer.predict(validation_set)

    if dev_set:
        print(f"Evaluated dev set {predictions}")
        
    print(predictions.metrics)
    
    
    if plot_conf_mat == "plot": 

        plot_confusion_matrix(predictions)

    elif plot_conf_mat == "plot_and_save": 

        plot_confusion_matrix(predictions, save_path = "./graphs/confusion_matrix", file_name = f"{model_path}_confusion_mat")

    elif plot_conf_mat == "save": 

        plot_confusion_matrix(predictions, save_path = "./graphs/confusion_matrix", file_name = f"{model_path}_confusion_mat", show = False)
    
    elif plot_conf_mat == "none": 
        print("Confusion matrix is not plotted or saved.")
    else: 
        print(f"'{plot_conf_mat}' is no valid value for this parameter. The confusion matrix is not plotted or saved.")

    return
    



#train_Roberta_model_and_save_best("roberta-base","data/public_data/train/track_a/eng.csv")
if __name__ == "__main__": 
    # load_and_validate_Roberta_model("roberta-base","roberta-base_2024-12-16_19-13-00","data/public_data/train/track_a/eng.csv", plot_conf_mat = "save")
    # train_Roberta_model_and_save_best(model_name = "roberta-base", dataset_path = "data/track_a/train/eng.csv", freeze_layers = False, freeze_to_layer = 12, loRa = False)
    # train_Roberta_model_and_save_best(model_name = "roberta-large", dataset_path = "Semeval_Task/data/public_data/train/track_a/eng.csv", freeze_layers = True, freeze_to_layer = 24, loRa = False)
    # train_Roberta_model_and_save_best(model_name = "roberta-base", dataset_path = "Semeval_Task/data/public_data/train/track_a/eng.csv", freeze_layers = True, freeze_to_layer = 24, loRa = False)
    # pretrain_Roberta_model(model_name = "distilbert/distilroberta-base", dataset_path = "data/public_data/train/track_a/eng.csv")
    # load_and_validate_Roberta_model("roberta-base","roberta-base_2025-01-03_16-56-32","data/public_data/dev/track_a/eng_a.csv", plot_conf_mat = "save",dev_set=True)
    # load_and_validate_Roberta_model_on_dev_set("roberta-base","roberta-base_2025-01-03_16-56-32","data/public_data/dev/track_a/eng_a.csv")
    #train_Roberta_model_and_save_best(model_name = "roberta-base", dataset_path = "data/public_data/train/track_a/eng.csv", freeze_layers = True, freeze_to_layer = 12, loRa = False)
    #train_auto_model_and_save_best(model_name = "distilbert/distilroberta-base", dataset_path = "data/public_data/train/track_a/eng.csv", freeze_layers = False, freeze_to_layer = 12, loRa = False)
    #train_t5_model_and_save_best(model_name = "google-t5/t5-small", dataset_path = "data/public_data/train/track_a/eng.csv", freeze_layers = False, freeze_to_layer = 12, loRa = False)
    
    # train_with_pretrained_model_and_save_best(pretrained_model = "roberta-base_2025-01-03_17-14-47", custom = True, loRa = True)
    train_with_pretrained_model_and_save_best(config_path = "./config/with_pretraining/config.yaml")


