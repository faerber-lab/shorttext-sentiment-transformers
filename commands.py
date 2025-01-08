import transformers

from utilities import *
from peft import LoraConfig, TaskType, get_peft_model
from upycli import command
#def train_t5_model_and_save_best(model_name,dataset_path, freeze_layers = False, freeze_to_layer = 12, loRa = False):


def pretrain_Roberta_model(model_name, dataset_path): 

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_set, validation_set = load_and_split_dataset(dataset_path, 0.95)

    training_set = tokenize_dataset_for_pretraining(dataset = training_set, tokenizer = tokenizer) 
    validation_set = tokenize_dataset_for_pretraining(dataset = validation_set, tokenizer = tokenizer) 

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm_probability = 0.15)

    model = AutoModelForMaskedLM.from_pretrained(model_name)

    save_file_path = get_save_file_path(model_name = model_name)

    training_args = TrainingArguments(
        output_dir = save_file_path,
        eval_strategy = "steps",
        eval_steps = 100,
        learning_rate = 2e-5,
        num_train_epochs = 3,
        weight_decay = 0.01,
        
        # logging 
        logging_dir='./logs',

        ## best model
        load_best_model_at_end=True,
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

    trainer.save_model(f"{save_file_path}/best_model")



def train_auto_model_and_save_best(model_name,dataset_path, freeze_layers = False, freeze_to_layer = 12, loRa = False):
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
    
    save_file_path = get_save_file_path(model_name)
    
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
def train_Roberta_model_and_save_best(model_name,dataset_path, freeze_layers = False, freeze_to_layer = 12, loRa = False):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    training_set, validation_set = load_and_split_dataset(dataset_path, 0.95)
    training_set_extension, _ = load_and_split_dataset("data/public_data/train/track_a/extended_eng.csv",1.0)
    print(training_set,training_set_extension)
    training_set = pd.concat([training_set, training_set_extension], axis=0, ignore_index=True)
    training_set = tokenize_dataset(training_set, tokenizer, 512)
    validation_set = tokenize_dataset(validation_set, tokenizer, 512)

    

    model = CustomClassifier(model_name = model_name, model_type = RobertaModel, classifier_size = 768)

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
    
    save_file_path = get_save_file_path(model_name)
    
    training_args = TrainingArguments(
        output_dir=save_file_path,
        num_train_epochs=2,
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
    model = CustomClassifier(model_name, RobertaModel, 768)
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
    print(trainer.evaluate(eval_dataset=validation_set))
    
    remove_all_files_and_folders_except_best_model(save_file_path)
    
    return

@command
def load_and_validate_Roberta_model(model_name, model_path, dataset_path, plot_conf_mat = "none"):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    test_set, validation_set = load_and_split_dataset(dataset_path,0.95)
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
    train_Roberta_model_and_save_best(model_name = "roberta-base", dataset_path = "data/public_data/train/track_a/eng.csv", freeze_layers = False, freeze_to_layer = 12, loRa = True)
    # pretrain_Roberta_model(model_name = "distilbert/distilroberta-base", dataset_path = "data/public_data/train/track_a/eng.csv")
    #load_and_validate_Roberta_model("roberta-base","roberta-base_2024-12-16_19-13-00","data/public_data/train/track_a/eng.csv", plot_conf_mat = "save")
    #train_Roberta_model_and_save_best(model_name = "roberta-base", dataset_path = "data/public_data/train/track_a/eng.csv", freeze_layers = True, freeze_to_layer = 12, loRa = False)
    #train_auto_model_and_save_best(model_name = "distilbert/distilroberta-base", dataset_path = "data/public_data/train/track_a/eng.csv", freeze_layers = False, freeze_to_layer = 12, loRa = False)
    #train_t5_model_and_save_best(model_name = "google-t5/t5-small", dataset_path = "data/public_data/train/track_a/eng.csv", freeze_layers = False, freeze_to_layer = 12, loRa = False)
    
    
    # model = RobertaModel.from_pretrained("roberta-base")

    # lora_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS, r=1, lora_alpha=1, lora_dropout=0.1
    # )

    # model = get_peft_model(model, lora_config)

    # print(model)

