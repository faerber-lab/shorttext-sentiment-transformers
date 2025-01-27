# Semeval_Task
Repository for the Codebase etc. of Semeval Task for Behind the Secrets of LLMS 


Clone this Repo via
```
git clone git@github.com:TUD-Semeval-Group/Semeval_Task.git
```


```
git lfs install
```

```
git lfs pull
```

## Setup

1. SSH on Hpc
2. Load modules (has to be done only for initial setup)
3. Create workspace (has to be done only for initial setup)
4. Link workspace (has to be done only for initial setup)
5. Create Venv in workspace (has to be done only for initial setup)
6. Run interactive job (you will be redirected to a compute node)
7. Activate Venv 
8. Run python script

### How To:
#### SSH On Hpc
```
ssh <zih-login>@login2.alpha.hpc.tu-dresden.de
```

#### Load Modules

```
module load release/24.04
module load GCCcore/11.3.0
module load Python/3.10.4
module save
```

#### Create Workspace On Hpc:
Workspace will be up for 100 days. 
```
ws_allocate <name> 100 -r 7 # more info: ws_allocate -h
```

#### Link Workspace
```
ln -s <ws_link> <name>
```

#### Create Venv In Workspace:
```
virtualenv <myvenv>
```

#### Activate Venv:
```
. myvenv/bin/activate
pip install -r requirements.txt
```

#### Run Interactive Job:
```
srun --ntasks=1 --cpus-per-task=1 --time=4:00:00 --mem-per-cpu=8000 --pty --nodes=1 --account=p_scads_llm_secrets --gres=gpu:1 bash -l
```

activate venv inside job
and work inside the workspace with running venv

batch job: TODO

## Known Bugs: 

### Default Trainer drops first row of model outputs

#### Workaroud
Custom Trainer with custom compute_loss function that adds a dummy line to the output. 

#### Solution 
Trainer expects the model to compute the loss and to return it as the first row of the output. Therefore this first row is dropped before continuing with the model output. -> Maybe change custom model accordingly?

### Trainer.predict() returns too little solutions sometimes
When the number of the examples in the test set which is given to the trainer.predict() method is not divisible by the number which is set for the 'per_device_eval_batch_size' in the TrainigArguments the number of returnt predictions is dropped to the nearest number that is divisable. 

#### Workaround
Input a divisible number of examples.

## Trained Models 

### roberta-base_2024-12-16_19-13-00

#### CustomClassifier
- model_name = "roberta-base" 
- model_type = RobertModel 
- classifier_size = 768 
- dropout_rate = 0.3 
- num_classes = 5

#### Dataset Configurations
- split_ratio = 0.95 
- random_state = 123 

#### Freeze Layers 
- not used 

#### loRa
- not used 

#### Training Arguments 
- num_train_epochs = 1
- per_device_train_batch_size = 4 
- per_device_eval_batch_size = 4 
- warmup_steps = 500
- weight_decay = 0.01

#### Performance 
- 



## Paperinhalt 
1. Aufgabenbeschreibung 
2. evtl. Einordnung 
3. Modellauswahl begründen, Warum pretrained Bert? 
4. Finetuning -> eigener Attention Head, Huggingface Model,...
5. Trainingsergebnisse / Hyperparameter Tuning 
6. Explainability SHAP
7. Vergleich mit anderen Modellen / Menschen 
8. Quellen


## Besprechung 14.01.2025: 
1. Powerlaws bei Auswertung mit einbeziehen? 
2. eventuell Datenset-Vorverarbeitung implementieren? 
3. Welche Variationen des Attention Heads? 
4. Subsection "Evaluation Metrics" vielleicht in Kaptitel 4 verschieben 
5. Wo Beschreibung des Datensetsplits + Trainingsargumenten einfügen? 
6. Wo SHAP einfügen? 
7. Noch mehr über die BERT models schreiben? 
8. Datensetsplitkombination von Pre-Training und Fine-Tuning

### Ergebnis: 
1. Werden sich Gedanken drüber gemacht 
2. Ja machen, mit in Metholodgy
3. Nur Size anpassen -> Größe immer weiter verdoppeln + Ergebnis aufzeichnen
4. Ne bleibt so, 
5. In Methodology, 3.2 allgemein über das Trainingsset 
6. Subsection "Explainability" bei Methodik 
7. Ne, aber Subsection mit Beschreibung Classification Heads
8. Pretraining einmal auf normalen Trainingsset und einmal auf extended Trainingsset 


- bei Methodology Subsection für Implimentation hinzufügen 

## Besprechung 28.01.2025: 
1. Sigmoid-layer bei Custom Classification Head nicht benutzt? 
2. Paper über Größe Classification Head. 


 




## ToDos: 
- In Related Word: Paper das sagt, dass Transformer / Encoder gut geeignet sind, bzw. mit anderen Verfahren wie Word2Vec / LSTM vergleichen -> Attention gut, um komplexe Zusammenhänge im Text mit einzubeziehen, ist wichtig für Sentimentanalysis -> auf Bert Paper / Roberta Paper verweisen, Paper "HowToFineTuneBertForTextClassification" -> weiteres Pretraining 
