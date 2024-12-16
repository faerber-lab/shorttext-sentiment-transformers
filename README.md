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

# Load modules
```
module load release/24.04
module load GCCcore/11.3.0
module load Python/3.10.4
```

create workspace on hpc:
```
ws_allocate name 100 -r 7 # more info: ws_allocate -h
```

link workspace
```
ln -s <ws_link> <name>
```

create venv in workspace:
```
virtualenv myvenv
```

activate venv:
```
. myvenv/bin/activate
pip install -r requirements.txt
```

run interactive job:
```
srun --ntasks=1 --cpus-per-task=1 --time=4:00:00 --mem-per-cpu=8000 --pty --nodes=1 --account=p_scads_llm_secrets --gpus-per-task=1 bash -l
```

activate venv inside job
and work inside the workspace with running venv

batch job: TODO



## Paperinhalt 
1. Aufgabenbeschreibung 
2. evtl. Einordnung 
3. Modellauswahl begründen, Warum pretrained Bert? 
4. Finetuning -> eigener Attention Head, Huggingface Model,...
5. Trainingsergebnisse / Hyperparameter Tuning 
6. Explainability SHAP
7. Vergleich mit anderen Modellen / Menschen 
8. Quellen