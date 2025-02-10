# Semeval_Task
Repository for the Codebase etc. of Semeval Task for Behind the Secrets of LLMS 


This Repository Contains Scripts as well as documentation for the semeval task 11 (Codabench)
Provided Features:
- Training Script via commands.py
- utilities via utilities.py
- webapp for human evaluation via human_eval.py
- human evaluation evaluation via human_eval_evaluation.py :)
- interactive shap view via shap_text_plot.html
- results found in results/final_eval
- trainings configs found in /config
- sbatch scripts found in /scripts
- and some legacy plots etc



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


run batch job via `sbatch ...` 
