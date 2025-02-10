#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/with_pretraining/config_4_bert_large.log
#SBATCH --error=logs/with_pretraining/error/config_4_bert_large.log
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nuni18092003@gmail.com
#SBATCH --account=p_scads_llm_secrets

source ./../semevalenv/bin/activate
export HF_HOME=

python commands.py train_with_pretrained_model_and_save_best \
        --config_path ./config/with_pretraining/config_4_bert_large.yaml
