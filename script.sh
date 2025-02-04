#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/roberta_base_ext_fc_768_4.log
#SBATCH --error=logs/error_roberta_base_ext_768_4.log
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nuni18092003@gmail.com
#SBATCH --account=p_scads_llm_secrets

source semevalenv/bin/activate
export HF_HOME=

python Semeval_Task/commands.py train_Roberta_model_and_save_best \
        --model_name roberta-base \
        --dataset_path Semeval_Task/data/track_a/train/eng.csv \
        --classification_head_size 768 \
        --head_type fc \
        --save_as fc_768_4 \
        --extended yes \
        --extended_split 0.1 \
        --classification_layers 4 \
        --attention_dim 256 \
        --num_attention_heads 4


#SBATCH --gres=gpu:1