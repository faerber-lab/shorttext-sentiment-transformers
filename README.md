## Supplementary materials: 

## Optimizing Small Transformer-based Language Models for Multi-Label Sentiment Analysis in Short Texts (LDD@ECAI'25)

**Abstract**: Sentiment classification in short text datasets faces significant challenges such as class imbalance, limited training samples, and the inherent subjectivity of sentiment labels—issues that are further intensified by the limited context in short texts. These factors make it difficult to resolve ambiguity and exacerbate data sparsity, hindering effective learning.In this paper, we evaluate the effectiveness of _small_ Transformer-based models (i.e., BERT and RoBERTa, with fewer than 1 billion parameters) for multi-label sentiment classification, with a particular focus on short-text settings. Specifically, we evaluated three key factors influencing model performance: **(1) continued domain-specific pre-training**, **(2) data augmentation** using automatically generated examples, specifically generative data augmentation, and **(3) architectural variations of the classification head**. Our experiment results show that data augmentation improves classification performance, while continued pre-training on augmented datasets can introduce noise rather than boost accuracy. Furthermore, we confirm that modifications to the classification head yield only marginal benefits. These findings provide practical guidance for optimizing BERT-based models in resource-constrained settings and refining strategies for sentiment classification in short-text datasets. 
### Features

- **Training Script:** `commands.py`
- **Utilities:** `utilities.py`
- **Human Evaluation Webapp:** `human_eval.py`
- **Evaluation Script:** `human_eval_evaluation.py`
- **Interactive SHAP Visualization:** `shap_text_plot.html`
- **Final Results:** `results/final_eval`
- **Training Configs:** `/config`
- **SBATCH Scripts:** `/scripts`
- **Additional Legacy Plots and Resources**

### Cloning the Repository

Clone the repository and initialize Git LFS:
```bash
git clone git@github.com:TUD-Semeval-Group/Semeval_Task.git
git lfs install
git lfs pull
```
