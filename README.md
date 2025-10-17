## Supplementary Materials  
### Optimizing Small Transformer-based Language Models for Multi-Label Sentiment Analysis in Short Texts (_LDD@ECAI 2025_)

This study explores the performance of small Transformer models (sub-1B parameters) for **multi-label sentiment classification** in **short texts**, addressing challenges like class imbalance and limited context.

We assess three optimization strategies:  
1. Domain-specific continued pre-training  
2. Generative data augmentation  
3. Classification head modifications  

**Findings**:  
- Generative augmentation boosts performance notably.  
- Continued pre-training on synthetic data can be noisy and counterproductive.  
- Classification head changes yield only minor improvements.

These results offer practical guidance for adapting lightweight models in low-resource, short-text sentiment analysis tasks.

---

### Features & Directory Overview

| Component                          | Description                                           |
|------------------------------------|-------------------------------------------------------|
| `commands.py`                      | Main training and evaluation script                   |
| `utilities.py`                     | Helper functions and utility code                     |
| `human_eval.py`                    | Lightweight webapp for human annotation               |
| `human_eval_evaluation.py`        | Evaluation script for human annotation results        |
| `shap_text_plot.html`             | Interactive SHAP visualization for model interpretability |
| `results/final_eval/`             | Final evaluation metrics and output files             |
| `config/`                          | Training configuration files (hyperparams, etc.)      |
| `scripts/`                         | SBATCH scripts for running experiments on HPC         |
| `legacy/` *(optional)*            | Additional legacy plots and resources                 |

---


### 📫 Contact

For any questions feel free to reach out: *[michael.faerber@tu-dresden.de](mailto:michael.faerber@tu-dresden.de)*
