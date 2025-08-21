# 📘 Supplementary Materials  
## Optimizing Small Transformer-based Language Models for Multi-Label Sentiment Analysis in Short Texts  (_LDD@ECAI 2025_)

### 📄 Abstract  
Sentiment classification in short text datasets presents unique challenges, including class imbalance, limited training data, and the inherent subjectivity of sentiment labels. These challenges are further exacerbated by the brevity of the texts, which limits contextual cues necessary for disambiguation.  

In this work, we evaluate the effectiveness of **small Transformer-based models** (BERT and RoBERTa with fewer than 1 billion parameters) for **multi-label sentiment classification**, specifically in short-text scenarios. We focus on three key optimization strategies:

1. **Continued domain-specific pre-training**
2. **Generative data augmentation**
3. **Architectural variations in the classification head**

**Key Findings**:
- Generative data augmentation significantly improves classification performance.
- Continued pre-training on augmented data may introduce noise, with minimal or negative effects on performance.
- Architectural tweaks to the classification head provide only marginal gains.

These insights aim to guide practitioners in efficiently adapting small Transformer models for nuanced sentiment analysis tasks, particularly in low-resource or short-text environments.

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

For questions or collaborations, feel free to reach out: *[michael.faerber@tu-dresden.de](mailto:michael.faerber@tu-dresden.de)*
