# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** Kasukabe-Defence-Group-Solution 
**Team Members:** akshit, akshat, anirudha, somil 
**Submission Date:** 13 october

---

## 1. Executive Summary
Performed Full Fine Tuning, Using BERT, we didn't use the LoRA or anything else, just Full Fine Tuning with a regressor head. used optuna for finetuning.


---

## 2. Methodology Overview

### 2.1 Problem Analysis
We perform EDA and found out that price data was skewed so we removed all the data points above 145 dollar price. also to incorporate the skewness we use log transform on our price variable. and the we preproccessed the catalog_content to remove the puncutation and the converted into lower case for better normalisation.

**Key Observations:**

### 2.2 Solution Strategy

**Approach Type:** Single Model (Text-Based)
**Core Innovation:** Our primary strategy involves **full fine-tuning** of a pre-trained NeoBERT model (`chandar-lab/NeoBERT`) with a custom regression head for price prediction. To ensure stable and efficient training, the solution incorporates robust techniques like mixed-precision computation and gradient clipping.

---

## 3. Model Architecture

### 3.1 Architecture Overview
The model architecture consists of two main components:
1.  **NeoBERT Base:** A pre-trained `chandar-lab/NeoBERT` model processes the input product `catalog_content`.
2.  **Regression Head:** The `[CLS]` token embedding from the final hidden layer of NeoBERT is extracted and fed into a custom Multi-Layer Perceptron (MLP). This MLP head acts as a regressor to predict the final (log-transformed) price.

### 3.2 Model Components

**Text Processing Pipeline:**
- **Preprocessing steps:** 
    - The target `price` variable is transformed using `log1p` for more stable training.
    - Product `catalog_content` is tokenized using the NeoBERT tokenizer with a maximum length of 512 tokens.
- **Model type:** `chandar-lab/NeoBERT` with a custom MLP regression head attached.
- **Key parameters:**
    - **Loss Function:** A custom SMAPE loss that correctly reverses the log-transformation on predictions and true values before calculation.
    - **Optimizer:** AdamW.
    - **Training:** Utilizes mixed-precision training (`torch.amp`) and gradient clipping (max norm 1.0) for faster and more stable training.
- **Fine-tuning Strategy:** The script supports three distinct fine-tuning strategies:
    1.  **Full Fine-tuning:** The entire model, including the base NeoBERT weights and the regression head, is trained end-to-end.
    

**Image Processing Pipeline:**
not used as the in our eda we found out that the image quality was very bad. enhancing images would require a lot a computation and time so we did not waste much time with that.
---


## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** 0.43
- **Other Metrics:** [MAE, RMSE, R² if calculated] not calc


## 5. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*
the first approach was using openai clip model to generate emmbedding for text and image and fusing them using ann regressor head which gave us initial smape 55. then we tried using qwen embeddings and use xgboost and ann regressor for regression head but didnt give good results.and also tried diberta but the results were not good either after enough experimentation and research finally i trained a neobert model to generate text embeddings and used a regressor head performing full finetuning and gave us smape score of 43. h100 is so powerful that it was bottling so we spent a chunk of time optimising that and finally landed on mixed precision.
---

## Appendix

### A. Code artefacts
*Include drive link for your complete code directory*




---

**Note:** This is a suggested template structure. Teams can modify and adapt the sections according to their specific solution approach while maintaining clarity and technical depth. Focus on highlighting the most important aspects of your solution.