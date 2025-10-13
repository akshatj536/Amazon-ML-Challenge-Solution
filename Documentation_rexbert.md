# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** Kasukabe-Defence-Group-Solution 
**Team Members:** akshit, akshat, anirudha, somil 
**Submission Date:** 13 october 

---

## 1. Executive Summary
Performed Full Fine Tuning, Using RexBERT, we didn't use the LoRA or anything else, just Full Fine Tuning with a regressor head.


---

## 2. Methodology Overview

### 2.1 Problem Analysis
We perform EDA and found out that price data was skewed so we removed all the data points above 145 dollar price. also to incorporate the skewness we use log transform on our price variable. and the we preproccessed the catalog_content to remove the puncutation and the converted into lower case for better normalisation.

**Key Observations:**

### 2.2 Solution Strategy

**Approach Type:** Single Model (Text-Based)
**Core Innovation:** Our primary strategy involves **full fine-tuning** of a pre-trained RexBERT model (`thebajajra/RexBERT-large`) with a custom regression head for price prediction. To ensure stable and efficient training, the solution incorporates robust techniques like mixed-precision computation and gradient clipping.

---

## 3. Model Architecture

### 3.1 Architecture Overview
The model architecture consists of two main components:
1.  **NeoBERT Base:** A pre-trained `thebajajra/RexBERT-large` model processes the input product `catalog_content`.
2.  **Regression Head:** The `[CLS]` token embedding from the final hidden layer of NeoBERT is extracted and fed into a custom Multi-Layer Perceptron (MLP). This MLP head acts as a regressor to predict the final (log-transformed) price.

### 3.2 Model Components

**Text Processing Pipeline:**
- **Preprocessing steps:** 
    - The target `price` variable is transformed using `log1p` for more stable training.
    - Product `catalog_content` is tokenized using the NeoBERT tokenizer with a maximum length of 512 tokens.
- **Model type:** `thebajajra/RexBERT-large` with a custom MLP regression head attached.
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
- **SMAPE Score:** [your best validation SMAPE]
- **Other Metrics:** [MAE, RMSE, R² if calculated]


## 5. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code artefacts
*Include drive link for your complete code directory*


### B. Additional Results
*Include any additional charts, graphs, or detailed results*

---

**Note:** This is a suggested template structure. Teams can modify and adapt the sections according to their specific solution approach while maintaining clarity and technical depth. Focus on highlighting the most important aspects of your solution.