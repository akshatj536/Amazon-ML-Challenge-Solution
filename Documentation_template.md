# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** Kasukabe-Defence-Group-Solution 
**Team Members:** akshit, akshat, anirudha, somil 
**Submission Date:** 13 october

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*



---

## 2. Methodology Overview

### 2.1 Problem Analysis
*Describe how you interpreted the pricing challenge and key insights discovered during EDA.*

**Key Observations:**

### 2.2 Solution Strategy
*Outline your high-level approach (e.g., multimodal learning, ensemble methods, etc.)*

**Approach Type:** Single Model (Text-Based)
**Core Innovation:** Fine-tuning a pre-trained NeoBERT model (`chandar-lab/NeoBERT`) with a custom regression head for price prediction. The solution incorporates parameter-efficient fine-tuning (PEFT) using LoRA and robust training techniques like mixed-precision and gradient clipping.

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
    2.  **PEFT with LoRA:** Low-Rank Adaptation is applied to the "query", "key", and "value" modules of the attention layers. This significantly reduces the number of trainable parameters, making fine-tuning more efficient.
    3.  **Feature Extraction (Frozen):** The base NeoBERT model's weights are frozen. Only the weights of the custom regression head are trained.

**Image Processing Pipeline:**
not used
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