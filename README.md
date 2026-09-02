# Team submission to the Amazon ML Challenge 2025. Forked to my account from the team repo.




# Amazon ML Challenge: Product Price Prediction

This repository contains the code and experiments for our solution to the Amazon ML Challenge, where the goal was to predict product prices based on their catalog information.

## Dataset

The dataset for this challenge consists of product catalog information from Amazon. The primary data files used in our experiments are located in the `dataset/` directory.

- **Training Data:** `dataset/train_filtered.csv` and a cleaned version `dataset/train_filtered_cleaned.csv`.
- **Test Data:** `dataset/test.csv`.
- **Image Data:** For multimodal experiments, product images were sourced from the `train_images/` directory based on links in the dataset.

### Features
- **`catalog_content`**: The primary text feature, containing product descriptions, specifications, and other attributes.
- **`image_link`**: A URL to the product image, used in multimodal experiments.

### Target
- **`price`**: The product price, which is the target variable for our regression task.

### Preprocessing
- The `price` target is heavily skewed, so for most neural network models, we apply a log-transformation (`np.log1p`) to stabilize variance and improve model performance.
- The `catalog_content` text undergoes cleaning and is then tokenized to a fixed maximum length depending on the model's input size.

## Experiments

We conducted a series of experiments with different models and techniques to find the best approach for this regression task. All experiments were tracked using **MLflow** to log parameters, metrics, and model artifacts, ensuring reproducibility and easy comparison. Below is a summary of each experiment with more detailed parameters.

### 1. DeBERTaV3-base
- **File:** `src/DeBERTaV3-base.py`
- **Description:** This experiment involved fine-tuning the `microsoft/deberta-v3-base` model. A custom regressor head (1-layer MLP with width 256) was added on top of the base model to predict prices. The model was trained for 40 epochs with a batch size of 64, using the AdamW optimizer, a learning rate of `2e-5`, and a custom SMAPE loss function.

### 2. Gemma with Regressor Head
- **File:** `src/GemmaEmbedding.py`
- **Description:** We used `google/embeddinggemma-300m` as a feature extractor. The weights of the Gemma model were frozen, and only a custom 5-layer deep MLP regressor head (width 256) was trained on the embeddings to predict the price. This was trained for 20 epochs with a batch size of 64.

### 3. NeoBERT
- **File:** `src/NeoBERT.py`
- **Description:** This experiment involved fully fine-tuning the `chandar-lab/NeoBERT` model, which is pre-trained on product catalog data. It was trained for 20 epochs with a batch size of 64, an AdamW optimizer with a learning rate of `2e-5`, and a dropout of 0.4 in the regressor head.

### 4. NeoBERT + CLIP Fusion (Multimodal)
- **File:** `src/Neobert_CLIP_fusion.py`
- **Description:** A multimodal approach where we combined text and image data. Text embeddings were from `chandar-lab/NeoBERT`, and image embeddings from a frozen `openai/clip-vit-base-patch32`. These features were fused using a cross-attention mechanism. The model was trained for 12 epochs using `SmoothL1Loss`, an AdamW optimizer with a low learning rate of `1e-5`, and significant image augmentations (RandomResizedCrop, ColorJitter, etc.).

### 5. Qwen with LoRA
- **File:** `src/QwenLoRA.py`
- **Description:** We applied Low-Rank Adaptation (LoRA) to efficiently fine-tune the `Qwen/Qwen3-Embedding-0.6B` model. LoRA was configured with a rank (r) of 8 and alpha of 32, targeting all linear layers. Both the LoRA-adapted model and a regressor head were trained.

### 6. Qwen with Regressor Head
- **File:** `src/QwenRegressorHead.py`
- **Description:** We used the `Qwen/Qwen3-Embedding-0.6B` model with its weights frozen. Only a 2-layer deep MLP regressor head was trained on the mean-pooled embeddings from the base model.

### 7. Qwen with Precomputation and PyTorch Regressor
- **File:** `src/QwenRegressorPrecomputation.py`
- **Description:** This was a two-stage approach. First, we pre-computed text embeddings for the entire dataset using `Qwen/Qwen3-Embedding-0.6B`. Then, we used Optuna for hyperparameter optimization to find the best architecture and learning parameters for a PyTorch-based MLP regressor (with BatchNorm) which was then trained on the pre-computed embeddings.

### 8. Qwen with XGBoost
- **File:** `src/QwenXgBoost.py`
- **Description:** Similar to the previous experiment, we first generated embeddings using `Qwen/Qwen3-Embedding-0.6B`. An XGBoost regressor was then trained on these embeddings. Optuna was used to find the optimal hyperparameters for the XGBoost model, such as `n_estimators` and `learning_rate`.

### 9. RexBERT
- **File:** `src/RexBERT.py`
- **Description:** We fine-tuned `thebajajra/RexBERT-large`, another model pre-trained on e-commerce data. It was trained for 25 epochs with a large batch size of 100 and a learning rate of `2e-5`.

### 10. RexBERT Continued Fine-Tuning
- **File:** `src/RexBERT_continueFIneTuning.py`
- **Description:** This script was used to continue the fine-tuning of a previously trained RexBERT model from a checkpoint. The training continued for 3 more epochs with a very low learning rate of `1e-6` to allow for further convergence without destabilizing the learned weights.

## Final Approach and Result

After evaluating multiple models, we achieved our best performance by creating a weighted ensemble of predictions from our top-performing **NeoBERT**, **RexBERT**, and **DeBERTaV3-base** models. This ensemble approach secured us the **26th rank** on the public leaderboard of the Amazon ML Challenge.
