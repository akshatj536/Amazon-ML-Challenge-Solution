#!/bin/bash

# This script runs the training for DeBERTaV3-base and the precomputed regressor.

echo "--- Running DeBERTaV3-base training ---"
python src/DeBERTaV3-base.py

if [ $? -ne 0 ]; then
    echo "DeBERTaV3-base.py failed. Aborting."
    exit 1
fi

echo "--- Running NeoBERT training ---"
python src/train_text_only_model.py

if [ $? -ne 0 ]; then
    echo "train_text_only_model.py failed. Aborting."
    exit 1
fi

echo "--- All training scripts executed successfully ---"
