# Training and evaluation

## Pretraining

The pretraining directory contains the GPU training script, masking plans, reconstruction model and losses. [Pretraining guide](../src/mi_nmr/pretraining/)

## Fine-tuning and evaluation

Use the selected pretrained encoder to initialize the five-phenotype model, then evaluate on the downstream test split. [Fine-tuning guide](../src/mi_nmr/finetuning/) · [Held-out evaluation](../src/mi_nmr/finetuning/#held-out-evaluation)

## Embeddings and phenotype predictions

[Extract participant embeddings](../src/mi_nmr/pretraining/#extract-participant-embeddings) · [Predict the five phenotypes](../src/mi_nmr/finetuning/#phenotype-predictions)
