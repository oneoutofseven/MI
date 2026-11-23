# Training and evaluation

## Pretraining

The pretraining directory contains the GPU training script, masking plans, reconstruction model and losses. [Pretraining guide](../pretraining/)

## Fine-tuning and evaluation

Use the selected pretrained encoder to initialize the five-phenotype model, then evaluate on the downstream test split. [Fine-tuning guide](../finetuning/) · [Held-out evaluation](../finetuning/#held-out-evaluation)

## Embeddings and phenotype predictions

[Extract participant embeddings](../pretraining/#extract-participant-embeddings) · [Predict the five phenotypes](../finetuning/#phenotype-predictions)
