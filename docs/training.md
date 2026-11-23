# Training and evaluation

## Pretraining

Run the GPU pretraining command from the [README](../README.md#pretrain-on-a-gpu) on your cohort. The default encoder has 12 Transformer blocks of width 768 with 12 attention heads and produces one 512-dimensional vector per participant.

Each time a participant is sampled, one mask is drawn from a bank containing equal numbers of random-field and subclass-first masks. Each plan hides 15–50% of the 107 positions before intersecting with measured values. A subclass-first plan selects all five fields of one lipoprotein subclass, then draws from the remaining subclasses and single fields to fill its budget. Naturally missing measurements have their own token and contribute no reconstruction targets.

The participant decoder predicts particle concentration, a four-lipid subtotal and four fractions for each subclass. Multiplying the subtotal by each fraction recovers its lipid concentration. A shared linear head additionally predicts the hidden values from the final field states. The objective adds three terms, each averaged over eligible participants.

| Term | Target | Coefficient |
| --- | --- | ---: |
| Participant reconstruction | Hidden NMR values, using MAE on standardized measurements | 1.0 |
| Field reconstruction | The same hidden values, predicted from field states | 0.2 |
| Subclass composition | Absolute errors in three standardized Helmert contrasts of the four lipid fractions | 0.2 |

A subclass qualifies for the composition loss when all four lipids were measured, are nonnegative, have a positive subtotal and are all artificially hidden. Checkpoint selection uses mean participant reconstruction MAE across fixed random 15%, 30% and 50% validation masks.

The default uses AdamW with a peak learning rate of 1e-4, five warm-up epochs and cosine decay. Training runs for up to 200 epochs with patience 20 and a 30-epoch minimum. To compare masking schemes with the same architecture, copy the default configuration and set `mask` to `ordinary` for randomized masks with matched plan sizes and per-field counts, or to `subclass` for exclusively subclass-first masks with matched plan sizes. Set `field_weight` to zero to remove the auxiliary field loss. [Default configuration](../src/mi_nmr/recipe.json)

## Fine-tuning and evaluation

After pretraining on your cohort, fit the five inflammatory phenotypes and evaluate the selected model on the test split.

```bash
mi-nmr finetune \
  --data data/cohort.csv \
  --encoder runs/pretrain/best.pt \
  --device cuda:0 \
  --out runs/finetune

mi-nmr evaluate \
  --data data/cohort.csv \
  --checkpoint runs/finetune/best.pt \
  --split test \
  --device cuda:0 \
  --out runs/metrics.json
```

The participant vector and seven clinical channels feed a two-block, 32-member ensemble. An additional branch uses 42 lipid-composition coordinates (three per subclass) to scale each member's hidden features between the first and second blocks. Missing subclass compositions take their training centers. Predictions are averaged across ensemble members. [TabM backbone](https://github.com/yandex-research/tabm)

All encoder layers are fine-tuned at 2e-6 and the readout at 5e-4. The loss is member-wise MSE on training-standardized log CRP and log1p cell counts. Validation MSE of the averaged predictions selects the checkpoint, with at most 40 epochs and patience 10. Evaluation reports per-phenotype transformed-scale RMSE, standardized RMSE and Spearman correlation, together with the five-task MSE and mean standardized RMSE.

For both training stages, `best.pt` contains the validation-selected model and `last.pt` supports resuming at the last completed epoch. To continue an interrupted run, repeat its training command with `--resume`.

## Embeddings and phenotype predictions

Use your trained encoder to extract one vector per participant. Use the fine-tuned model for five phenotype predictions. [Required columns for new participants](data.md)

```bash
mi-nmr encode \
  --data data/new_participants.csv \
  --checkpoint runs/pretrain/best.pt \
  --device cuda:0 \
  --out runs/embeddings.npz

mi-nmr predict \
  --data data/new_participants.csv \
  --checkpoint runs/finetune/best.pt \
  --device cuda:0 \
  --out runs/predictions.csv
```

The NPZ file contains `id` and `embedding`. The prediction CSV contains participant IDs and predictions on the ln(CRP) and log1p(cell count) scales.
