# Fine-tuning for inflammatory phenotypes

Fine-tune all NMR encoder layers and a composition-modulated ensemble to predict concurrent CRP and absolute neutrophil, monocyte, lymphocyte and eosinophil counts. [Training loop](train.py) · [Readout architecture](readout.py)

## Train

Run from the repository root after pretraining. Use the same participants, row order and split assignments. Add the four clinical columns (age, sex, BMI and smoking) and the five outcome columns. [Pretraining](../pretraining/) · [Cohort format](../docs/data.md)

```bash
python finetuning/train.py \
  --data data/cohort.csv \
  --encoder runs/pretrain/best.pt \
  --device cuda:0 \
  --out runs/finetune
```

The installed command `mi-nmr finetune` accepts the same options. The selected model is saved to `runs/finetune/best.pt`. Add `--resume` to resume at the last completed epoch.

## Readout and model selection

The participant vector and seven clinical channels feed a two-block, 32-member ensemble. An additional branch uses 42 lipid-composition coordinates (three per subclass) to scale each member's hidden features between the first and second blocks. Missing subclass compositions take their training centers. Predictions are averaged across ensemble members. [TabM backbone](https://github.com/yandex-research/tabm)

All encoder layers are fine-tuned at 2e-6 and the readout at 5e-4. The loss is member-wise MSE on training-standardized log CRP and log1p cell counts. Validation MSE of the averaged predictions selects the checkpoint, with at most 40 epochs and patience 10. Evaluation reports per-phenotype transformed-scale RMSE, standardized RMSE and Spearman correlation, together with the five-task MSE and mean standardized RMSE.

## Held-out evaluation

Evaluate the selected model on the participants assigned to the downstream test split.

```bash
mi-nmr evaluate \
  --data data/cohort.csv \
  --checkpoint runs/finetune/best.pt \
  --split test \
  --device cuda:0 \
  --out runs/metrics.json
```

## Phenotype predictions

For new participants, supply the 107 NMR measurements and four clinical columns. The output CSV contains participant IDs and predictions on the ln(CRP) and log1p(cell count) scales.

```bash
mi-nmr predict \
  --data data/new_participants.csv \
  --checkpoint runs/finetune/best.pt \
  --device cuda:0 \
  --out runs/predictions.csv
```
