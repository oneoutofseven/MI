# NMR pretraining

Train a 12-layer field Transformer on 107 plasma NMR measurements to learn a 512-dimensional representation for each participant. The targets are observed measurements hidden during training. [Participant encoder](../src/mi_nmr/encoder.py) · [Reconstruction model](model.py)

## Train

After installation, run the command from the repository root. Supply the participant ID, the 107 NMR columns and the two stage-specific split columns. [Installation](../README.md#installation) · [Cohort format](../docs/data.md)

```bash
python pretraining/train.py \
  --data data/cohort.csv \
  --device cuda:0 \
  --out runs/pretrain
```

The installed command `mi-nmr pretrain` accepts the same options. Validation reconstruction MAE selects `runs/pretrain/best.pt`, the input to fine-tuning. Use the same command with `--resume` to continue from the last completed epoch. [Training loop](train.py) · [Continue to fine-tuning](../finetuning/)

## Masking and reconstruction

Each time a participant is sampled, one mask is drawn from a bank containing equal numbers of random-field and subclass-first masks. Each plan hides 15–50% of the 107 positions before intersecting with measured values. A subclass-first plan selects all five fields of one lipoprotein subclass, then draws from the remaining subclasses and single fields to fill its budget. Naturally missing measurements have their own token and contribute no reconstruction targets.

The participant decoder predicts particle concentration, a four-lipid subtotal and four fractions for each subclass. Multiplying the subtotal by each fraction recovers its lipid concentration. A shared linear head additionally predicts the hidden values from the final field states. The objective adds three terms, each averaged over eligible participants.

| Term | Target | Coefficient |
| --- | --- | ---: |
| Participant reconstruction | Hidden NMR values, using MAE on standardized measurements | 1.0 |
| Field reconstruction | The same hidden values, predicted from field states | 0.2 |
| Subclass composition | Absolute errors in three standardized Helmert contrasts of the four lipid fractions | 0.2 |

A subclass qualifies for the composition loss when all four lipids were measured, are nonnegative, have a positive subtotal and are all artificially hidden. Checkpoint selection uses mean participant reconstruction MAE across fixed random 15%, 30% and 50% validation masks.

The default uses AdamW with a peak learning rate of 1e-4, five warm-up epochs and cosine decay. Training runs for up to 200 epochs with patience 20 and a 30-epoch minimum. To compare masking schemes with the same architecture, copy the default configuration and set `mask` to `ordinary` for randomized masks with matched plan sizes and per-field counts, or to `subclass` for exclusively subclass-first masks with matched plan sizes. Set `field_weight` to zero to remove the auxiliary field loss. [Default configuration](../src/mi_nmr/recipe.json)

[Mask construction](masks.py) · [Loss computation](objectives.py)

## Extract participant embeddings

For a new cohort, supply participant IDs and the same 107 named NMR fields.

```bash
mi-nmr encode \
  --data data/new_participants.csv \
  --checkpoint runs/pretrain/best.pt \
  --device cuda:0 \
  --out runs/embeddings.npz
```

The NPZ file contains `id` and `embedding`, with one 512-dimensional vector per participant under the default configuration.
