# MI

MI learns a participant representation from 107 baseline plasma NMR measurements and fine-tunes it to estimate five concurrent inflammatory phenotypes. The package includes the field schema, preprocessing, pretraining, supervised readout and held-out evaluation. The worked example uses generated data.

## Model

1. **Measurements and masking.** Seventy fields describe 14 lipoprotein subclasses. Each has a particle concentration and four lipid concentrations (PL, CE, FC, TG). Another 37 fields describe other NMR measurements. The tokenizer combines numerical values, field identities and measurement metadata. The default bank contains equal numbers of random-field and subclass-first plans. Each participant draws one plan, hiding 15–50% of the 107 positions before intersecting with actually observed measurements. A subclass-first plan takes one whole subclass, then samples additional units to fill its budget. True missing values have their own token.
2. **Pretraining.** A 12-block field Transformer (width 768, 12 attention heads) produces a 512-dimensional participant vector. A decoder uses that vector to predict particle concentration, a four-lipid subtotal and four fractions per subclass. Their product recovers lipid concentrations. A shared linear head on the final field states also predicts the hidden values. The loss is participant-level hidden-value MAE plus 0.2 times field-level MAE plus 0.2 times the composition error. The composition term uses three standardized Helmert contrasts of raw fractions when all four lipids are measured, nonnegative and artificially hidden and their subtotal is positive. Each term is averaged over its eligible participants.
3. **Phenotype readout.** The participant vector and seven clinical covariate channels feed a two-block, 32-member TabM ensemble. A separate branch encodes 42 composition coordinates (three per subclass). It multiplies the first block's member states by a learned bounded adjustment around one. An undefined composition contributes its training center for that subclass. The encoder and readout train jointly with five standardized outputs. At inference, member predictions are averaged.

The measurement order, units and numerical identifiers are in [schema.json](src/mi_nmr/schema.json). The study configuration is in [recipe.json](src/mi_nmr/recipe.json). The ensemble backbone uses the upstream [TabM implementation](https://github.com/yandex-research/tabm).

## Installation and a CPU example

In a Python 3.10+ environment on Linux or macOS, run the following from the repository root. For GPU training install a CUDA-compatible PyTorch build.

~~~bash
python -m pip install -e '.[test]'
python examples/make_demo.py work/cohort.csv
mi-nmr pretrain --data work/cohort.csv --config examples/small.json --device cpu --out work/pretrain
mi-nmr finetune --data work/cohort.csv --encoder work/pretrain/best.pt --config examples/small.json --device cpu --out work/finetune
mi-nmr evaluate --data work/cohort.csv --checkpoint work/finetune/best.pt --split test --device cpu --out work/metrics.json
mi-nmr encode --data work/cohort.csv --checkpoint work/pretrain/best.pt --device cpu --out work/embeddings.npz
mi-nmr predict --data work/cohort.csv --checkpoint work/finetune/best.pt --device cpu --out work/predictions.csv
python -m pytest
~~~

The example uses a two-block encoder, a 16-dimensional participant embedding and two epochs per stage. To run the study architecture, omit the two occurrences of <code>--config examples/small.json</code> and select a device such as <code>--device cuda:0</code>. Both training commands save <code>best.pt</code>, an epoch-boundary <code>last.pt</code> and validation history. Add <code>--resume</code> to the same training command to continue an interrupted run.

## Input

Supply one UTF-8 CSV row per participant. NMR columns can appear in any order. Use the 107 exact field headers and original units listed in the schema. Empty measurement cells denote missing values. This recipe uses the original baseline alanine field <code>p23460_i0</code>. Standardization uses observed pretraining-train measurements and sample standard deviations. Readout calibration uses only supervised training rows.

| Column | Contents |
|---|---|
| <code>id</code> | Unique participant identifier, treated as text |
| 107 NMR columns | UKB-style headers such as <code>p23404_i0</code> |
| <code>pretrain_split</code>, <code>finetune_split</code> | Explicit <code>train</code>, <code>val</code>, <code>test</code> or <code>unused</code> per stage. Downstream test participants must be outside pretraining training and validation. Use the same row order and split manifest when fitting both stages and evaluating the holdout. |
| <code>age</code>, <code>male</code>, <code>bmi</code>, <code>smoking</code> | Age in years, male sex coded 0/1, BMI in kg/m² and smoking coded <code>never</code>, <code>previous</code>, <code>current</code> or <code>unknown</code>. Empty smoking is treated as unknown. Missing BMI uses the training mean and a separate indicator. |
| <code>CRP</code>, <code>neutrophils</code>, <code>monocytes</code>, <code>lymphocytes</code>, <code>eosinophils</code> | Baseline CRP in mg/L, followed by absolute cell counts in 10⁹/L. Corresponding UKB field IDs in order are 30710, 30140, 30130, 30120 and 30150. |

Pretraining reads the ID, splits and NMR columns. New participants for <code>encode</code> need only the ID and the NMR columns. <code>predict</code> additionally needs the four clinical columns. Keep private input tables, weights and participant-level outputs in a local working directory.

## Training and metrics

- Pretraining uses AdamW with a peak learning rate of 10⁻⁴, a five-epoch warm-up and cosine decay. The maximum is 200 epochs with patience 20 and a 30-epoch minimum. Checkpoint selection uses the mean participant reconstruction MAE across fixed random 15%, 30% and 50% validation masks.
- Fine-tuning uses all encoder layers at 2 × 10⁻⁶ and the readout at 5 × 10⁻⁴. Training minimizes the member-wise MSE of training-standardized log CRP and log1p cell counts. The validation MSE of the averaged predictions selects the epoch, with a maximum of 40 epochs and patience 10.
- Evaluation reports per-phenotype transformed-scale RMSE, training-standardized RMSE and Spearman correlation, as well as five-task MSE and the mean of the five standardized RMSEs. Prediction CSVs use ln(mg/L) for CRP and log1p(count) for each cell type. Embedding files contain <code>id</code> and <code>embedding</code>.
- To compare masking schemes, copy the full recipe and change <code>mask</code> to <code>ordinary</code> (randomized with matched row and field totals) or <code>subclass</code> (100% subclass-first with matched budgets). Set <code>field_weight</code> to zero to remove the field-level supervision. Keep the other settings fixed and give each fit its own output directory.

The inference commands also accept an existing <code>nmr_measure_pretraining_v1</code> participant export. For a corresponding older five-output multiplicative readout, pass its clinical checkpoint to <code>predict</code> or <code>evaluate</code> and the matching participant export via <code>--encoder</code>. The loader verifies the checkpoint pair before loading parameters.
