# Cohort format

Use one row per participant in a UTF-8 CSV. The loader aligns NMR fields by header, so the column order can vary. Supply concentrations in the original units and leave missing measurements empty. The 107-field schema includes the original baseline alanine field `p23460_i0`. [Field names, units and subclass membership](../src/mi_nmr/shared/schema.json)

| Columns | Values |
| --- | --- |
| `id` | Unique participant identifier, read as text. |
| 107 named NMR fields | Use the exact UKB-style headers in the schema, such as `p23404_i0`. |
| `pretrain_split`, `finetune_split` | Set each role to `train`, `val`, `test` or `unused`. Keep downstream test participants outside the pretraining training and validation partitions. Use the same participant order and split columns when fitting both stages and evaluating. |
| `age`, `male`, `bmi`, `smoking` | Age in years, a male-sex indicator (1 for male, 0 for female), BMI in kg/m² and smoking coded `never`, `previous`, `current` or `unknown`. Age and sex are required for the readout. An empty smoking value is treated as `unknown`. |
| `CRP`, `neutrophils`, `monocytes`, `lymphocytes`, `eosinophils` | Observed CRP in mg/L (positive) and absolute cell counts in 10⁹/L (nonnegative). Their UKB field IDs are 30710, 30140, 30130, 30120 and 30150, respectively. Leave missing outcomes empty. |

Pretraining uses the ID, split roles and NMR measurements. Each NMR field is standardized with the mean and sample standard deviation of its observed values in the pretraining training set. Fine-tuning additionally uses the four clinical columns and five outcomes. Each supervised training row needs at least one observed outcome. The seven clinical channels comprise age, sex, BMI, a missing-BMI indicator and indicators for previous, current and unknown smoking status. Missing BMI is filled with the supervised training mean. Training-set statistics are reused at evaluation.

Embedding extraction for new participants requires only `id` and the 107 NMR fields. Phenotype prediction additionally requires the four clinical columns.

## Example input

Generate a CSV with simulated measurements, covariates, outcomes and both split columns from the repository root.

```bash
python examples/make_demo.py work/cohort.csv
```
