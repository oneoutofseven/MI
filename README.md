# MI

MI learns a 512-dimensional representation of each participant from 107 plasma NMR measurements. The encoder is then fine-tuned to estimate C-reactive protein (CRP) and four absolute blood cell counts measured at the same visit (neutrophils, monocytes, lymphocytes and eosinophils).

## Model

Seventy measurements describe 14 lipoprotein subclasses. Each subclass has a particle concentration and four lipid concentrations. During pretraining, a 12-layer Transformer learns from both randomly hidden fields and masks that hide an entire subclass. The decoder predicts a lipid subtotal and four fractions per subclass, whose products recover the individual lipid concentrations. The downstream ensemble combines the participant representation with clinical covariates and uses lipid composition to scale each member’s hidden features. [Pretraining](src/mi_nmr/pretraining/) · [Fine-tuning](src/mi_nmr/finetuning/) · [Shared encoder and data processing](src/mi_nmr/shared/)

## Installation

The GPU workflow has been tested on an NVIDIA RTX A6000 with Python 3.10 and PyTorch 2.6.0 (CUDA 12.4).

```bash
git clone https://github.com/oneoutofseven/MI.git
cd MI
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

For a different CUDA version, select the corresponding PyTorch build. [PyTorch installation options](https://pytorch.org/get-started/previous-versions/#v260)

## Pretrain on a GPU

Use a cohort CSV with the 107 NMR fields and split assignments. [Pretraining code and guide](src/mi_nmr/pretraining/) · [Input format](docs/data.md)

```bash
mi-nmr pretrain \
  --data data/cohort.csv \
  --device cuda:0 \
  --out runs/pretrain
```

The 12-layer encoder trains for up to 200 epochs with an effective batch size of 512. The checkpoint with the lowest validation reconstruction MAE is saved to `runs/pretrain/best.pt`. [Default training configuration](src/mi_nmr/shared/recipe.json)

## Fine-tune on a GPU

Use the same cohort CSV and split assignments. For supervised training, include age, sex, BMI, smoking status and the five phenotype columns. [Fine-tuning code and guide](src/mi_nmr/finetuning/)

```bash
mi-nmr finetune \
  --data data/cohort.csv \
  --encoder runs/pretrain/best.pt \
  --device cuda:0 \
  --out runs/finetune
```

All encoder layers and the five-output readout are updated jointly, with learning rates of 2e-6 and 5e-4, respectively. Fine-tuning runs for up to 40 epochs and stops after 10 epochs without improvement. The lowest five-task validation MSE selects `runs/finetune/best.pt`. [Held-out evaluation](src/mi_nmr/finetuning/#held-out-evaluation) · [Phenotype predictions](src/mi_nmr/finetuning/#phenotype-predictions)
