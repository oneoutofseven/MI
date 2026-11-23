# MI

MI learns a 512-dimensional representation of each participant from 107 plasma NMR measurements. The encoder is then fine-tuned to estimate C-reactive protein (CRP) and four absolute blood cell counts measured at the same visit (neutrophils, monocytes, lymphocytes and eosinophils).

## Model

Seventy measurements describe 14 lipoprotein subclasses. Each subclass has a particle concentration and four lipid concentrations. During pretraining, a 12-layer Transformer learns from both randomly hidden fields and masks that hide an entire subclass. The decoder predicts a lipid subtotal and four fractions per subclass, whose products recover the individual lipid concentrations. The downstream ensemble combines the participant representation with clinical covariates and uses lipid composition to scale each member’s hidden features. [Architecture and training objectives](docs/training.md#pretraining)

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

Prepare a cohort CSV with one row per participant, the 107 named NMR measurements and the two split columns. [Input format and field list](docs/data.md)

```bash
mi-nmr pretrain \
  --data data/cohort.csv \
  --device cuda:0 \
  --out runs/pretrain
```

This runs the full 12-layer encoder with an effective batch size of 512 for up to 200 epochs. [Default training configuration](src/mi_nmr/recipe.json)

## Downstream analysis

Fine-tune the encoder and five-phenotype readout jointly, evaluate on held-out participants or extract a participant embedding for another analysis. [Downstream workflow](docs/training.md#fine-tuning-and-evaluation)
