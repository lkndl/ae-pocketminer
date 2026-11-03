# AE-PocketMiner — Cryptic Pocket & Allosteric Coupling Prediction

An AI method for simultaneously predicting cryptic binding pockets and their 
allosteric coupling to the rest of the protein from a single input structure,
using an attention-enabled Geometric Vector Perceptron graph neural network.

<p align="center">
  <img src="docs/imgs/model_archt.png" width="400" alt="AE-PocketMiner model architecture"/>
  <br/>
  <em>Figure 1. AE-PocketMiner model architecture.</em>
</p>

---

## Table of contents

- [Installation](#installation)
- [Local prediction](#local-prediction)
- [Training](#training)
- [Training data](#training-data)
- [Background](#background)
- [Citation](#citation)
- [License](#license)

---

## Installation

Install the `ae-pocketminer` command with [uv](https://docs.astral.sh/uv/). It
comes with the weights of both models, runs on the CPU, and works on Linux,
macOS and Windows with Python 3.10 to 3.13.

```bash
uv tool install git+https://github.com/lkndl/ae-pocketminer@<tag>

# or from the wheel attached to a release, which saves cloning the repository
uv tool install https://github.com/lkndl/ae-pocketminer/releases/download/<tag>/ae_pocketminer-<version>-py3-none-any.whl
```

To work on a checkout, `uv sync` sets up the environment and `uv run pytest`
runs the tests; see [docs/development.md](docs/development.md). Training needs
more, see [Training](#training).

---

## Local prediction

For most users, we recommend trying the [PocketMiner web interface](https://pocketminer.azurewebsites.net/) first — no installation required. *(**Note**: this currently serves the PocketMiner model; a version pointing to AE-PocketMiner is coming soon.)*

To run predictions locally instead:

```bash
ae-pocketminer your_protein.pdb -o results -j
```

This runs AE-PocketMiner and PocketMiner (`--model` picks one) on all protein
chains of the first model in the file together, and writes to `results/`:
- `your_protein_ae-pocketminer.tsv` — the per-residue pocket probabilities of
  both models, by chain, residue number and insertion code
- with `-j`: `your_protein_ae-pocketminer.json` and
  `your_protein_pocketminer.json` — the same, as one
  [residue score track](https://github.com/HelmholtzAI-Consultants-Munich/af-molstar-viewer/blob/main/docs/residue-score-track.md)
  per model
- with `--attention`: `your_protein_ae-pocketminer_attention.npz` — the
  attention weights of AE-PocketMiner (heads x N x N), with the residue of
  each row

The inputs can be PDB or mmCIF files, also gzipped, or directories of them.
Residues are named by their author chain and number. Modified residues count
as their parent amino acid (MSE as MET). A residue that lacks N, CA, C or O
is left out of the model input and is empty (null) in the outputs. When an
input fails, the others are still predicted and the command exits with
status 1.

To write a PDB with B-factors set to pocket probability (for PyMOL
visualisation), from a checkout:

```bash
uv run python src/write_bfactor_pdb.py your_protein.pdb results/your_protein_ae-pocketminer.json
```

To get allosteric predictions, provide the residues of a cryptic pocket (known or predicted — e.g., `A:10 A:14 A:15`) and run the commands below. This outputs 1) the top 40 residues influencing this cryptic pocket, and 2) a PDB with B-factors set to attention weights relative to this cryptic pocket (for PyMOL visualisation):

```bash
ae-pocketminer your_protein.pdb -o results --attention
uv run python src/find_key_attention_residues.py \
    --pdb your_protein.pdb \
    --attention-weights results/your_protein_ae-pocketminer_attention.npz \
    --pocket-resnums A:10 A:14 A:15 \
    --top-n 40 \
    --output-pdb results/your_protein-attn_Bfactor.pdb
```

The key **practical difference** between the two models is that PocketMiner
predicts per-residue cryptic pocket probabilities only, with no allosteric
predictions: it has no attention weights.

---

## Training

The training scripts in `src/` need the `train` dependency group
(`uv sync --group train`) and import the model from `src/ae_pocketminer`.
They have not been run since the models moved to Keras 2 (`tf-keras`) and
TensorFlow 2.21. `environment.yml`, the authors' original GPU environment with
TensorFlow 2.10, has no `tf-keras` and no longer fits.

AE-PocketMiner was sequentially trained on two types of simulation labels: those derived from LIGSITE, and those derived from fpocket. Please refer to our paper for more detail. If you decide to retrain the model or extend this work, you can follow the steps below.

**1. LIGSITE labels**

```bash
python src/train_xtal_predictor.py src/training-yaml-files/train_ligsite_bigdataset_attn.yaml
```

This trains the base model using LIGSITE-derived labels from sampled
structures. Training data arrays (X and y) are loaded from `data/task2/` as
numpy `.npy` files, following the same format as the original repo.
Using `train_ligsite_orignal.yaml` instead of `train_ligsite_bigdataset_attn.yaml`, 
you can retrain PocketMiner instead. (For convenience, we provide this script for 
both PocketMiner and AE-PocketMiner.)

**2. fpocket refinement**

```bash
python src/train_fpocket_drug_score_labels.py *.yaml
```

This fine-tunes the previous checkpoint using fpocket druggability score
labels, which incorporate both pocket geometry and chemical environment.

Both checkpoints are stored in `src/ae_pocketminer/weights/` — PocketMiner's is
copied over from the original repo, while AE-PocketMiner's is the one trained above.

---

## Training data

Training data are stored as numpy arrays under `data/`, using the same
conventions as the PocketMiner repository:

```
data/
  task2/
    X-train-*.npy        # structure/trajectory references for LIGSITE training
    y-train-*.npy        # LIGSITE pocket labels
    X-train-fpocket-*.npy   # structure references for fpocket training
    y-train-fpocket-*.npy   # fpocket druggability score labels
  pm-dataset/
    val*.npy / test*.npy    # validation and test dataset
```

---

## Background

AE-PocketMiner is a new model developed in the Bowman Lab that predicts both
cryptic binding pockets and allosteric coupling from a single protein
structure. It uses the GVP-GNN from PocketMiner as a protein embedding
backbone — chosen for its proven performance in pocket prediction — and adds
an attention mechanism to capture long-range residue–residue dependencies.
For full details on the architecture, training data, and benchmarking, see our
paper: https://doi.org/10.64898/2026.05.21.726899

The GVP source files in `src/` are adapted from the PocketMiner
codebase with modifications to incorporate the attention layers. For the
original model and GVP-GNN methodology, see:

**PocketMiner repository:**
👉 https://github.com/Mickdub/gvp/tree/pocket_pred

> Meller, A., Ward, M., Borowsky, J. *et al.* Predicting locations of cryptic 
> pockets from single protein structures using the PocketMiner graph neural 
> network. *Nature Communications*, 14, 1177 (2023).
> https://doi.org/10.1038/s41467-023-36699-3

---

## Citation

If you use AE-PocketMiner in your research, please cite:
https://doi.org/10.64898/2026.05.21.726899

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).
