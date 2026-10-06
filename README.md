# STAIR 

Instance-adaptive fusion and transfer learning with frozen protein language model (PLM) embeddings.

This project provides an implementation of STAIR for training and evaluation. It maps precomputed embeddings from six PLMs into a shared task space, combines predictions through cross-expert semantic reassembly and instance-level routing, and regularizes representation adaptation with a JMMD–HSIC objective. It supports five protein property regression tasks and one subcellular localization classification task.

Associated paper: **STAIR: Selective Protein Language Model Embedding Fusion via Adaptive Routing**

**Before running: prepare precomputed embeddings from all six PLMs. This package includes datasets, configurations, and downstream model code, but does not include embeddings, PLM weights, embedding extraction scripts, or trained STAIR checkpoints.**

## Method Overview

A PLM's usefulness varies across tasks and proteins. STAIR learns source contributions for each protein instead of selecting one fixed embedding combination for an entire task.

1. **Source-specific adaptation:** apply BatchNorm to the concatenated embeddings, split them by source, and map each source into a shared hidden space using a `Linear → LayerNorm → ReLU` adapter.
2. **Cross-expert semantic reassembly:** apply all prediction heads to each adapted source feature and combine their predictions according to cosine similarity between the source feature and the heads' hidden representations. During the first `floor(epochs / 10)` training epochs, each source uses only its own prediction head. Validation and inference always use full reassembly.
3. **Instance-adaptive routing:** construct a global query from all adapted features, compare it with each source key, and obtain softmax routing weights. Retain the top-k sources, renormalize their weights, and aggregate their predictions. Training uses a straight-through estimator for routing gradients.
4. **Representation transfer:** construct a source-proxy view with a factorized projector. Label-weighted JMMD aligns this view with adapted features, while a negative HSIC term encourages statistical dependence between adapted features and labels.



## Project Structure

```text
STAIR_minimal/
├── README.md
├── requirements.txt
├── configs/
│   ├── aav.yaml
│   ├── gb1.yaml
│   ├── gfp.yaml
│   ├── location.yaml
│   ├── meltome.yaml
│   └── stability.yaml
├── src/
│   ├── data.py           # Label parsing, six-source embeddings, and DataLoader
│   ├── model.py          # Adapters, prediction heads, reassembly, and routing
│   ├── losses.py         # Task loss, auxiliary supervision, JMMD, and HSIC
│   └── train.py          # Training, checkpoint loading, evaluation, and export
└── data/
    ├── aav/
    ├── gb1/
    ├── gfp/
    ├── location/
    ├── meltome/
    ├── stability/
    ├── cath/             # Extension data; no experiment entry point in this package
    └── ppi/              # Extension data; no experiment entry point in this package
```

Prepare `embeddings/` separately. The program creates `outputs/` when it runs.

## Supported Tasks

Sample counts below come from the label TSV files in this package. Each test split is determined by `directories.test` in the corresponding YAML configuration.

| CLI Name | Task | Type / Metric | Train | Validation | Test |
|---|---|---|---:|---:|---:|
| `aav` | AAV variant fitness prediction | Regression / Spearman ρ | 28,626 | 3,181 | 50,776 |
| `gb1` | GB1 variant fitness prediction | Regression / Spearman ρ | 2,691 | 299 | 5,743 |
| `gfp` | GFP fluorescence prediction | Regression / Spearman ρ | 21,446 | 5,362 | 27,217 |
| `location` | 10-class subcellular localization | Classification / Accuracy | 9,503 | 1,678 | 490 |
| `meltome` | Protein melting temperature prediction | Regression / Spearman ρ | 22,335 | 2,482 | 3,134 |
| `stability` | Protein stability prediction | Regression / Spearman ρ | 53,614 | 2,512 | 12,851 |


## Installation

Run the following commands from the project root. `requirements.txt` pins these dependencies:

```text
torch==2.2.2
numpy==1.26.4
scipy==1.15.3
PyYAML==6.0.3
```




### Embedding Sources and Dimensions

Source order and dimensions are fixed by `SOURCES` in `src/data.py`. The concatenated embedding dimension is **5248** per protein.

| PLM Source | Embedding Directory | Vector Dimension |
|---|---|---:|
| ESM3 | `esm3` | 1536 |
| OntoProtein | `ontoprotein` | 1024 |
| ProteinCLIP | `proteinclip_t5` | 128 |
| ProtST | `protst` | 512 |
| ProTrek | `protrek` | 1024 |
| ProteinDT | `proteindt` | 1024 |

Each task requires six files with the following path structure:

```text
<embeddings-dir>/<dataset>/<source>/protein_dictionary.pt
```

For AAV:

```text
embeddings/
└── aav/
    ├── esm3/protein_dictionary.pt
    ├── ontoprotein/protein_dictionary.pt
    ├── proteinclip_t5/protein_dictionary.pt
    ├── protst/protein_dictionary.pt
    ├── protrek/protein_dictionary.pt
    └── proteindt/protein_dictionary.pt
```



## Training

```text
python train.py --mode train --dataset aav --embeddings-dir embeddings --device cpu --seed 42
```
