# Polaris x ASAP Discovery x OpenADMET Antiviral Ligand Poses Challenge 2025

Prediction of antiviral ligand poses using [DiffDock-L](https://github.com/gcorso/DiffDock) for the Antiviral Ligand Poses Challenge 2025 hosted by Polaris. 

See the [report](./Report.pdf) for details on our approach.

### Installation

1. Create the environment:
```
conda env create -f config/environment.yaml
```
2. Activate environment:
```
conda activate polaris_diffdock
```
3. Install the required packages:
```
pip install -r requirements.txt
```

### Additions for the Challenge
This repository is a fork of [DiffDock](https://github.com/gcorso/DiffDock), with additional code files and scripts to support the Polaris Antiviral Ligand Poses Challenge.

* `config/`: Contains configuration files for running inference and evaluation.
* `notebooks/`: Jupyter notebooks for data exploration and visualization.
* `src/`: Contains the code added for running DiffDock-L on the given Polaris dataset.
* `scripts/`: Scripts for data processing and submission.

To run inference or evaluate predictions, use the `main.py` script and configure the YAML files in `/config` as needed.