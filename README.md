# Spatiodynamic inference using vision-based generative modelling
We combine a generative deep learning framework with ABC to infer the underlying data-generating model and assess uncertainty in its parameter estimation.

Jun Won Park, Kangyu Zhao, Sanket Rane. \textit{Spatiodynamic Inference Using Vision-Based Generative Modelling}. ArXiv Preprint, 2025. https://arxiv.org/abs/2507.22256

## Main Frameworks

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

[Hydra](https://github.com/facebookresearch/hydra) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

<br>

## Project Structure

The directory structure of the project looks like this:

```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── experiment               <- Experiment configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
│
├── src                    <- Source code
│   ├── data                     <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── inference.py             <- Run viaABC
│   └── train.py                 <- Run training
```

<br>

## 🚀  Quickstart

To train a lotka-volterra system using TSMVAE, 

```bash
python src/train.py experiment=lotka
```
> **Note**: This looks at a configuration file defined in `configs/experiement/lotka.yaml` where the model and data configurations are defined.

> **Note**: You must defined your logger or set it to null. See [Experiment Tracking] below


To run viaABC using the trained model,
```bash
python src/inference.py inference=lotka
```
> **Note**: This looks at a configuration file defined in `configs/inference/lotka.yaml` where abc configurations are defined.

> **Note**: You must change the `run_folder_path` and `checkpoint_substr` to correctly load the saved trained model weights

> **Note**: The log and ABC results will be saved to a folder called inference_output which is created inside the `run_folder_path`


## Workflow

**Basic workflow**

1. Write your PyTorch Model Module according to the defined lightning module (see [models/TSMVAE/model.py](src/models/TSMVAE/model.py) for example). Your output from forward must be exactly the same.
2. Write your viaABC system and its config. (see [viaABC/systems.py](src/viaABC/systems.py) and [configs/system/lotka_volterra.yaml] for example).
3. Write your experiment config, containing paths to model, datamodule, and system. (see [configs/experiment/lotka.yaml](configs/experiment/lotka.yaml) for example.)
4. Run training with chosen experiment config:
   ```bash
   python src/train.py experiment=experiment_name.yaml
   ```
5. Run viaABC with chosen inference config:
   ```bash
   python src/inference.py inference=experiment_name.yaml
   ```

## Experiment Tracking

PyTorch Lightning supports many popular logging frameworks: [Weights&Biases](https://www.wandb.com/), [Neptune](https://neptune.ai/), [Comet](https://www.comet.ml/), [MLFlow](https://mlflow.org), [Tensorboard](https://www.tensorflow.org/tensorboard/).

These tools help you keep track of hyperparameters and output metrics and allow you to compare and visualize results. 

To use wandb or neptune, just change the `- logger:` line defined in `configs/train.yaml`. You can receive a free neptune/wandb account for academic research purposes.
The default is set to wand. You can also set it to null. 

## Installation

```bash
# clone project
git clone https://github.com/jp4474/viaABC.git
cd viaABC

# [OPTIONAL] create venv environment (only tested on 3.10.12)
python -m venv myenv
source myenv/bin/activate

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## TODO
1. cpp code for spatial2D is slow using pybind (~0.7s per iteration). Suggest using cpython. 

```bibtex
@misc{park2025spatiodynamicinferenceusingvisionbased,
      title={Spatiodynamic inference using vision-based generative modelling}, 
      author={Jun Won Park and Kangyu Zhao and Sanket Rane},
      year={2025},
      eprint={2507.22256},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2507.22256}, 
}
```

