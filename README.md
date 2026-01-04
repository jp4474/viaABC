# Spatiodynamic Inference Using Vision-Based Generative Modelling

We combine a generative deep learning framework with ABC to infer the underlying data-generating model and assess uncertainty in its parameter estimation.

Jun Won Park, Kangyu Zhao, Sanket Rane. *Spatiodynamic Inference Using Vision-Based Generative Modelling*. arXiv preprint, 2025. https://arxiv.org/abs/2507.22256

## Main Frameworks

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) — a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

[Hydra](https://github.com/facebookresearch/hydra) — a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

<br>

## Project Structure

The directory structure of the project looks like this:

```

├── .github                   <- GitHub Actions workflows
│
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callback configs
│   ├── data                     <- Data configs
│   ├── experiment               <- Experiment configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project path configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml                <- Main config for evaluation
│   └── train.yaml               <- Main config for training
│
├── data                      <- Project data
│
├── src                       <- Source code
│   ├── data                     <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── inference.py             <- Run viaABC
│   └── train.py                 <- Run training

````

<br>

## 🚀 Quickstart

To train a Lotka–Volterra system using TSMVAE:

First, simulate synthetic data using the script
```bash
python src/generate_training_data.py # Change line 44 to LotkaVolterra()
```

then run to start training

```bash
python src/train.py experiment=lotka
````

> **Note**: This uses a configuration file defined in `configs/experiment/lotka.yaml`, where the model and data configurations are specified.

> **Note**: You must define your logger or set it to `null`. See [Experiment Tracking] below.

> **Note**: You can set `kld_weight: 0` to use AE instead of VAE. This will automatically turn off, VAE related settings and losses. 

To run viaABC using the trained model:

```bash
python src/inference.py inference=lotka
```

> **Note**: This uses a configuration file defined in `configs/inference/lotka.yaml`, where ABC configurations are specified.

> **Note**: You must change `run_folder_path` and `checkpoint_substr` to correctly load the saved trained model weights.

> **Note**: Logs and ABC results will be saved to a folder called `inference_output`, which is created inside `run_folder_path`.

## Workflow

**Basic workflow**

1. Write your PyTorch model module according to the defined Lightning module (see `src/models/TSMVAE/model.py` for an example). The output of `forward` must match the expected format exactly.
2. Write your viaABC system and its configuration (see `src/viaABC/systems.py` and `configs/system/lotka_volterra.yaml` for examples).
3. Write your experiment config, containing paths to the model, datamodule, and system (see `configs/experiment/lotka.yaml` for an example).
4. Run `src/generate_training_data.py` to generate your train data.
5. Run training with the chosen experiment config:

   ```bash
   python src/train.py experiment=experiment_name.yaml
   ```
6. Run viaABC with the chosen inference config:

   ```bash
   python src/inference.py inference=experiment_name.yaml
   ```

## Experiment Tracking

PyTorch Lightning supports many popular logging frameworks: [Weights & Biases](https://www.wandb.com/), [Neptune](https://neptune.ai/), [Comet](https://www.comet.ml/), [MLflow](https://mlflow.org), and [TensorBoard](https://www.tensorflow.org/tensorboard/).

These tools help you keep track of hyperparameters and output metrics, and allow you to compare and visualize results.

To use Weights & Biases or Neptune, change the `- logger:` line defined in `configs/train.yaml`. You can obtain a free Neptune or Weights & Biases account for academic research purposes.
The default is set to `wandb`. You can also set it to `null`.

## Installation

```bash
# clone project
git clone https://github.com/jp4474/viaABC.git
cd viaABC

# [OPTIONAL] create a virtual environment (only tested on Python 3.10.12)
python -m venv myenv
source myenv/bin/activate

# install PyTorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Resources
You can read more about VAE annealing in [this paper](https://arxiv.org/pdf/1903.10145) and [this blog](https://medium.com/@chengjing/a-must-have-training-trick-for-vae-variational-autoencoder-d28ff53b0023).

Related configuration options are `vae_warmup_steps` and `annealing`.  
See `src/models/lightning_module.py` and `src/models/components/annealing.py` for implementation details.

You can turn off annealing by setting `annealing: false` in the config file. In this case, the model uses **β-VAE**, where the strength of the KL divergence term is controlled by `kld_weight`. 

## TODO

1. C++ code for spatial2D is slow when using pybind (~0.7 s per iteration). Consider using CPython instead.

```bibtex
@misc{park2025spatiodynamicinferenceusingvisionbased,
  title        = {Spatiodynamic Inference Using Vision-Based Generative Modelling},
  author       = {Jun Won Park and Kangyu Zhao and Sanket Rane},
  year         = {2025},
  eprint       = {2507.22256},
  archivePrefix= {arXiv},
  primaryClass = {q-bio.QM},
  url          = {https://arxiv.org/abs/2507.22256}
}
```
