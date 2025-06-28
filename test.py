import argparse
import os
import yaml
import numpy as np
import torch
from matplotlib import pyplot as plt

from models import MaskedAutoencoderViT3D
from lightning_module import PreTrainLightningSpatial2D
from systems import SpatialSIR3D
from dataset import SpatialSIRDataset

def labels2map(y):
    susceptible = (y == 0)
    infected = (y == 1)
    resistant = (y == 2)
    return np.stack([susceptible, infected, resistant], axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--metric', type=str, default='bertscore_batch')
    args = parser.parse_args()

    folder_name = args.folder
    metric = args.metric

    config = yaml.safe_load(open(f"{folder_name}/config.yaml"))
    model = MaskedAutoencoderViT3D(**config["model"]["params"], in_chans=3)
    
    pretrain_model_path = [f for f in os.listdir(folder_name) if f.endswith(".ckpt")][0]
    pl_model = PreTrainLightningSpatial2D.load_from_checkpoint(
        os.path.join(folder_name, pretrain_model_path), model=model)

    print("Successfully loaded model")

    raw_data = np.load("data/SPATIAL/data.npy")
    raw_data = labels2map(raw_data).astype(np.float32)
    print("Raw data shape:", raw_data.shape)

    sir_abc = SpatialSIR3D(observational_data=raw_data, metric=metric)
    sir_abc.update_model(pl_model)

    train_dataset = SpatialSIRDataset("data/SPATIAL", "train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    simulated_np, _ = sir_abc.simulate([1, 1])
    x = torch.tensor(simulated_np, dtype=torch.float32).to(pl_model.device).unsqueeze(0)

    with torch.no_grad():
        x_, mask, ids_restore = pl_model.model.forward_encoder(x, 0.0)
        y = pl_model.model.forward_decoder(x_, ids_restore)
        y = pl_model.model.unpatchify(y)
        y = y.cpu().numpy().squeeze(0).argmax(axis=0)

    x = x.cpu().numpy().squeeze(0).argmax(axis=0)

    fig, ax = plt.subplots(15, 2, figsize=(5, 40))
    for i in range(15):
        ax[i, 0].imshow(x[i], cmap='viridis')
        ax[i, 1].imshow(y[i], cmap='viridis')
    ax[0, 0].set_title("Input")
    ax[0, 1].set_title("Reconstruction")
    plt.tight_layout()
    plt.savefig(f"{folder_name}/spatial_sir_reconstruction.png", dpi=300, bbox_inches='tight')

    sir_abc.run(num_particles=1000)
    output_file = os.path.join(folder_name, f"spatial_sir_inference_{metric}.npz")
    np.savez(output_file, generations=sir_abc.generations)

if __name__ == "__main__":
    main()
