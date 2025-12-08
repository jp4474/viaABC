import os
import numpy as np
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.viaABC_utils.inference_utils import load_pretrained_model

@hydra.main(config_path="../configs", config_name="lotka_gaf")
def main(cfg: DictConfig):

    # -----------------------------
    # 1. Load pretrained model
    # -----------------------------
    model = load_pretrained_model(
        model_cfg=cfg.model.net,
        lightning_cfg=cfg.model,
        checkpoint_substr=cfg.checkpoint_substr,
        folder_name=cfg.ckpt_folder_path,
    )

    transform = instantiate(cfg.system.transform) if cfg.system.transform is not None else None
    # -----------------------------
    # 2. Instantiate system
    # -----------------------------
    system = instantiate(cfg.system, model=model, transform=transform)
    # -----------------------------
    # 3. Run viaABC
    # -----------------------------
    system.run(
        num_particles=cfg.num_particles,
        k=cfg.k,
        q_threshold=cfg.q_threshold,
    )

    generations = system.generations

    # -----------------------------
    # 4. Save result
    # -----------------------------
    if not os.path.exists(cfg.folder_name):
        os.makedirs(cfg.folder_name)

    filename = f"{cfg.system.name}_{cfg.k}_{cfg.pooling_method}_{cfg.metric}_{cfg.num_particles}.npz"
    output_path = os.path.join(cfg.folder_name, filename)

    np.savez(output_path, generations=generations)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
