from typing import Any, Dict
import lightning as L
import torch
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.models.components import Annealer

class PreTrainLightning(L.LightningModule):
    def __init__(self, 
                net: DictConfig,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                compile: bool,
                vae_warmup_steps: int = 100000,):
        super().__init__()
        
        self.save_hyperparameters(logger=False)

        self.model = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile = compile
        self.prog_bar = True
        self.vae_warmup_steps = vae_warmup_steps

    def forward(self, simulations, mask_ratio=None):
        # forward should ONLY do forward-pass logic
        recon_loss, space_loss, _ = self.model(simulations, mask_ratio)
        return recon_loss, space_loss

    def forward_step(self, simulations):
        """Shared loss computation for both training and validation."""
        recon_loss, space_loss = self(simulations)
        scaled_space_loss = self.annealer(space_loss)
        total_loss = recon_loss + scaled_space_loss
        return recon_loss, space_loss, scaled_space_loss, total_loss

    def training_step(self, batch, batch_idx):
        recon_loss, space_loss, scaled_space_loss, total_loss = self.forward_step(batch)
        # Logging
        self.log("train/recon_loss", recon_loss, prog_bar=self.prog_bar, on_step=True, on_epoch=True)
        self.log("train/space_loss", space_loss, prog_bar=self.prog_bar, on_step=True, on_epoch=True)
        self.log("train/scaled_space_loss", scaled_space_loss, prog_bar=self.prog_bar, on_step=True, on_epoch=True)
        self.log("train/loss", total_loss, prog_bar=self.prog_bar, on_step=True, on_epoch=True)
        self.log("train/kld_weight", self.annealer._slope(), prog_bar=self.prog_bar, on_step=True, on_epoch=True)

        # Move scheduler AFTER logging
        self.annealer.step()
        
        return total_loss

    def validation_step(self, batch, batch_idx):        
        recon_loss, space_loss, _ = self.model(batch, 0.0)

        scaled_space_loss = self.annealer(space_loss)
        total_loss = recon_loss + scaled_space_loss
        # No annealer stepping here!
        self.log("val/recon_loss", recon_loss, prog_bar=self.prog_bar, on_step=True, on_epoch=True)
        self.log("val/space_loss", space_loss, prog_bar=self.prog_bar, on_step=True, on_epoch=True)
        self.log("val/scaled_space_loss", scaled_space_loss, prog_bar=self.prog_bar, on_step=True, on_epoch=True)
        self.log("val/loss", total_loss, prog_bar=self.prog_bar, on_step=True, on_epoch=True)

        return total_loss

    def test_step(self, batch):
        pass
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile and stage == "fit":
            self.model = torch.compile(self.model)
            self.annealer = Annealer(total_steps=self.vae_warmup_steps, shape='cosine', baseline=0.0, cyclical=True, disable=False)
            
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss", # We do not have a validation metric here since it is pre-training
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def get_latent(self, x, pooling_method):
        return self.model.extract_features(x, pooling_method)
    