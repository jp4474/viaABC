import lightning as L
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from neptune.types import File

import torch
import pandas as pd
import numpy as np
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

class loss_fn(nn.Module):
    def __init__(self, alpha):
        super(loss_fn, self).__init__()
        self.alpha = alpha
    
    def forward(self, y_pred, y_true, mean, log_var):
        RECON = torch.nn.functional.mse_loss(y_pred, y_true, reduction='mean') 
        KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        return RECON, self.alpha * KLD

class PreTrainLightning(L.LightningModule):
    def __init__(self, model, multi_tasks = False, lr=1e-3, warmup_steps=500, total_steps=78000):
        super().__init__()
        self.model = model
        self.lr = lr
        self.multi_tasks = True if multi_tasks else False
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, simulations, parameters=None, mask_ratio = None):
        if self.multi_tasks:
            return self.model(x = simulations, y = parameters, mask_ratio = mask_ratio)
        else:
            return self.model(x = simulations, mask_ratio = mask_ratio)

    def training_step(self, batch):
        parameters, simulations = batch
        loss, reg_loss, space_loss, _, _ = self(simulations, parameters)
        self.log("train_recon_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        if self.multi_tasks:
            self.log("train_reg_loss", reg_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_space_loss", space_loss, prog_bar=False, on_step=False, on_epoch=True)
        total_loss = loss + space_loss + reg_loss
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch):
        parameters, simulations = batch
        loss, reg_loss, space_loss, _, _ = self(simulations, parameters)
        self.log("val_recon_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        if self.multi_tasks:
            self.log("val_reg_loss", reg_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_space_loss", space_loss, prog_bar=False, on_step=False, on_epoch=True)
        total_loss = loss + space_loss + reg_loss
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss
    
    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            params=params,
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.01,
        )
        def linear_warmup_decay(step, warmup_steps, total_steps):
            if step < warmup_steps:
                return step / warmup_steps
            return max((total_steps - step) / (total_steps - warmup_steps), 0)

        # Linear Warmup + Decay
        scheduler = LambdaLR(
            optimizer, 
            lr_lambda=lambda step: linear_warmup_decay(step, self.warmup_steps, self.total_steps)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1
            }
        }

    def get_latent(self, x):
        return self.model.get_latent(x)
    
class PlotReconstruction(L.Callback):
    def __init__(self, data):
        super().__init__()
        self.data = data

        # Extract data from the input dictionary
        self.obs_data = data.get('obs_data')
        self.scaled_obs_data = data.get('scaled_obs_data')
        self.obs_scale = data.get('obs_scale')
        self.ground_truth = data.get('ground_truth')
        self.scaled_ground_truth = data.get('scaled_ground_truth')
        self.ground_truth_scale = data.get('ground_truth_scale')

        # Parameter names for plotting
        self.param_names = [r'$\alpha$', r'$\delta$']

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 == 0:
            pl_module.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                # Prepare the input data
                torch_data = torch.from_numpy(self.scaled_obs_data).double().to(pl_module.device).unsqueeze(0)
                _, _, _, param_est, reconstruction = pl_module(torch_data)

                # Convert reconstruction to numpy
                reconstruction_np = reconstruction.cpu().numpy().squeeze(0)

                # Plotting
                channel_names = ['Prey', 'Predator']
                fig, ax = plt.subplots(2, 2, figsize=(12, 10))

                for i in range(2):
                    # Scaled data plots (first row)
                    ax[0, i].plot(reconstruction_np[:, i], label='Reconstructed')
                    ax[0, i].plot(self.scaled_obs_data[:, i], label='Observed')
                    ax[0, i].plot(self.scaled_ground_truth[:, i], label='Ground Truth')
                    ax[0, i].set_title(f'Scaled {channel_names[i]}')
                    ax[0, i].grid(True)
                    ax[0, i].set_xlabel('Time Steps')
                    ax[0, i].set_ylabel('Population')
                    ax[0, i].legend()

                    # Unscaled data plots (second row)
                    ax[1, i].plot(reconstruction_np[:, i] * self.obs_scale[i], label='Reconstructed')
                    ax[1, i].plot(self.obs_data[:, i], label='Observed')
                    ax[1, i].plot(self.ground_truth[:, i], label='Ground Truth')
                    ax[1, i].set_title(f'Unscaled {channel_names[i]}')
                    ax[1, i].grid(True)
                    ax[1, i].set_xlabel('Time Steps')
                    ax[1, i].set_ylabel('Population')
                    ax[1, i].legend()

                plt.tight_layout()

                # Log the figure to Neptune
                current_epoch = trainer.current_epoch
                log_key = f"validation/reconstructed_images/epoch_{current_epoch}"
                try:
                    trainer.logger.experiment[log_key].append(File.as_image(fig))
                except Exception as e:
                    print(f"Failed to log image to Neptune: {e}")

                plt.close(fig)  # Close the figure to free memory

    def on_train_end(self, trainer, pl_module):
        self.on_validation_epoch_end(trainer, pl_module)  # Log the last epoch's data

class FineTuneLightning(L.LightningModule):
    def __init__(self, pl_module, lr: float = 1e-4, num_parameters: int = 6, linear_probe: bool = False):
        super().__init__()
        # self.save_hyperparameters()
        self.pl_module = pl_module
        self.latent_dim = self.pl_module.model.decoder_embed_dim
        self.lr = lr
        self.linear_layer = nn.Sequential(
            nn.Linear(self.latent_dim, num_parameters)
        )
        self.criterion = nn.MSELoss()

        # freeze the model parameters
        if linear_probe:
            self.pl_module.freeze()
        else:
            self.pl_module.unfreeze()

        #self.finetuning = True
        self.pl_module.model.decoder_embed.finetuning = True
    
    def training_step(self, batch):
        parameters, simulations = batch
        parameter_estimates = self(simulations)
        loss = self.criterion(parameter_estimates, parameters)
        self.log(f"train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        parameters, simulations = batch
        parameter_estimates = self(simulations)
        loss = self.criterion(parameter_estimates, parameters)
        self.log(f"val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(params, lr=self.lr)

    def forward(self, inputs):
        logits = self.get_latent(inputs)
        return self.linear_layer(logits)
    
    def get_latent(self, x):
        return self.pl_module.get_latent(x).mean(dim=1)
    

class PreTrainLightningTriplet(L.LightningModule):
    def __init__(self, model, multi_tasks = False, lr=1e-3, warmup_steps=1000, total_steps=52000):
        super().__init__()
        self.model = model
        self.lr = lr
        self.multi_tasks = True if multi_tasks else False
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.criterion = loss_fn(0.00025)

    def forward(self, simulations):
        return self.model(simulations)

    def training_step(self, batch):
        anchors, positives, negatives = batch

        loss_anchor, _, space_loss_anchor, _, _ = self(anchors)
        loss_positive, _, space_loss_positive, _, _ = self(positives)
        loss_negative, _, space_loss_negative, _, _ = self(negatives)

        loss = (loss_anchor + loss_positive + loss_negative) / 3
        space_loss = (space_loss_anchor + space_loss_positive + space_loss_negative) / 3

        self.log("train_recon_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_reg_loss", reg_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_space_loss", space_loss, prog_bar=False, on_step=False, on_epoch=True)
        total_loss = loss + space_loss
        total_loss = loss + reg_loss + space_loss
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss
    
    def validation_step(self, batch):
        parameters, simulations = batch
        loss, reg_loss, space_loss, _, _ = self(simulations, parameters, 0)
        self.log("val_recon_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_reg_loss", reg_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_space_loss", space_loss, prog_bar=False, on_step=False, on_epoch=True)
        total_loss = loss + reg_loss + space_loss
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss
    
    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            params=params,
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.01,
        )
        def linear_warmup_decay(step, warmup_steps, total_steps):
            if step < warmup_steps:
                return step / warmup_steps
            return max((total_steps - step) / (total_steps - warmup_steps), 0)

        # Linear Warmup + Decay
        scheduler = LambdaLR(
            optimizer, 
            lr_lambda=lambda step: linear_warmup_decay(step, self.warmup_steps, self.total_steps)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1
            }
        }

    def get_latent(self, x):
        # return self.model.get_latent(x).mean(dim=1)
        return self.model.get_latent(x)