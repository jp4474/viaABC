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

class PreTrainLightning(L.LightningModule):
    def __init__(self, model, multi_tasks = False, lr=1e-3, warmup_steps=1000, total_steps=52000):
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
        self.log("train_reg_loss", reg_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_space_loss", space_loss, prog_bar=False, on_step=False, on_epoch=True)
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
    
class PlotReconstruction(L.Callback):
    def __init__(self, data):
        super().__init__()
        self.data = data

        self.param_names =  [
            r'$\alpha$',
            r'$\delta$',
        ]

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 == 0:
            pl_module.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                # Prepare the input data
                torch_data = torch.from_numpy(self.data).double().to(pl_module.device).unsqueeze(0)  # Use .float() instead of .double()
                _, _, _, param_est, reconstruction = pl_module(torch_data)

                # Convert reconstruction to numpy
                reconstruction_np = reconstruction.cpu().numpy().squeeze(0)
                
                # Plotting
                # channel_names = ['MZB', 'Donor Ki67', 'Host Ki67', 'Nfd']
                channel_names = ['Prey', 'Predator']
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))

                for i in range(2):
                    ax[i].plot(reconstruction_np[:, i], label='Scaled Reconstruction')
                    ax[i].plot(self.data[:, i], label='Scaled Original')
                    ax[i].set_title(channel_names[i])  # Set title for each subplot
                    ax[i].grid(True)
                    ax[i].set_xlabel('Time Steps')
                    ax[i].set_ylabel('Amplitude')
                    ax[i].legend()

                plt.tight_layout()

                # Log the figure to Neptune
                current_epoch = trainer.current_epoch  # Get the current epoch
                log_key = f"validation/reconstructed_images/epoch_{current_epoch}"  # Dynamic key with epoch
                try:
                    trainer.logger.experiment[log_key].append(
                        File.as_image(fig)  # Log the figure as an image
                    )
                except Exception as e:
                    print(f"Failed to log image to Neptune: {e}")

                plt.close(fig)  # Close the figure to free memory

                # Get the numpy array from param_est
                param_est_np = param_est.cpu().numpy()

                # Create the DataFrame ensuring shape matches (rows, number of columns)
                param_est_df = pd.DataFrame(param_est_np, columns=self.param_names)

                log_key_2 = f"validation/parameter_estimates/epoch_{current_epoch}"  # Dynamic key with epoch
                trainer.logger.experiment[log_key_2].upload(
                    File.as_html(param_est_df)  # Log the figure as an image
                )
             
    def on_train_end(self, trainer, pl_module):
        self.on_validation_epoch_end(trainer, pl_module)  # Call the validation end method to log the last epoch's data

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
        logits = self.pl_module.get_latent(inputs)
        return self.linear_layer(logits)
    
    def get_latent(self, x):
        return self.pl_module.get_latent(x)