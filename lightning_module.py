import lightning as L
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from neptune.types import File

class PreTrainLightning(L.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, inputs):
        loss, reconstruction = self.model(inputs)
        return loss, reconstruction

    def training_step(self, batch):
        parameters, simulations = batch
        loss, reconstruction = self(simulations)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        parameters, simulations = batch
        loss, reconstruction = self(simulations)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        return optimizer
    
    def get_latent(self, x):
        return self.model.get_latent(x).mean(dim=1)
    
class PlotReconstruction(L.Callback):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Prepare the input data
            torch_data = torch.from_numpy(self.data).float().to(pl_module.device).unsqueeze(0)
            loss, reconstruction = pl_module(torch_data)

            # Convert reconstruction to numpy
            reconstruction_np = reconstruction.cpu().numpy().squeeze(0)
            
            # Plotting
            channel_names = ['MZB', 'Donor Ki67', 'Host Ki67', 'Nfd']
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))

            for i in range(4):
                ax[i].plot(reconstruction_np[:, i], label='Reconstruction')
                ax[i].plot(self.data[:, i], label='Original')
                ax[i].set_title(channel_names[i])  # Set title for each subplot
                ax[i].grid(True)
                ax[i].set_xlabel('Time Steps')
                ax[i].set_ylabel('Amplitude')
                ax[i].legend()

            plt.tight_layout()

            # Log the figure to Neptune
            if self.logger:
                current_epoch = trainer.current_epoch  # Get the current epoch
                log_key = f"validation/reconstructed_images/epoch_{current_epoch}"  # Dynamic key with epoch
                self.logger.experiment[log_key].append(
                    File.as_image(fig)  # Log the figure as an image
                )
            plt.close(fig)  # Close the figure to free memory

class FineTuneLightning(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-4, num_parameters: int = 6, linear_probe: bool = False):
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.latent_dim = model.embed_dim
        self.lr = lr
        self.linear_layer = nn.Sequential(
            nn.Linear(self.latent_dim, num_parameters)
        )

        self.criterion = nn.MSELoss()

        # freeze the model parameters
        if linear_probe:
            self.model.freeze()
        else:
            self.model.unfreeze()
    
    def training_step(self, batch):
        parameters, simulations = batch
        logits = self.get_latent(simulations)
        parameter_estimates = self.linear_layer(logits)
        loss = self.criterion(parameter_estimates, parameters)
        self.log(f"train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

    def validation_step(self, batch):
        parameters, simulations = batch
        logits = self.get_latent(simulations)
        parameter_estimates = self.linear_layer(logits)
        loss = self.criterion(parameter_estimates, parameters)
        self.log(f"val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, inputs):
        logits = self.get_latent(inputs)
        return self.linear_layer(logits)
    
    def get_latent(self, x):
        return self.model.get_latent(x).mean(dim=1)