import lightning as L
import torch
import torch.nn as nn

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
    
    def get_latent(self, x, mask_ratio=0):
        return self.model.get_latent(x, mask_ratio)
    
class FineTuneLightning(L.LightningModule):
    def __init__(self, model: nn.Module, latent_dim: int = 64, lr: float = 1e-4, num_parameters: int = 2):
        super().__init__()
        # self.save_hyperparameters()
        
        self.model = model
        self.lr = lr
        self.linear_layer = nn.Linear(latent_dim, num_parameters)
        self.criterion = nn.MSELoss()
    
    def training_step(self, batch):
        self.model.train()
        parameters, simulations = batch
        logits = self.model.get_latent(simulations, mask_ratio=0).mean(dim=1)
        parameter_estimates = self.linear_layer(logits)
        loss = self.criterion(parameter_estimates, parameters)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        self.model.eval()
        parameters, simulations = batch
        logits = self.model.get_latent(simulations, mask_ratio=0).mean(dim=1)
        parameter_estimates = self.linear_layer(logits)
        loss = self.criterion(parameter_estimates, parameters)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    def forward(self, inputs):
        logits = self.model.get_latent(inputs, mask_ratio=0).mean(dim=1)
        return self.linear_layer(logits)