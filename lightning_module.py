import lightning as L
import torch
from models import TiMAE

class CustomLightning(L.LightningModule):
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
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        parameters, simulations = batch
        loss, reconstruction = self(simulations)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)

        return optimizer
    
    def get_latent(self, x, mask_ratio=0):
        return self.model.get_latent(x, mask_ratio)
    
class FineTuningLightning(CustomLightning):
    def __init__(self, model, latent_dim = 64, lr=1e-4):
        super().__init__(model, lr)
        self.model = model
        self.lr = lr
        self.linear_layer = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, 2)
        )
        
        self.criterion = torch.nn.MSELoss()
    
    def training_step(self, batch):
        self.model.train()
        parameters, simulations = batch
        logits = self.model.get_latent(simulations, mask_ratio=0)
        parameter_estimates = self.linear_layer(logits)
        loss = self.criterion(parameter_estimates, parameters)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        parameters, simulations = batch
        logits = self.model.get_latent(simulations, mask_ratio=0)
        parameter_estimates = self.linear_layer(logits)
        loss = self.criterion(parameter_estimates, parameters)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, inputs):
        logits = self.model.get_latent(inputs, mask_ratio=0)
        parameter_estimates = self.linear_layer(logits)
        return parameter_estimates

