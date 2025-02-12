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
    
    def get_latent(self, x, mask_id=None):
        return self.model.get_latent(x, mask_id).mean(dim=1)
        
class FineTuneLightning(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-4, num_parameters: int = 6):
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.finetuning = True
        self.latent_dim = model.embed_dim
        # self.model.decoder_embed[1].finetuning = True
        self.lr = lr
        self.linear_layer = nn.Sequential(
            nn.Linear(self.latent_dim, num_parameters)
        )

        self.criterion = nn.L1Loss() # nn.MSELoss()
    
    def training_step(self, batch):
        self.model.train()
        parameters, simulations = batch
        logits = self.get_latent(simulations)
        parameter_estimates = self.linear_layer(logits)
        loss = self.criterion(parameter_estimates, parameters)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        self.model.eval()
        parameters, simulations = batch
        logits = self.get_latent(simulations)
        parameter_estimates = self.linear_layer(logits)
        loss = self.criterion(parameter_estimates, parameters)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    def forward(self, inputs):
        logits = self.get_latent(inputs)
        return self.linear_layer(logits)
    
    def get_latent(self, x, mask_id=None):
        return self.model.get_latent(x, mask_id).mean(dim=1)