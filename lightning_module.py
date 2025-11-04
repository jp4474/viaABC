import lightning as L
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

class loss_fn(nn.Module):
    def __init__(self, alpha):
        super(loss_fn, self).__init__()
        self.alpha = alpha
    
    def forward(self, y_pred, y_true, mean, log_var):
        RECON = torch.nn.functional.mse_loss(y_pred, y_true, reduction='mean') 
        KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        return RECON, self.alpha * KLD

class PreTrainLightning(L.LightningModule):
    def __init__(self, model, lr=1e-3, warmup_steps=500, total_steps=80000, prog_bar = False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.prog_bar = prog_bar

    def forward(self, simulations, mask_ratio = None):
        return self.model(x = simulations, mask_ratio = mask_ratio)

    def training_step(self, batch):
        simulations = batch
        loss, space_loss, _ = self(simulations)
        self.log("train_recon_loss", loss, prog_bar=self.prog_bar, on_step=False, on_epoch=True)
        self.log("train_space_loss", space_loss, prog_bar=self.prog_bar, on_step=False, on_epoch=True)
        total_loss = loss + space_loss
        self.log("train_loss", total_loss, prog_bar=self.prog_bar, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch):
        simulations = batch
        loss, space_loss, _ = self(simulations)
        self.log("val_recon_loss", loss, prog_bar=self.prog_bar, on_step=False, on_epoch=True)
        self.log("val_space_loss", space_loss, prog_bar=self.prog_bar, on_step=False, on_epoch=True)
        total_loss = loss + space_loss
        self.log("val_loss", total_loss, prog_bar=self.prog_bar, on_step=False, on_epoch=True)
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


    def get_latent(self, x, pooling_method = None):
        return self.model.get_latent(x, pooling_method)

class PreTrainLightningSpatial(L.LightningModule):
    def __init__(self, model, lr=1e-3, warmup_steps=500, total_steps=80000, mask_ratio=0.75):
        super().__init__()
        self.model = model
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.mask_ratio = mask_ratio

    def forward(self, simulations, mask_ratio = None):
        return self.model(simulations, mask_ratio = mask_ratio)

    def training_step(self, batch):
        simulations = batch
        loss, space_loss, _ = self(simulations, self.mask_ratio)
        self.log("train_recon_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_space_loss", space_loss, prog_bar=False, on_step=False, on_epoch=True)
        total_loss = loss + space_loss
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch):
        simulations = batch
        loss, space_loss, _ = self(simulations, self.mask_ratio)
        self.log("val_recon_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_space_loss", space_loss, prog_bar=False, on_step=False, on_epoch=True)
        total_loss = loss + space_loss
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

    def get_latent(self, x, pooling_method = None):
        return self.model.get_latent(x, pooling_method)
    