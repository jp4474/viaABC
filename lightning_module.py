import lightning as L
import torch
from models import TiMAE

class CustomLightning(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

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
        optimizer = torch.optim.AdamW(params=self.parameters(),lr=1e-3)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=1e-3, 
            epochs=300, 
            steps_per_epoch=10,
            three_phase=True)
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    
    def get_latent(self, batch):
        parameters, simulations = batch
        return self.model.get_latent(simulations)
    
class LotkaVolterraLightning(CustomLightning):
    def __init__(self):
        model = TiMAE(seq_len=8, in_chans=2, 
                      embed_dim=64, num_heads=8, depth=6, 
                      decoder_embed_dim=64, decoder_num_heads=8, decoder_depth=4, 
                      z_type="vae", lambda_ = 0.00025, mask_ratio=0.75, bag_size=1024, dropout = 0.0)
        super().__init__(model=model)
