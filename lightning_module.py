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
    
        # scheduler = get_linear_schedule_with_warmup(
        #         optimizer,
        #         num_warmup_steps=self.n_warmup_steps,
        #         num_training_steps=self.n_training_steps,
        #     )
        
        # return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))
    def get_latent(self, x, mask_ratio=0):
        return self.model.get_latent(x, mask_ratio)
    
# class LotkaVolterraLightning(CustomLightning):
#     def __init__(self, lr=1e-3):
#         model = TiMAE(seq_len=8, in_chans=2, 
#                       embed_dim=8, num_heads=8, depth=2, 
#                       decoder_embed_dim=8, decoder_num_heads=8, decoder_depth=1, 
#                       z_type="vae", lambda_ = 0.00025, mask_ratio=0.25, bag_size=1024, dropout = 0.0)
#         super().__init__(model=model, lr=lr)
