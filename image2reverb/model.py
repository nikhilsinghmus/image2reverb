import numpy
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import pyroomacoustics
from .networks import Encoder, Generator, Discriminator
from .stft import STFT
from .mel import LogMel


# Hyperparameters
G_LR = 2e-4
D_LR = 4e-4
ENC_LR = 1e-5
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
LAMBDA = 100


class Image2Reverb(pl.LightningModule):
    def __init__(self, encoder_path, depthmodel_path, latent_dimension=512, spec="stft", d_threshold=0.2, opt=False):
        super().__init__()
        self._latent_dimension = latent_dimension
        self._d_threshold = d_threshold
        self._opt = opt
        self.enc = Encoder(encoder_path, depthmodel_path)
        self.g = Generator(latent_dimension, spec == "mel")
        self.d = Discriminator(365, spec == "mel")
        self.validation_inputs = []
        self.stft = (LogMel if spec == "mel" else STFT)()

    def forward(self, x):
        f = self.enc.forward(x)[0]
        z = torch.cat((f, torch.randn((f.shape[0], (self._latent_dimension - f.shape[1]) if f.shape[1] < self._latent_dimension else f.shape[1], f.shape[2], f.shape[3])).type_as(x)), 1)
        return self.g(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        opts = None
        if self._opt:
            opts = self.optimizers()
        
        spec, label, _ = batch
        spec.requires_grad = True # For the backward pass, seems necessary for now
        
        # Forward passes through models
        f = self.enc.forward(label)[0].type_as(spec)
        z = torch.cat((f, torch.randn((f.shape[0], (self._latent_dimension - f.shape[1]) if f.shape[1] < self._latent_dimension else f.shape[1], f.shape[2], f.shape[3])).type_as(spec)), 1)
        fake_spec = self.g(z)
        d_fake = self.d(fake_spec.detach(), f)
        d_real = self.d(spec, f)

        # Train Generator or Encoder
        if optimizer_idx == 0 or optimizer_idx == 1:
            d_fake2 = self.d(fake_spec.detach(), f)
            G_loss1 = F.mse_loss(d_fake2, torch.ones(d_fake2.shape).type_as(spec))
            G_loss2 = F.l1_loss(fake_spec, spec)
            G_loss = G_loss1 + (LAMBDA * G_loss2)

            if self._opt:
                self.manual_backward(G_loss, self.opts[optimizer_idx])
                opts[optimizer_idx].step()
                opts[optimizer_idx].zero_grad()

            tqdm_dict = {"G": G_loss}
            self.log_dict(tqdm_dict)

            return G_loss
        else: # Train Discriminator
            l_fakeD = F.mse_loss(d_fake, torch.zeros(d_fake.shape).type_as(spec))
            l_realD = F.mse_loss(d_real, torch.ones(d_real.shape).type_as(spec))
            D_loss = (l_realD + l_fakeD)

            if self._opt and (D_loss > self._d_threshold):
                self.manual_backward(D_loss, self.opts[optimizer_idx])
                opts[optimizer_idx].step()
                opts[optimizer_idx].zero_grad()
        
            tqdm_dict = {"D": D_loss}
            self.log_dict(tqdm_dict)

            return D_loss
            

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(self.g.parameters(), lr=G_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        d_optim = torch.optim.Adam(self.d.parameters(), lr=D_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        enc_optim = torch.optim.Adam(self.enc.parameters(), lr=ENC_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        return [enc_optim, g_optim, d_optim], []
    
    def validation_step(self, batch, batch_idx):
        spec, label, _ = batch
        spec.requires_grad = True # For the backward pass, seems necessary for now
        
        # Forward passes through models
        f = self.enc.forward(label)[0].type_as(spec)
        z = torch.cat((f, torch.randn((f.shape[0], (self._latent_dimension - f.shape[1]) if f.shape[1] < self._latent_dimension else f.shape[1], f.shape[2], f.shape[3])).type_as(spec)), 1)
        fake_spec = self.g(z)
        
        # Get audio
        y_r = [self.stft.inverse(s) for s in spec]
        y_f = [self.stft.inverse(s) for s in fake_spec]

        # RT60 error (in percentages)
        f = pyroomacoustics.experimental.rt60.measure_rt60
        t60_r = [f(y) for y in y_r]
        t60_f = [f(y) for y in y_f]
        val_pct = numpy.mean([((t_b - t_a)/t_a) for t_a, t_b in zip(t60_r, t60_f)])

        return {"val_t60err": val_pct, "val_spec": fake_spec}
    
    def validation_epoch_end(self, outputs):
        if not len(outputs):
            return {}
        val_t60errmean = torch.Tensor([output["val_t60err"] for output in outputs]).mean()
        grid = torchvision.utils.make_grid([x for y in [output["val_spec"] for output in outputs] for x in y])
        self.logger.experiment.add_image("generated_spectrograms", grid, self.current_epoch)
        return {"log": {"val_t60err": val_t60errmean, "step": self.current_epoch}}