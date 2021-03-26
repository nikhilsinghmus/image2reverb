import os
import json
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
from .util import compare_t60


# Hyperparameters
G_LR = 4e-4
D_LR = 2e-4
ENC_LR = 1e-5
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
LAMBDA = 100


class Image2Reverb(pl.LightningModule):
    def __init__(self, encoder_path, depthmodel_path, latent_dimension=512, spec="stft", d_threshold=0.2, t60p=True, constant_depth = None, test_callback=None):
        super().__init__()
        self._latent_dimension = latent_dimension
        self._d_threshold = d_threshold
        self.constant_depth = constant_depth
        self.t60p = t60p
        self.confidence = {}
        self.tau = 50
        self.test_callback = test_callback
        self._opt = (d_threshold != None) and (d_threshold > 0) and (d_threshold < 1)
        self.enc = Encoder(encoder_path, depthmodel_path, constant_depth=self.constant_depth, device=self.device)
        self.g = Generator(latent_dimension, spec == "mel")
        self.d = Discriminator(365, spec == "mel")
        self.validation_inputs = []
        self.stft_type = spec

    def forward(self, x):
        f = self.enc.forward(x)[0]
        z = torch.cat((f, torch.randn((f.shape[0], (self._latent_dimension - f.shape[1]) if f.shape[1] < self._latent_dimension else f.shape[1], f.shape[2], f.shape[3]), device=self.device)), 1)
        return self.g(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        opts = None
        if self._opt:
            opts = self.optimizers()
        
        spec, label, p = batch
        spec.requires_grad = True # For the backward pass, seems necessary for now
        
        # Forward passes through models
        f = self.enc.forward(label)[0]
        z = torch.cat((f, torch.randn((f.shape[0], (self._latent_dimension - f.shape[1]) if f.shape[1] < self._latent_dimension else f.shape[1], f.shape[2], f.shape[3]), device=self.device)), 1)
        fake_spec = self.g(z)
        d_fake = self.d(fake_spec.detach(), f)
        d_real = self.d(spec, f)

        # Train Generator or Encoder
        if optimizer_idx == 0 or optimizer_idx == 1:
            d_fake2 = self.d(fake_spec.detach(), f)
            G_loss1 = F.mse_loss(d_fake2, torch.ones(d_fake2.shape, device=self.device))
            G_loss2 = F.l1_loss(fake_spec, spec)
            
            
            G_loss = G_loss1 + (LAMBDA * G_loss2)
            if self.t60p:
                t60_err = torch.Tensor([compare_t60(torch.exp(a).sum(-2).squeeze(), torch.exp(b).sum(-2).squeeze()) for a, b in zip(spec, fake_spec)]).to(self.device).mean()
                G_loss += t60_err
                self.log("t60", t60_err, on_step=True, on_epoch=True, prog_bar=True)

            if self._opt:
                self.manual_backward(G_loss, self.opts[optimizer_idx])
                opts[optimizer_idx].step()
                opts[optimizer_idx].zero_grad()

            self.log("G", G_loss, on_step=True, on_epoch=True, prog_bar=True)

            return G_loss
        else: # Train Discriminator
            l_fakeD = F.mse_loss(d_fake, torch.zeros(d_fake.shape, device=self.device))
            l_realD = F.mse_loss(d_real, torch.ones(d_real.shape, device=self.device))
            D_loss = (l_realD + l_fakeD)

            if self._opt and (D_loss > self._d_threshold):
                self.manual_backward(D_loss, self.opts[optimizer_idx])
                opts[optimizer_idx].step()
                opts[optimizer_idx].zero_grad()
        
            self.log("D", D_loss, on_step=True, on_epoch=True, prog_bar=True)

            return D_loss

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(self.g.parameters(), lr=G_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        d_optim = torch.optim.Adam(self.d.parameters(), lr=D_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        enc_optim = torch.optim.Adam(self.enc.parameters(), lr=ENC_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        return [enc_optim, g_optim, d_optim], []
    
    def validation_step(self, batch, batch_idx):
        spec, label, paths = batch
        examples = [os.path.basename(s[:s.rfind("_")]) for s, _ in zip(*paths)]
        
        # Forward passes through models
        f = self.enc.forward(label)[0]
        z = torch.cat((f, torch.randn((f.shape[0], (self._latent_dimension - f.shape[1]) if f.shape[1] < self._latent_dimension else f.shape[1], f.shape[2], f.shape[3]), device=self.device)), 1)
        fake_spec = self.g(z)
        
        # Get audio
        stft = LogMel() if self.stft_type == "mel" else STFT()
        y_r = [stft.inverse(s.squeeze()) for s in spec]
        y_f = [stft.inverse(s.squeeze()) for s in fake_spec]

        # RT60 error (in percentages)
        val_pct = 1
        try:
            f = lambda x : pyroomacoustics.experimental.rt60.measure_rt60(x, 22050)
            t60_r = [f(y) for y in y_r if len(y)]
            t60_f = [f(y) for y in y_f if len(y)]
            val_pct = numpy.mean([((t_b - t_a)/t_a) for t_a, t_b in zip(t60_r, t60_f)])
        except:
            pass

        return {"val_t60err": val_pct, "val_spec": fake_spec, "val_audio": torch.Tensor(y_f), "val_img": label, "val_examples": examples}
    
    def validation_epoch_end(self, outputs):
        if not len(outputs):
            return
        # Log mean T60 errors (in percentages)
        val_t60errmean = torch.Tensor([output["val_t60err"] for output in outputs]).mean()
        self.log("val_t60err", val_t60errmean, on_epoch=True, prog_bar=True)

        # Log generated spectrogram images
        grid = torchvision.utils.make_grid([torch.flip(x, [0]) for y in [output["val_spec"] for output in outputs] for x in y])
        self.logger.experiment.add_image("generated_spectrograms", grid, self.current_epoch)

        # Log model input images
        grid = torchvision.utils.make_grid([x for y in [output["val_img"] for output in outputs] for x in y])
        self.logger.experiment.add_image("input_images_with_depthmaps", grid, self.current_epoch)
        
        # Log generated audio examples
        for output in outputs:
            for example, audio in zip(output["val_examples"], output["val_audio"]):
                y = audio.unsqueeze(0)
                self.logger.experiment.add_audio("generated_audio_%s" % example, y, sample_rate=22050)
        
        print("Writing confidence.json.")
        with open("confidence.json", "w") as json_file:
            json.dump(self.confidence, json_file, indent=4)

    def test_step(self, batch, batch_idx):
        spec, label, paths = batch
        examples = [os.path.basename(s[:s.rfind("_")]) for s, _ in zip(*paths)]
        
        # Forward passes through models
        f, img = self.enc.forward(label)
        img = (img + 1) * 0.5
        z = torch.cat((f, torch.randn((f.shape[0], (self._latent_dimension - f.shape[1]) if f.shape[1] < self._latent_dimension else f.shape[1], f.shape[2], f.shape[3]), device=self.device)), 1)
        fake_spec = self.g(z)
        
        # Get audio
        stft = LogMel() if self.stft_type == "mel" else STFT()
        y_r = [stft.inverse(s.squeeze()) for s in spec]
        y_f = [stft.inverse(s.squeeze()) for s in fake_spec]

        # RT60 error (in percentages)
        val_pct = 1
        f = lambda x : pyroomacoustics.experimental.rt60.measure_rt60(x, 22050)
        val_pct = []
        for y_real, y_fake in zip(y_r, y_f):
            try:
                t_a = f(y_real)
                t_b = f(y_fake)
                val_pct.append((t_b - t_a)/t_a)
            except:
                val_pct.append(numpy.nan)

        return {"test_t60err": val_pct, "test_spec": fake_spec, "test_audio": y_f, "test_img": img, "test_examples": examples}
    
    def test_epoch_end(self, outputs):
        if not self.test_callback:
            return
            
        examples = []
        t60 = []
        spec_images = []
        audio = []
        input_images = []
        input_depthmaps = []

        for output in outputs:
            for i in range(len(output["test_examples"])):
                img = output["test_img"][i]
                if img.shape[0] == 3:
                    rgb = img
                    img = torch.cat((rgb, torch.zeros((1, rgb.shape[1], rgb.shape[2]), device=self.device)), 0)
                t60.append(output["test_t60err"][i])
                spec_images.append(output["test_spec"][i].cpu().squeeze().detach().numpy())
                audio.append(output["test_audio"][i])
                input_images.append(img.cpu().squeeze().permute(1, 2, 0)[:,:,:-1].detach().numpy())
                input_depthmaps.append(img.cpu().squeeze().permute(1, 2, 0)[:,:,-1].squeeze().detach().numpy())
                examples.append(output["test_examples"][i])
        
        self.test_callback(examples, t60, spec_images, audio, input_images, input_depthmaps)
    
    @property
    def automatic_optimization(self) -> bool:
        return not self._opt