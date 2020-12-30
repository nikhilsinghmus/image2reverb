import os
import torch
from .networks import Encoder, Generator, Discriminator
from .util import estimate_t60


# Hyperparameters
G_LR = 2e-4
D_LR = 4e-4
ENC_LR = 1e-5
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
LAMBDA = 100


class Room2Reverb:
    def __init__(self, encoder_path, depthmodel_path, latent_dimension=512):
        """GAN model class, puts everything together."""
        self._encoder_path = encoder_path
        self._depthmodel_path = depthmodel_path
        self._latent_dimension = latent_dimension
        self._init_network()
        self._init_optimizer()
        self._criterion_GAN = torch.nn.MSELoss().cuda()
        self._criterion_extra = torch.nn.L1Loss().cuda()

    def _init_network(self): # Initialize networks
        self.enc = Encoder(self._encoder_path, self._depthmodel_path)
        self.g = Generator()
        self.d = Discriminator()
        self.g.cuda()
        self.d.cuda()
        
    def _init_optimizer(self): # Initialize optimizers
        self.g_optim = torch.optim.Adam(self.g.model.parameters(), lr=G_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        self.d_optim = torch.optim.Adam(self.d.model.parameters(), lr=D_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        enc_params = list(self.enc.model.parameters()) + list(self.enc.depth_encoder.parameters()) + list(self.enc.depth_decoder.parameters())
        self.enc_optim = torch.optim.Adam(enc_params, lr=ENC_LR, betas=ADAM_BETA, eps=ADAM_EPS)

    def train_step(self, spec, label):
        """Perform one training step."""
        spec.requires_grad = True # For the backward pass, seems necessary for now
        
        # Forward passes through models
        f = self.enc(label)[0].cuda().detach()
        z = torch.randn((f.shape[0], self._latent_dimension - f.shape[1], f.shape[2], f.shape[3])).cuda()
        fake_spec = self.g(torch.cat((f, z), 1))
        d_fake = self.d(fake_spec.detach(), f)
        d_real = self.d(spec, f)

        # MSE + L1 version
        self.g.zero_grad()
        self.enc.zero_grad()
        d_fake2 = self.d(fake_spec.detach(), f)
        # d_real2 = self.d(spec)
        G_loss1 = self._criterion_GAN(d_fake2, torch.ones(d_fake2.shape).cuda())
        G_loss2 = self._criterion_extra(fake_spec, spec)
        self.G_loss = G_loss1 + (LAMBDA * G_loss2)
        self.G_loss.backward()
        self.g_optim.step()
        self.enc_optim.step()

        self.d.zero_grad()
        l_fakeD = self._criterion_GAN(d_fake, torch.zeros(d_fake.shape).cuda())
        l_realD = self._criterion_GAN(d_real, torch.ones(d_real.shape).cuda())
        self.D_loss = (l_realD + l_fakeD)
        
        if self.D_loss > 0.2: # Avoid overly strong discriminator
            self.D_loss.backward()
            self.d_optim.step()
    
    def load_generator(self, path): # Load a pre-trained generator
        self.g.load_state_dict(path)
    
    def load_discriminator(self, path): # Load a pre-trained discriminator
        self.d.load_state_dict(path)
    
    def inference(self, img): # Generate output
        f = self.enc.forward(img)[0].cuda()
        return self.g(torch.cat((f, torch.randn((f.shape[0], (512 - f.shape[1]) if f.shape[1] < 512 else f.shape[1], f.shape[2], f.shape[3])).cuda()), 1))
