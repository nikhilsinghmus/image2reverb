import os
import torch
from .networks import Encoder, Generator, Discriminator


# Hyperparameters
G_LR = 2e-4
D_LR = 4e-4
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
LAMBDA = 1


class Room2Reverb:
    def __init__(self, encoder_path):
        """GAN model class, puts everything together."""
        self._encoder_path = encoder_path
        self._init_network()
        self._init_optimizer()

    def _init_network(self): # Initialize networks
        self.enc = Encoder(self._encoder_path)
        self.g = Generator()
        self.d = Discriminator()
        self.g.cuda()
        self.d.cuda()
        
    def _init_optimizer(self): # Initialize optimizers
        self.g_optim = torch.optim.Adam(self.g.model.parameters(), lr=G_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        self.d_optim = torch.optim.Adam(self.d.model.parameters(), lr=D_LR, betas=ADAM_BETA, eps=ADAM_EPS)

    def train_step(self, epoch, spec, label, folder, epoch_iter):
        """Perform one training step."""
        spec.requires_grad = True # For the backward pass, seems necessary for now
        
        # Forward passes through models
        f = self.enc.forward(label).cuda()
        fake_spec = self.g(f)
        d_fake = self.d(fake_spec.detach())
        d_real = self.d(spec)

        # Update the discriminator weights
        for p in self.d.parameters():
            p.requires_grad = True
        self.d.zero_grad()

        # Backward pass
        mean_fake = d_fake.mean()
        mean_real = d_real.mean()
        gradient_penalty = 10 * self.wgan_gp(spec.data, fake_spec.data)
        self.D_loss = mean_fake - mean_real + gradient_penalty
        self.D_loss.backward()
        self.d_optim.step()

        if epoch_iter % 3 == 0: # Train generator once every three iterations
            for p in self.d.parameters():
                p.requires_grad = False
            self.g.zero_grad()
            
            d_fake = self.d(fake_spec)
            mean_fake = d_fake.mean()
            self.G_loss = -mean_fake
            self.G_loss.backward()
        self.g_optim.step()

        if epoch % 10 == 0 and epoch > 0: # Every 10 epochs, store the model
            torch.save(self.g.state_dict(), os.path.join(folder, "Gnet_%d.pth.tar" % epoch))
            torch.save(self.g.state_dict(), os.path.join(folder, "Dnet_%d.pth.tar" % epoch))
            torch.save(self.g.state_dict(), os.path.join(folder, "Gnet_latest.pth.tar"))
            torch.save(self.g.state_dict(), os.path.join(folder, "Dnet_latest.pth.tar"))
        
    def wgan_gp(self, real_data, fake_data): # Gradient penalty to promote Lipschitz continuity. Implementation taken from https://github.com/caogang/wgan-gp
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda()
        interpolates.requires_grad = True
        disc_interpolates = self.d(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda())[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty
