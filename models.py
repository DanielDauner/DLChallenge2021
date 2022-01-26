import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )   


class BetaUNet2Large(nn.Module):
    """A mixture of BetaVAE and UNet."""
    def __init__(self, beta=1, latent_dim=2048):
        super().__init__()


        self.beta = beta
        self.latent_dim = latent_dim
        
        self.dconv_start = conv(3, 64, kernel_size=7, stride=1, padding=3)

        self.dconv_down1 = conv(64, 64, kernel_size=5, stride=2, padding=2)
        self.dconv_down2 = conv(64, 128, kernel_size=5, stride=2, padding=2)
        self.dconv_down3 = conv(128, 256, kernel_size=5, stride=2, padding=2)
        self.dconv_down4 = conv(256, 512, kernel_size=5, stride=2, padding=2)
        self.dconv_down5 = conv(512, 1024, kernel_size=5, stride=1, padding=0)

        self.mu_fc = nn.Linear(1024*2*2, self.latent_dim)
        self.logvar_fc = nn.Linear(1024*2*2, self.latent_dim)  

        self.latent_fc = nn.Linear(self.latent_dim, 1024*2*2)

        self.up5 = nn.ConvTranspose2d(1024, 1024, kernel_size=5)
        self.dconv_up5 = conv(1024+512, 512, kernel_size=3, stride=1)

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)  
        self.dconv_up4 = conv(512+256, 256, kernel_size=3, stride=1)

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)  
        self.dconv_up3 = conv(256+128, 128, kernel_size=3, stride=1)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)  
        self.dconv_up2 = conv(128+64, 64, kernel_size=3, stride=1)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1) 
        self.conv_last = nn.Conv2d(64+64, 3, 1)


    def forward(self, x):
        start = self.dconv_start(x)

        conv1 = self.dconv_down1(start)
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv5 = self.dconv_down5(conv4)

        encoding = conv5.flatten(start_dim=1)

        mu = self.mu_fc(encoding)
        logvar = self.logvar_fc(encoding)
        std = torch.exp(0.5 * logvar)

        latent_sample = std * torch.randn_like(std) + mu

        latent_out = self.latent_fc(latent_sample)
        latent_out = latent_out.view(-1, 1024, 2, 2)
        
        conv5_up = self.up5(latent_out)
        dconv_up5 = self.dconv_up5(torch.cat([conv5_up, conv4], dim=1))
        
        conv4_up = self.up4(dconv_up5)
        dconv_up4 = self.dconv_up4(torch.cat([conv4_up, conv3], dim=1))
        conv3_up = self.up3(dconv_up4)
        dconv_up3 = self.dconv_up3(torch.cat([conv3_up, conv2], dim=1))
        conv2_up = self.up2(dconv_up3)
        dconv_up2 = self.dconv_up2(torch.cat([conv2_up, conv1], dim=1))
        conv1_up = self.up1(dconv_up2)
        out = self.conv_last(torch.cat([conv1_up,start],dim=1))
        
        return torch.sigmoid(out), mu, logvar
    
    def loss(self, prediction, original, mu, logvar):
        reconstruction_loss = F.binary_cross_entropy(prediction, original, reduction="sum")
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        
        return reconstruction_loss/prediction.shape[0]+self.beta*kld_loss, reconstruction_loss/prediction.shape[0], self.beta*kld_loss
    