import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )   

class BetaUNet(nn.Module):
    """A mixture of BetaVAE and UNet."""
    def __init__(self, beta=1):
        super().__init__()
        
        self.dconv_down1 = conv(3, 64, kernel_size=3, stride=2)
        self.dconv_down2 = conv(64, 128, kernel_size=3, stride=2)
        self.dconv_down3 = conv(128, 256, kernel_size=3, stride=2)
        self.dconv_down4 = conv(256, 512, kernel_size=3, stride=2)
        self.dconv_down5 = conv(512, 512, kernel_size=6, padding=0)

        self.mu_fc = nn.Linear(512, 256)
        self.logvar_fc = nn.Linear(512, 256)  # isotropic Gaussian

        self.latent_fc = nn.Linear(256, 512)

        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=6)
        self.dconv_up5 = conv(512+512, 512, kernel_size=3, stride=1)

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  
        self.dconv_up4 = conv(512+256, 256, kernel_size=3, stride=1)

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dconv_up3 = conv(256+128, 128, kernel_size=3, stride=1)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dconv_up2 = conv(128+64, 64, kernel_size=3, stride=1)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_last = nn.Conv2d(64, 3, 1)

        self.beta = beta


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv5 = self.dconv_down5(conv4)
        encoding = conv5.view(-1, 512)
        
        mu = self.mu_fc(encoding)
        logvar = self.logvar_fc(encoding)
        std = torch.exp(0.5 * logvar)
        latent_sample = std * torch.randn_like(logvar) + mu
        latent_out = self.latent_fc(latent_sample)
        latent_out = latent_out.view(-1, 512, 1, 1)
        
        conv5_up = self.up5(latent_out)
        dconv_up5 = self.dconv_up5(torch.cat([conv5_up, conv4], dim=1))
        conv4_up = self.up4(dconv_up5)
        dconv_up4 = self.dconv_up4(torch.cat([conv4_up, conv3], dim=1))
        conv3_up = self.up3(dconv_up4)
        dconv_up3 = self.dconv_up3(torch.cat([conv3_up, conv2], dim=1))
        conv2_up = self.up2(dconv_up3)
        dconv_up2 = self.dconv_up2(torch.cat([conv2_up, conv1], dim=1))
        conv1_up = self.up1(dconv_up2)
        out = self.conv_last(conv1_up)
        
        return torch.sigmoid(out), mu, std**2
    
    def loss(self, prediction, original, mu, var):
        reconstruction_loss = F.binary_cross_entropy(
            prediction, original, reduction="sum"
        )

        kl_divergence = -0.5 * (1 + var.log() - mu**2 - var).sum()
        
        return reconstruction_loss + self.beta * kl_divergence

class UNet(nn.Module):

    def __init__(self, n_class=3):
        super().__init__()
                
        self.dconv_down1 = conv(3, 64, kernel_size=3, stride=2)
        self.dconv_down2 = conv(64, 128, kernel_size=3, stride=2)
        self.dconv_down3 = conv(128, 256, kernel_size=3, stride=2)
        self.dconv_down4 = conv(256, 512, kernel_size=3, stride=2)        

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  
        # self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   
        self.dconv_up4 = conv(512+256, 256, kernel_size=3, stride=1) 

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  
        # self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   
        self.dconv_up3 = conv(256+128, 128, kernel_size=3, stride=1) 


        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  
        # self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   
        self.dconv_up2 = conv(128+64, 64, kernel_size=3, stride=1) 

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  
        # self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x, return_feature=False):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        
       
        conv4_up = self.up4(conv4)
        dconv_up4 = self.dconv_up4(torch.cat([conv4_up, conv3], dim=1))
        conv3_up = self.up3(dconv_up4)
        
        dconv_up3 = self.dconv_up3(torch.cat([conv3_up, conv2], dim=1))
        
        
        conv2_up = self.up2(dconv_up3)
        dconv_up2 = self.dconv_up2(torch.cat([conv2_up, conv1], dim=1))
        conv1_up = self.up1(dconv_up2)

        out = self.conv_last(conv1_up)
        if return_feature:
            return torch.sigmoid(out), conv4
        else:
            return torch.sigmoid(out)




class Discriminator(nn.Module):

    def __init__(self, n_class=3):
        super().__init__()
                
        self.dconv_down1 = conv(3, 32, kernel_size=5, stride=2)
        self.dconv_down2 = conv(32, 64, kernel_size=5, stride=2)
        self.dconv_down3 = conv(64, 128, kernel_size=5, stride=2)
        self.dconv_down4 = conv(128, 256, kernel_size=5, stride=2)    

        self.classfier = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256*25, 1024),
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )

        
    def forward(self, x):
        
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)

        return  self.classfier(conv4)




    