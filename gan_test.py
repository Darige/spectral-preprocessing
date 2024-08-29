#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pock

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



if __name__ == '__main__':
    # Root directory for dataset
    dataroot = "data/celeba"
    # Number of workers for dataloader
    workers = 2
    # Batch size during training
    batch_size = 128
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64
    # Number of channels in the training images. For color images this is 3
    nc = 1
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Number of training epochs
    num_epochs = 5
    # Learning rate for optimizers
    lr = 0.0002
    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 0
    #number of spectral components
    N = 3
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    #image list
    img_list = []
    img_list_full = []
    #dataset 
    netD = Discriminator(ngpu).to(device)
    netD.load_state_dict(torch.load(f'saved_model_netD0.pt'))
    netD.main[11] = nn.Identity()
    dataset = torchvision.datasets.CelebA(root='./data/celeba',split='train',download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   transforms.Grayscale()
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=True, num_workers=workers)
    dataiter = iter(dataloader)
    images,labels = next(dataiter)
    images,labels = images.to(device),labels.to(device)
    real_mean = torch.mean(netD(images).flatten(1))
    real_var = torch.cov(netD(images).flatten(1))

    
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    # Create the generator
    netG = Generator(ngpu).to(device)
    #loading each generator and appending each generators output to a list u(n)
    for component in range(4):
        netG.load_state_dict(torch.load(f'saved_model_netG{component}.pt'))
        img_list.append(vutils.make_grid(netG(fixed_noise).detach().cpu(), padding=2, normalize=True))

    #combining the images to reconstruct
    combined_image = pock.phi_sum(N-2,img_list) + pock.f_r(N-2,img_list)
    #reconstructed images
    image = np.transpose(combined_image,(1,2,0))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(img_list[0],(1,2,0)).numpy())
    plt.subplot(1,2,2)
    plt.title("Reconstructed Images")
    plt.axis("off")
    plt.imshow(image.numpy())
    plt.savefig('generated_spectral.png')

    #script for calculating the FID SCORE
    fixed_noise_full = torch.randn(64, nz, 1, 1, device=device)
    for component in range(4):
        netG.load_state_dict(torch.load(f'saved_model_netG{component}.pt'))
        img_list_full.append(netG(fixed_noise_full).detach().cpu())
    combined_image_full = pock.phi_sum(N-2,img_list_full) + pock.f_r(N-2,img_list_full)    
    comb_mean = torch.mean(netD(combined_image_full).flatten(1))
    comb_var = torch.cov(netD(combined_image_full).flatten(1))
    gen_mean = torch.mean(netD(img_list_full[0]).flatten(1))
    gen_var = torch.cov(netD(img_list_full[0]).flatten(1))
    FID_comb = torch.norm(input = comb_mean - real_mean,p=2) + torch.trace(comb_var + real_var - 2*torch.sqrt(comb_var*real_var))
    FID_gen = torch.norm(input = gen_mean - real_mean,p=2) + torch.trace(gen_var + real_var - 2*torch.sqrt(gen_var*real_var))
    print(f'FID of combined model: {FID_comb}')
    print(f'FID of generated model: {FID_gen}')