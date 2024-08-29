#importing packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import numpy as np


from network_pt import Net
import pock

import torch.nn as nn
import torch.nn.functional as F

from network_pt import Net

#device gpu if available cpu if not
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##try not to change these parameters
lam = 7
tau = 0.08
sigma = 1.0/(lam*tau)
theta = 1.0
gamma = 0.35*lam
n_iter = 1000 #50 


if __name__ == '__main__':
    ## transforms we will use
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale()])

    batch_size = 20
    #loading the CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    #extracting a subset of the dataset
    trainset = torch.utils.data.Subset(trainset,list(range(0,25000)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


   
   


    ## resnet-18 with no preloaded weights
    net = torchvision.models.resnet18()
    #changing the first convolutional layer to accept grayscale images
    net.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride = (2,2),padding=(3,3),bias = False)
    #changing the fully connected layer of the network
    net.fc = nn.Sequential(nn.Linear(512,256),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(256,128),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(128,10))

    #loading the model to the device we have
    net.to(device)

    #number of spectral components that will be computed
    N = 5
    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    ## training loop
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        print(f"epoch:{epoch+1}")
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #loading into the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            #spec_image is a vector of 2 images and a tensor
            #original image, augmented image
            #the tensor is the iterations of the TV denoising u(n)
            spec_image = pock.spectral(N,inputs)
            #we combine both the augmented image and the original image together
            input_image = torch.concatenate((spec_image[0],spec_image[1]))
            #the labels are exactly equal so we want to get two multiples of the labels
            cur_labels = torch.concatenate((labels,labels))



            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            #forward pass
            outputs = net(input_image)
            #calculate loss
            loss = criterion(outputs, cur_labels)
            #backward pass
            loss.backward()
            #update the parameters of the network
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


    print('Training done.')

    # save trained model
    torch.save(net.state_dict(), 'saved_model_aug.pt')
    print('Model saved.')
