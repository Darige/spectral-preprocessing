#importing packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#importing functions
from network_pt import Net_Flower
import pock

##try not to change these parameters
lam = 7
tau = 0.08
sigma = 1.0/(lam*tau)
theta = 1.0
gamma = 0.35*lam
n_iter = 1000 #50 


#device gpu if available cpu if not
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    #transforming the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale()])

    batch_size = 20
    #our trainset is the testset for this experiment
    trainset = torchvision.datasets.Flowers102(root='./data', split = 'test', download=True, transform=transform)
    #splitting the training data into a smaller subset
    trainset = torch.utils.data.Subset(trainset,list(range(0,6000)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    

   
   


    #loading resnet18 model
    net = torchvision.models.resnet18()
    #changing the 1st layer to accept grayscale images
    net.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride = (2,2),padding=(3,3),bias = False)
    #changing the fully connected layers to fit our problem
    net.fc = nn.Sequential(nn.Linear(512,256),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(256,102))
    net.to(device)
    #number of components
    N = 5
    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    


    ## train
    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs,labels in trainloader:
            #loading into cuda
            inputs = inputs.to(device)
            labels = labels.to(device)
            # get the inputs; data is a list of [inputs, labels]
            spec_image = pock.spectral(N,inputs)
            #putting the data augmented images and normal images together
            input_image = torch.concatenate((spec_image[0],spec_image[1]))
            #the labels too
            cur_labels = torch.concatenate((labels,labels))


            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            #forward pass
            outputs = net(input_image)
            #computing the loss
            loss = criterion(outputs, cur_labels)
            #backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'epoch:{epoch},running_loss = {running_loss}')


    print('Training done.')

    # save trained model
    torch.save(net.state_dict(), 'saved_model_aug_flower.pt')
    print('Model saved.')
