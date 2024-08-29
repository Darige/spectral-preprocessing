#loading packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from network_pt import Net
import pock

#gpu if available else cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    #transforing the data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale()])#transforms.Graysclae

    batch_size = 20
    #loading the imagenette data (change to True if running for first time) (set to False after download or has some problems)
    trainset = torchvision.datasets.Imagenette(root='./data', split = 'train', download=False, transform=transform)
    #splitting the trainset so we only use a subset
    trainset = torch.utils.data.Subset(trainset,list(range(0,9000)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)


    


    ## loading the resnet18
    net = torchvision.models.resnet18()
    #changing the 1st layer of the convolution for grayscale images
    net.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride = (2,2),padding=(3,3),bias = False)
    #changing fully connected layers as shown in paper
    net.fc = nn.Sequential(nn.Linear(512,256),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(256,128),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(128,10))
    #changing to gpu
    net.to(device)

    #number of spectral components
    N = 5
    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)

    ## training loop
    for epoch in range(15):  # loop over the dataset multiple times
        running_loss = 0
        for inputs, labels in trainloader:
            #loading into gpu or cpu
            inputs = inputs.to(device)
            labels = labels.to(device)

            #calculating the spectral images
            spec_image = pock.spectral(N,inputs)
            #putting them together
            input_image = torch.concatenate((spec_image[0],spec_image[1]))
            cur_labels = torch.concatenate((labels,labels))



            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            #forward pass
            outputs = net(input_image)
            #calculating loss
            loss = criterion(outputs, cur_labels)
            #backward pass
            loss.backward()
            #updating the parameters
            optimizer.step()
            # print statistics
            running_loss += loss.item()
        print(f'epoch:{epoch},running loss:{running_loss}')


    print('Training done.')

    # save trained model
    torch.save(net.state_dict(), 'saved_model_aug_imagenette.pt')
    print('Model saved.')
