#importing packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

from network_pt import Net_Flower


#gpu if available cpu if not
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    ## transforms for our dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale()])#transforms.Graysclae

    batch_size = 20
    #flowers102 test set (used for training)
    trainset = torchvision.datasets.Flowers102(root='./data', split = 'test', download=True, transform=transform)
    #extracting subset of the trainset
    trainset = torch.utils.data.Subset(trainset,list(range(0,3000)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)


    


    ## loading the resnet18 model
    net = torchvision.models.resnet18()
    #changing the first layer to accept grayscale
    net.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride = (2,2),padding=(3,3),bias = False)
    #changing the linear layers
    net.fc = nn.Sequential(nn.Linear(512,256),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(256,102))
    net.to(device)


    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)

    ## training loop
    for epoch in range(60):  # loop over the dataset multiple times
        running_loss = 0
        for inputs, labels in trainloader:
            #inputs and labels to chosen device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #forward pass
            outputs = net(inputs)
            #compute loss
            loss = criterion(outputs, labels)
            #backward pass
            loss.backward()
            #update parameters
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'epoch:{epoch},running loss:{running_loss}')


    print('Training done.')

    # save trained model
    torch.save(net.state_dict(), 'saved_model_flower.pt')
    print('Model saved.')
