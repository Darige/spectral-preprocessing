#importing packages
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from network_pt import Net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    ## transforms we will use
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale()])

    #batch size
    batch_size = 100
    #CIFAR-10 testset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #testset = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    dataiter = iter(testloader)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    ## loading the trained resnet-18 model
    ## cnn
    model = torchvision.models.resnet18()
    #change the cnn
    model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride = (2,2),padding=(3,3),bias = False)
    model.fc = nn.Sequential(nn.Linear(512,256),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(256,128),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(128,10))
    #saved_model.pt for normal 
    #saved_model_aug.pt for augmented model
    model.load_state_dict(torch.load('saved_model.pt'))
    #setting model to evaluation state
    model.eval()



    #code block for calculating accuracy
    #torch.no_grad so we have consistent results
    with torch.no_grad():
        accuracy = 0
        for images,labels in testloader:
            outputs = model(images)
            _,predicted = torch.max(outputs,1)
            accuracy += torch.sum(predicted == labels)
        accuracy = accuracy/len(testset)
        print(f"Accuracy: {accuracy}")


