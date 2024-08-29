# test script
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from network_pt import Net_Flower


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale()])

    batch_size = 20
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testset = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    dataiter = iter(testloader)


    ## load the trained model
    model = torchvision.models.resnet18()
    #change the cnn
    model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride = (2,2),padding=(3,3),bias = False)
    model.fc = nn.Sequential(nn.Linear(512,256),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(256,102))
    #use 'saved_model_aug_flower.pt' for augmented model
    #use 'saved_model_flower.pt' for normal model
    model.load_state_dict(torch.load('saved_model_aug_flower.pt'))
    model.eval()


    

    accuracy = 0
    with torch.no_grad():
        for images,labels in testloader:
            outputs = model(images)
            _,predicted = torch.max(outputs,1)
            accuracy += torch.sum(predicted == labels)
        accuracy = accuracy/len(testset)
        print(f"Accuracy: {accuracy}")

