# test script
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from network_pt import Net

if __name__ == '__main__':
    ## same transforms as before
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale()])

    #batch size doesnt matter that much here
    batch_size = 30
    
    #test set of imagenette (only train and val split) (set download to false if already downloaded once)
    testset = torchvision.datasets.Imagenette(root='./data', split = 'val', download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    dataiter = iter(testloader)


    ## load the trained model
    ## resnet-18
    model = torchvision.models.resnet18()
    #change the cnn as before
    model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride = (2,2),padding=(3,3),bias = False)
    model.fc = nn.Sequential(nn.Linear(512,256),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(256,128),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(128,10))
    #loading the model 'saved_model_aug_imagenette.pt' for preprocessed model
    #'saved_model_imagenette.pt' for base model
    model.load_state_dict(torch.load('saved_model_aug_imagenette.pt'))
    #set model into evaluation mode
    model.eval()



    #evaluation script for torch no grad
    with torch.no_grad():
        accuracy = 0
        print('starting evaluation')
        for images,labels in testloader:
            outputs = model(images)
            _,predicted = torch.max(outputs,1)
            accuracy += torch.sum(predicted == labels)
        accuracy = accuracy/len(testset)
        print(f"Accuracy: {accuracy}")

