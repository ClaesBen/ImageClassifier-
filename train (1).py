import argparse
import numpy as np
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import json


def get_input_args():
       
    parser = argparse.ArgumentParser(description = 'Parser for predict.py')
    
    parser.add_argument('--data_dir', default='/home/workspace/ImageClassifier/flowers' ,action = 'store',type=str)
    parser.add_argument('--save_dir', type=str, dest='save_dir')
    parser.add_argument('--arch', type=str,default='vgg16',dest='arch', help= 'vgg 16 or 19')
    parser.add_argument('--learn_rate', type=float , dest='learn_rate' , default=0.0001)
    parser.add_argument('--hidden_units',type=int, default= 2048)
    parser.add_argument('--epochs', type=int, dest='epochs',default= 5)
    parser.add_argument('--gpu',dest = 'gpu' , action="store_true")
    
    return parser.parse_args()

def main():
    
    input_args = get_input_args()
    
    data_dir = input_args.data_dir
    
    if (input_args.save_dir is None):
        save_dir = 'check.pth'
    else:
        save_dir = input_args.save_dir
    arch = input_args.arch
    learn_rate = input_args.learn_rate
    hidden = input_args.hidden_units
    e = input_args.epochs
    gpu = input_args.gpu
    
    print(gpu)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    print(device)
    train_dir = data_dir + '/train'
    #valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
      
    random_transforms = transforms.Compose([transforms.RandomRotation(25),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
    
    data_transforms = transforms.Compose([transforms.Resize(256), 
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]) 
    
    train_data = datasets.ImageFolder(train_dir, transform=random_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)
    #val_data = datasets.ImageFolder(valid_dir, transform=data_transforms)

    batch_size = 32
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle= True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    #valloader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle= True)
    
 
    model_att = getattr(models,arch)
    print(model_att)
    my_model = model_att(pretrained = True)

    output_node = len(class_to_idx)
    print(output_node)
    
    if hidden is None: 
        hidden_units = 2048
    else:
        hidden_units = hidden
        
    for param in my_model.parameters():
        param.requires_grad = False
    
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, hidden_units)),
                                            ('dropout1', nn.Dropout(p = 0.25)),
                                            ('relu1', nn.ReLU()),
                                            ('fc2', nn.Linear(hidden_units,1024)),
                                            ('dropout2', nn.Dropout(p = 0.25)),
                                            ('relu2', nn.ReLU()),
                                            ('fc3', nn.Linear(1024, output_node)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
    
    my_model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(my_model.classifier.parameters(), lr=learn_rate)
    
    
    train_my_model(my_model,e,device,trainloader,testloader,criterion,optimizer)
    
    
    checkpoint = {
        'epochs': e,
        'learn_rate': learn_rate,
        'batch_size': batch_size,
        'data_transforms': data_transforms,
        'model': model_att(pretrained = True),
        'state_dict': my_model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': my_model.classifier}
    
    torch.save(checkpoint, save_dir)

        
    
def train_my_model(my_model,e,device,trainloader,testloader,criterion,optimizer):
    
    print_every = 25
    steps = 0
    epochs = int(e)

    my_model = my_model.to(device)
    my_model.train()
    
    for e in range(epochs):
        run_loss = 0 
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = my_model.forward(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            
            if steps % print_every == 0:
                my_model.eval()
                test_loss = 0
                accuracy = 0
                  
                for ii, (inputs, labels) in enumerate(testloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    out = my_model.forward(inputs)
                    test_loss += criterion(out, labels)
                    #calc acc
                    ps = torch.exp(out).data
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Training Loss: {run_loss/print_every:.3f}.. "
                      f"test Loss: {test_loss/len(testloader):.3f}.. "
                      f"Accuracy: {accuracy/len(testloader):.3f}")
                run_loss = 0
                my_model.train()
            
    print("Done training for now!")
    

    
if __name__ == '__main__':
    main()

