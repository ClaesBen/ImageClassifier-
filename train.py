import argparse
import numpy as np
import os
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image

def main():
    global arg
    arg = get_input_args()
    generate_model()
    print("I've done what you asked")
    return None

def get_input_args():
       
    parser = argparse.ArgumentParser(description = 'Parser for predict.py')
    
    parser.add_argument('--data_dir',default='/home/workspace/ImageClassifier/flowers' ,action = 'store',type=str)
    parser.add_argument('--save_dir',type=str,dest='save_dir')
    parser.add_argument('--arch',type=str,default='vgg',dest='arch', help= 'vgg or densenet')
    parser.add_argument('--learningrate',type=float , dest='learning_rate' , default=0.0001)
    parser.add_argument('--hidden_units',type=int, default= 2048)
    parser.add_argument('--epochs', type=int, dest='epochs',default= 5)
    parser.add_argument('--gpu',dest = 'gpu' , action="store_true")
    
    arg = parser.parse_args()
    return arg    

    
    
def transformer(data_dir):

    train_dir,test_dir = data_dir
    
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

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32,shuffle= True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle= True)
    #valloader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle= True)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    result_dict = {'train':trainloader, 'test':testloader,'label':cat_to_name}
    return result_dict
    
def data_loader():
    
    arg = get_input_args()
    
    train_dir = arg.data_dir + '/train'
    #valid_dir = arg.data_dir + '/valid'
    test_dir = arg.data_dir + '/test'
    data_dir = [train_dir,test_dir]
    return transformer(data_dir)
    
def my_model(data):
    
    if arg.arch == 'vgg':
        model = models.vgg16(pretrained='true')
        input_node = 25088
    if arg.hidden_units is None: 
        hidden_units = 2048
    else:
        hidden_units = arg.hidden_units
        
    for param in model.parameters():
        param.requires_grad = False
    
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_node, hidden_units)),
                                           ('relu1', nn.ReLU()),
                                           ('fc2', nn.Linear(hidden_units,1024)),
                                           ('dropout1', nn.Dropout(p = 0.25)),
                                           ('relu2', nn.ReLU()),
                                           ('fc3', nn.Linear(1024, 102)),
                                           ('output', nn.LogSoftmax(dim=1))
                                            ]))
    
    model.classifier = classifier
    return model 
                                           
    
def train_my_model(my_model,loaded_data):
    
    print_every = 25
    steps = 0
    learn_rate = float(arg.learning_rate)
    epochs = int(arg.epochs)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(my_model.classifier.parameters(), lr=learn_rate)
    
    if arg.gpu: 
        device = 'cuda'
    else:
        device = 'cpu'
        
    trainloader = loaded_data['train']
    testloader = loaded_data['test']
   

    my_model = my_model.to(device)
    
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
                test_loss = 0
                accuracy = 0
                my_model.eval()
                
                with torch.no_grad():
                    for ii, (inputs, labels) in enumerate(testloader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        out = my_model.forward(inputs)
                        batch_loss += criterion(out, labels)
                        test_loss += batch_loss.item()

                        #calc acc
                        ps = torch.exp(out)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Training Loss: {running_loss/print_every:.3f}.. "
                      f"test Loss: {test_loss/len(testloader):.3f}.. "
                      f"Accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                my_model.train()
            
    print("Done training for now!")
    
def saver(model):
    
    if arg.save_dir is None:
        save_dir = 'check.pth'
    else:
        save_dir = args.save_dir
    
    checkpoint = {
        'epochs': epochs,
        'input_size': input_size,
        'output_size': output_size,
        'learn_rate': learn_rate,
        'batch_size': batch_size,
        'data_transforms': data_transforms,
        'model': model_to_user(pretrained = True),
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': model.classifier}
    
    torch.save(checkpoint, save_dir)
    

    
def generate_model():
    
    print(arg.data_dir)
    loaded_data = data_loader()
    model = my_model(loaded_data)
    model = train_my_model(model,loaded_data)
    saver(model)
    
    
if __name__ == '__main__':
    main()

