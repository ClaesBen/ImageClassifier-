import argparse
import numpy as np
import pandas as pd 
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

def get_input_args():
    
    parser = argparse.ArgumentParser(description = 'Parser for predict.py')
    
    parser.add_argument('--input', action = 'store', help = 'image input')
    parser.add_argument('--checkpoint', action="store", help = 'saved version of the model')
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type = int)
    parser.add_argument('--category_names', dest="category_names", action="store")
    parser.add_argument('--gpu', action="store", dest="gpu")
    
    return  parser.parse_args()

def pocess_img(img):

    im_size = 256,256
    im = Image.open(img)
    im = im.resize(im_size)
    im = im.crop(16, 16 , 240, 240)
    np_img= np.array(im)
    np_img_mormal = ((np_img/255) - ([0.485, 0.456, 0.406])) / ([0.229, 0.224, 0.225])
    np_img_mormal = np_image_normal.transpose((2,0,1))
    return np_img_normal
    
def load_model():
    
    saved_model = torch.load(arg.checkpoint)
    model = saved_model['model']
    model.classifier = saved_model['classifier']
    model.load_state_dict(saved_model['state_dict'])
    
    return model

    
def prediction(img_path, model, device, topk):
    
    topk = int(topk)
    
    with torch.no_grad():
        img = torch.from_numpy(process_img(img_path))
        img.unsqueeze(0)
        img = img.float()
        model, img = model.to(device), img.to(device)
        model.eval()
        probs, classes = torch.exp(model(img)).topk(top_k)
        probs, classes = probs[0].tolist(),classes[0].add(1).tolist()
        return probs, classes
                 
    
def my_prediction(probs, classes, json_name):
    
    if json_name is None:
        cat_class = classes 
    else:
        with open(json_name, 'r') as f:
            json_name_data = json.load(f)
        cat_class = [json_name_data[i] for i in classes]
    df = pd.Dataframe({'Class' : pd.Series(data = json_name), 'Vals' : pd.Series(data = probs, dtype='float64')})
    print(df)


def main():
    
    global arg
    arg = get_input_args()
    
    image_p = arg.input
    check= arg.checkpoint
    top_k = arg.top_k
    json_name = arg.category_names
    dev = arg.gpu
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and dev else "cpu")
    model = load_model()
    probs, classes = prediction(image_p, model, device, top_k)
    my_prediction(probs, classes, json_name)
    
if __name__ == '__main__':
    main()
    