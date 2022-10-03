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
    
    parser.add_argument('--input', action = 'store',default='./flowers/valid/100/image_07904.jpg' , help = 'image input')
    parser.add_argument('--checkpoint', action="store",default= 'check.pth', help = 'saved version of the model')
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type = int)
    parser.add_argument('--category_names',default= './cat_to_name.json' , dest="category_names", action="store")
    parser.add_argument('--gpu', action="store", dest="gpu")
    
    return  parser.parse_args()

def pocess_img(img):
    
    im = Image.open(img)
    im_size = im.size
    im = im.resize(im_size)
    im = im.crop((16,16 ,240,240))
    np_img= np.array(im)
    np_img_normal = ((np_img/255) - ([0.485, 0.456, 0.406])) / ([0.229, 0.224, 0.225])
    np_img_normal = np_img_normal.transpose((2,0,1))
    return np_img_normal
    
def load_model(check):
    
    saved_model = torch.load(check)
    model = saved_model['model']
    model.classifier = saved_model['classifier']
    model.load_state_dict(saved_model['state_dict'])
    
    return model

    
def prediction(img_path, model, device, topk, check):
    
    topk = int(topk)

    with torch.no_grad():
        im = pocess_img(img_path)
        im = torch.from_numpy(im)
        im.unsqueeze_(0)
        im = im.float()
        model = load_model(check)
        model, im = model.to(device), im.to(device)
        out = model(im)
        probs, classes = torch.exp(out).topk(topk)
        probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
        return probs, classes


def my_prediction(results, json_name):

    json_file = json.loads(open(json_name).read())
    index = 0
    for prob, klas in results:
        index = index + 1
        prob = str(round(prob,4) * 100.) + '%'
        if (json_file):
            klas = json_file.get(str(klas),'None')
        else:
            klas = ' class {}'.format(str(klas))
        print("{}.{} ({})".format(index, klas,prob))


def main():
    
    input_args = get_input_args()
    
    image_p = input_args.input
    print(image_p)
    check= input_args.checkpoint
    top_k = input_args.top_k
    json_name = input_args.category_names
    print(json_name)
    dev = input_args.gpu
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and dev else "cpu")
    model = load_model(check)
    probs, classes = prediction(image_p, model, device, top_k, check)
    result = zip(probs,classes)
    my_prediction(result, json_name)
    
if __name__ == '__main__':
    main()
    