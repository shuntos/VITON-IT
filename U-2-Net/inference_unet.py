import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from PIL import Image
from skimage import io
import cv2
# normalize the predicted SOD probability map

transform_=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])

def image_loader(image):
    """load image, returns cuda tensor"""
 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sample = {'imidx':np.array([0]), 'image':image, 'label':image}
    shape= image.shape
    sample = transform_(sample)   # Perform rescale and color  format conversion RGB to tensorlab
    image = sample['image']
    image  = torch.unsqueeze(image,0)

    return image,[shape[1],shape[0],3] #assumes that you're using GPU


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def main(img_original):

    # --------- 1. get image path and name ---------
    model_name='u2net'

    model_dir = "saved_models/u2netnew_bce_itr_174000_train_0.113706_tar_0.007825.pth"

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    #net.load_state_dict(torch.load(model_dir))
    device = torch.device('cpu')
    net.load_state_dict(torch.load(model_dir,map_location=device))
    #if torch.cuda.is_available():
        #net.cuda()
    net.eval()


    original = img_original.copy()
    imgx = img_original.copy()
    inputs_test, im_size = image_loader(imgx)

    print(inputs_test.shape)
    bk = np.full(img_original.shape, 255, dtype=np.uint8)  # white bk, same size and type of image

    inputs_test = inputs_test.type(torch.FloatTensor)
    
    inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

    pred = d4[:,0,:,:]
    pred = normPRED(pred)

    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')

    img = im.resize((im_size[0],im_size[1]),resample=Image.BILINEAR)

    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray,1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return mask 


if __name__ == "__main__":
    img_file = "imgs/1.jpeg"
    filename, ext = os.path.splitext(img_file)
    outfile = filename+"_mask"+".jpg"
    img = cv2.imread(img_file)
    mask = main(img)
    cv2.imwrite(outfile,mask)
