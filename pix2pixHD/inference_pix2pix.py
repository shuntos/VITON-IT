import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models_.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import cv2

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.label_nc = 0 
opt.netG = "global"
opt.no_instance = True
opt.dataroot = "/root/Deep_learning/santosh/pix2pixHD/datasets/vd2.0_2/"   
opt.name = "vd2.0_2"


def main(img_file):
    data_loader = CreateDataLoader(opt, file_= img_file)
    dataset = data_loader.load_data()

    model = create_model(opt)
    mask = cv2.imread(img_file)
    orig = mask.copy()

    for i, data in enumerate(dataset):
        generated = model.inference(data['label'], data['inst'], data['image'])
        img = util.tensor2im(generated.data[0])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        outfile = "output/wrapped_"+".jpg"
        cv2.imwrite(outfile,img)

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

        img = cv2.resize(img,(orig.shape[1], orig.shape[0]))
        return img, mask
            
