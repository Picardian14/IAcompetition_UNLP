#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:32:00 2019

@author: mrmister
"""
import os
from PIL import Image
from matplotlib import image, pyplot

train_data_path = '/home/mrmister/Ivan/Facultad/4to/deeplearning/gcloud/models/torax/trainer/chest_xray/train/'
"""        
train_images_negative = []    
for filename in os.listdir(train_data_path + 'normal/'):
    img = Image.open(train_data_path + 'normal/' + filename)
    cropped = img.crop((100,50, 500,600))
    cropped.save(train_data_path + 'normal/' + 'cropped' + filename, format='jpeg')
"""
train_images_positive = []
for filename in os.listdir(train_data_path + 'pneumonia'):
    img = Image.open(train_data_path + 'pneumonia/' + filename)
    cropped = img.crop((100,50, 500,600))
    cropped.save(train_data_path + 'pneumonia/' + 'cropped' + filename, format='jpeg')
    
val_data_path = '/home/mrmister/Ivan/Facultad/4to/deeplearning/gcloud/models/torax/trainer/chest_xray/val/'
    
val_images_negative = []    
for filename in os.listdir(val_data_path + 'normal/'):
    img = Image.open(val_data_path + 'normal/' + filename)
    cropped = img.crop((100,50, 500,600))
    cropped.save(val_data_path + 'normal/' + 'cropped' + filename, format='jpeg')
    
    
val_images_positive = []
for filename in os.listdir(val_data_path + 'pneumonia'):
    img = Image.open(val_data_path + 'pneumonia/' + filename)
    cropped = img.crop((100,50, 500,600))
    cropped.save(val_data_path + 'pneumonia/' + 'cropped' + filename, format='jpeg')
