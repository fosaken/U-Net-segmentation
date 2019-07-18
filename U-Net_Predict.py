#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:CaoZhihui

from __future__ import print_function

import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Reshape, Activation, core, \
    Permute

from keras import backend as K
import os
import configparser as ConfigParser
import warnings  # 不显示乱七八糟的warning
from base_functions import get_test_data, pred_to_imgs
import h5py
from keras.models import model_from_json
from PIL import Image
from IPython.terminal.tests.test_help import test_profile_list_help

warnings.filterwarnings("ignore")
K.set_image_dim_ordering('th')

# --read configuration file and get parameters-- #
config = ConfigParser.RawConfigParser()
config.read('./configuration.txt')
path_local = config.get('unet_parameters', 'path_local')
model_path = path_local + config.get('unet_parameters', 'unet_model_dir')
test_images_dir = path_local + config.get('unet_parameters', 'test_images_dir')
test_labels_dir = path_local + config.get('unet_parameters', 'test_labels_dir')
img_h = int(config.get('unet_parameters', 'img_h'))
img_w = int(config.get('unet_parameters', 'img_w'))
C = int(config.get('unet_parameters', 'C'))

print('-'*30)
print('Loading model and weights...')
print('-'*30)
model = model_from_json(open(model_path + 'unet.json').read())
model.load_weights(model_path+'unet_weights.h5')
print('Loading test data...')
print('-'*30)
test_x, test_y = get_test_data(test_images_dir, test_labels_dir, img_h, img_w)
print('Predicting...')
predictions = model.predict(test_x)

pred_images = pred_to_imgs(predictions, img_h, img_w, C=C)
pred_images /= np.max(pred_images)

print('-' * 30 + '\n' + 'Saving predicted images...' + '\n' + '-' * 30)
for i in range(pred_images.shape[0]):
    img = Image.fromarray(pred_images[i]*255)
    img = img.convert('L')
    img.save('./U-Net/pred_images/pred_' + str(i) + '.png')


