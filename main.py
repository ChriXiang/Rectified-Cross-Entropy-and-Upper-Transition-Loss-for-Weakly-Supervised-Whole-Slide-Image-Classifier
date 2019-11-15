import ResNet

from keras import regularizers

from keras.layers import Subtract, multiply, dot, add, MaxPool1D,Layer,Input, Concatenate,Dense, Flatten, Dropout, GlobalMaxPooling2D,GlobalAveragePooling2D, ZeroPadding2D, Reshape, LeakyReLU, Lambda, BatchNormalization, Activation, Add, concatenate, Multiply, MaxPooling2D, Lambda
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D,UpSampling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta

from keras.utils import to_categorical, conv_utils

from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import roc_auc_score

import random

import os
import numpy as np

import keras.backend as K
from keras.preprocessing import image

from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import openslide

import h5py

import csv

from keras.utils import to_categorical
from keras.initializers import glorot_uniform

import sys
from functools import partial


def RCE_UT_loss(y_true, y_pred, weights):
    weights = K.reshape(weights, (K.shape(weights)[0], 1))
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    loss = y_true * K.log(y_pred) * weights
    rce_loss = -K.sum(loss, -1)

    l,c = y_pred.get_shape()
    preds = tf.split(y_pred,c,axis=1)

    l,c = y_true.get_shape()
    truth = tf.split(y_true,2,axis=1)

    loss2 = Lambda(lambda a: K.switch(K.greater(a[-1],truth[-1]),a[-1],K.zeros_like(a[-1])))(preds)

    ones = K.ones_like(loss2)
    tensor = Subtract()([ones,loss2])
    loss2 = K.log(tensor)

    up_loss = -K.sum(loss2,-1)

    
    loss = rce_loss + 10*up_loss

    return loss

def train(model,epochs,n_pick,t_pick,n_path,t_path):
    
    miu = 0.2
    ad = Adam(0.00001)
    BS = 32
    
    #load data configures
    x_train = []
    y_train = []
    
    name_pool = []
    for name in n_pick:
        patches = os.listdir(n_path+'/'+name+'/'+name[:-1])
        random.shuffle(patches)
        for patch in patches:
            name_pool.append((n_path+'/'+name+'/'+name[:-1]+'/'+patch,np.array([1,0])))

    for name in t_pick:
        patches = os.listdir(t_path+'/'+name+'/'+name[:-1])
        random.shuffle(patches)
        for patch in patches:
            name_pool.append((t_path+'/'+name+'/'+name[:-1]+'/'+patch,np.array([0,1])))

    #shuffle and load data
    random.shuffle(name_pool)
    for tup in name_pool:
        img = image.load_img(tup[0])
        img = image.img_to_array(img)
        
        x_train.append(img)
        y_train.append(tup[1])
        
    print('load finish')
    x_train = np.array(x_train)
    x_train /= 255
    y_train = np.array(y_train)
            
    s = []
    for e in range(1,epochs+1):
        print('epoch:',e)
        aug = ImageDataGenerator()
        loss = 0
        acc = 0
        if e == 1:
            for _ in range(len(y_train)):
                s.append(1)  
        else:
            his = model.fit_generator(aug.flow([x_train,w],y_train,batch_size=BS),steps_per_epoch=len(y_train)//BS,epochs=1,verbose=1)

            predictions = model.predict([x_train,w])
            
            #update s
            for p in range(predictions.shape[0]):
                s[p] = (1-miu)*s[p]+miu*predictions[p,np.argmax(y_train[p])]
        
        
        sep_s = []
        for idx, lable in enumerate(y_train):
            if lable[1] >= 0.99:
                sep_s.append(s[idx])
        sep_s = np.array(sep_s)
        
        #find threshold
        thred1 = np.percentile(sep_s,30)
        thred2 = np.percentile(sep_s,70)
        
        #update weights
        w = []
        for idx, lable in enumerate(y_train):
            if lable[1] >= 0.99:
                if s[idx] < thred1:
                    w.append(0)
                elif s[idx] < thred2:
                    w.append((s[idx]-thred1)/(thred2-thred1))
                else:
                    w.append(1)
            else:
                w.append(0)
        w = np.array(w)
        
        
    print('training finished')
    
    #model.save('tmp.h5')
    model.save_weights('rep.h5')
    return model

def test(model):
    test_batches = ImageDataGenerator(rescale=1./255)
    def generate_generator_multiple(generator,dir1, batch_size, img_height,img_width):
    
        genX1 = test_batches.flow_from_directory(dir1,
                                              target_size = (img_height,img_width),
                                              class_mode = 'categorical',
                                              batch_size = batch_size,
                                              shuffle=False, 
                                              seed=7)
        while True:
            X1i = genX1.next()
            yield [X1i[0], np.array([1])]  #weight does not matter in forward pass
    
    #specify WSI patches path here
    path = ""
    record_path = ""
    wsi = os.listdir(path)
   
    for idx, name in enumerate(wsi):

        print(name[:-1],'start')
        f = open(record_pass,'a')
        
        inputgenerator=generate_generator_multiple(generator=test_batches,
                                           dir1=path+'/'+name,
                                           batch_size=32,
                                           img_height=112,
                                           img_width=112)  

        predictions = model.predict_generator(inputgenerator, steps=(len(os.listdir(path+'/'+name+'/'+name[:-1]))//32)+1, verbose=1,workers=1,use_multiprocessing=True)

        count = np.argmax(predictions,axis=-1)
        tnum = np.sum(count)
        f.write(name[:-1]+'_'+str(tnum)+'_'+str(predictions.shape[0]-tnum)+'\n')
        
        print(name[:-1], 'tumor patch:',tnum,'nomral pacth',predictions.shape[0]-tnum)
        f.close()
        
if __name__ == '__main__':
    model = ResNet.ResNet50()
    
    #specify main directory of the patches
    # - n_path:
    #       - WSI1
    #           - 1_patches
    #               - 1_1.png
    #               - ...
    #       - WSI2
    #           - ...
    #       - ...
    
    n_path = "" #normal
    t_path = "" #tumor
    nwsi = os.listdir(n_path)
    twsi = os.listdir(t_path)
    random.shuffle(nwsi)
    random.shuffle(twsi)

    #train phase
    for i in range(100):
        print("Pair: "i,"***"+str(nwsi[i]),str(twsi[i]+" start ***"))
        
        #memory limit, pass only one WSI for each class
        model = train(model,5,[nwsi[i]],[twsi[i]],n_path,t_path)
    
    #test phase
    test(model)
            
