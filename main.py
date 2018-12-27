# TRADITIONAL IMPORTS
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
import cv2
import matplotlib.pyplot as plt
import os
import json
import subprocess as sbp
sbp.call('clear', shell=True)
#USE TENSORFLOW AS BACKEND
import tensorflow
from model import *

#DATALOADING ALGORITHMS IMPORTS 
from load_data import *


#GLOBAL PARAMETERS 
image_height = 512
image_width = 640
data_shape = image_height*image_width

class_weighting= [0.0001, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418]


def train_file():
    with open('segmentation_model.json') as file:
        net=keras.models.model_from_json(file.read())

    net.summary()
    net.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["categorical_accuracy"])
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    earlystopcheck = EarlyStopping(monitor="loss", min_delta=0.00001, patience=3)

    callbacks_list = [checkpoint, earlystopcheck]
    nb_epoch = 1
    batch_size = 1 # 6
    num_iters = 1 # 10
    training_loss = []
    val_loss = []
    training_acc = []
    val_acc = []

    for iter in range(0, num_iters):
    
        train_subset_size = 1 # 12
        val_subset_size = 1 # 5
        
        # sample training data
        train_data, train_label = get_more_data(1, train_subset_size)
        print(train_label.shape)
        train_label = train_label.reshape((4,327680, 8))
        
        # sample validation data
        val_data, val_label = get_more_data(2, val_subset_size)
        print("val_label shape {}".format(val_label.shape))
        val_label = np.reshape(val_label, (4, data_shape, 8))
        
        print ("Iteration # {0}".format(iter))
        print ("Generating sample size {0} from train_data".format(train_subset_size))

        history = net.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch, verbose=1, class_weight=class_weighting, callbacks=callbacks_list, validation_data=(val_data, val_label), shuffle=True)

        training_loss += history.history['loss']
        val_loss += history.history['val_loss']
        training_acc += history.history['categorical_accuracy']
        val_acc += history.history['val_categorical_accuracy']



if __name__=='__main__':
    train_file()