import keras
import keras.models as models
from keras.layers import Conv2D, Reshape, merge, BatchNormalization, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dense, Activation, Flatten, Permute, Layer, Dropout, Reshape

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
import subprocess as sbp
import json

sbp.call('clear', shell=True)     
image_height = 1024
image_width = 1280
def print_last_layer_info(model):
  '''
    takes a model as input and prints the info
    of the last layer
    
  '''
  print ("last layer information")
  print ("name: " + model.layers[-1].name)
  print ("input shape:" + str(model.layers[-1].input_shape))
  print ("output shape:" + str(model.layers[-1].output_shape))

def encoding_layers():
    kernel=3
    filter_size=64
    pad=1
    pool_size=2

    return[
        ZeroPadding2D(padding=(pad,pad), data_format="channels_first"),
        Conv2D(filter_size, (kernel, kernel), padding='valid', data_format="channels_first"),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"),
        
        ZeroPadding2D(padding=(pad,pad), data_format="channels_first"),
        Conv2D(128, (kernel, kernel), padding='valid', data_format="channels_first"),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"),


        ZeroPadding2D(padding=(pad,pad), data_format="channels_first"),
        Conv2D(256, (kernel, kernel), padding='valid', data_format="channels_first"),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"),
    ]
def decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    return[
        ZeroPadding2D(padding=(pad,pad), data_format="channels_first"),
        Conv2D(256, (kernel, kernel), padding='valid', data_format="channels_first"),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size), data_format="channels_first"),
        Dropout(0.01),
        ZeroPadding2D(padding=(pad,pad), data_format="channels_first"),
        Conv2D(128, (kernel, kernel), padding='valid', data_format="channels_first"),
        BatchNormalization(),


        UpSampling2D(size=(pool_size,pool_size), data_format="channels_first"),
        Dropout(0.01),
        ZeroPadding2D(padding=(pad,pad), data_format="channels_first"),
        Conv2D(64, (kernel, kernel), padding='valid', data_format="channels_first"),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size), data_format="channels_first"),
        Dropout(0.01),
        ZeroPadding2D(padding=(pad,pad), data_format="channels_first"),
        Conv2D(8, (kernel, kernel), padding='valid', data_format="channels_first"),
        BatchNormalization(),
    ]
def create_classification_layer():
    num_classes = 8
    kernel = 3
    return[
        Reshape((-1, 1310720)),
        # Flatten(data_format='channels_first'),
        Permute((2, 1)),
        Activation('softmax'),
    ]
def create_whole_model(save_model=False):
    save_model=save_model
    net=models.Sequential()
    net.add(Layer(input_shape=(3,image_height, image_width)))
    print_last_layer_info(net)

    for layer in encoding_layers():
        net.add(layer)
        print_last_layer_info(net)
    for layer in decoding_layers():
        net.add(layer)
        print_last_layer_info(net)
    for layer in create_classification_layer():
        net.add(layer)
        print_last_layer_info(net)
    if save_model:
        with open('segmentation_model.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(net.to_json()), indent=2))
    net.summary()
    plot_model(net, show_shapes=True, show_layer_names=True, rankdir='TB', to_file='net.png')
if __name__=='__main__':
    create_whole_model(True)