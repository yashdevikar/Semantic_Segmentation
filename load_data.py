import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path

image_height = 1024
image_width = 1280

def get_all_pixels_labels(label):
    # Since all pixels of labels are in the format of either a pixel value 
    # of multiple of 32 or 0, we seperate the pixels from the label and 
    # feed it in the max branch list to get all the varied variables.
    # input: label of shape [image_height, image_width]
    # output: a list with all the different values of the pixel value in input label
    max = []
    for i in range(label.shape[0]):
        for  j in range(label.shape[1]):
            if label[i, j] not in max:
                max.append(label[i, j])
    return sorted(max)

def divide_images_into_small_parts(image=None):
    if image is None:
        raise Exception("Image given to divide_images_into_small_parts function cannot be None")
    image_height = image.shape[1]
    image_width = image.shape[2]
    
    # trying to divide image in 4 folds
    a1 = np.zeros((image.shape[0], int(image_height/2), int(image_width/2) ))
    a2 = np.zeros((image.shape[0], int(image_height/2), int(image_width/2) ))
    a3 = np.zeros((image.shape[0], int(image_height/2), int(image_width/2) ))
    a4 = np.zeros((image.shape[0], int(image_height/2), int(image_width/2) ))
    
    
def sanity_check_for_label(label):
    # for a check wether the label is correctly modified or not. 
    # Since label is one hot encoded, it cannot be displayed easily.
    # Hence it is good to view it using this function. 
    # input: label of dimension [image_height, image_width, no_of_channels]
    # output: displays an the label using matplotlib.
    a = np.zeros((image_height, image_width))
    for k in range(8):
        for i in range(image_height):
            for j in range(image_width):
                if label[i, j, k]!=0:
                    a[i, j] = label[i,j,k]
    plt.imshow(a)
    plt.show()
def normalized(rgb):
    # Normalises Histogram. Function directly copied from opencv official tutorials
    # from opencv website.
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)
    b = rgb[:,:,0]
    g = rgb[:,:,1]
    r = rgb[:,:,2]
    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)
    return norm
def preprocess_label(label):
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            label[i, j] = label[i, j]//32
    return one_hot_it(label)
def one_hot_it(label):
    # Does one hot encoding of the images. Seperates various 
    # pixel values from the labels and converts the 2D array 
    # into a shape of [image_height, image_width, num_channels]
    # input: a 2D label of normalised pixel value between 0 and 
    #        num_channels
    # output: a modified label with above mentioned dimensions with 8 channels
    cropped_label = np.zeros((image_height, image_width, 8))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            cropped_label[i, j, label[i,j]] = label[i, j]
    return cropped_label
def get_file_path(indice=1):
    parent_dir = Path("data")
    instrument_dir_path = os.path.join(parent_dir, 'cropped_train', 'instrument_dataset_{}'.format(indice))
    images_path = os.path.join(os.getcwd(), instrument_dir_path, 'images')
    labels_path = os.path.join(os.getcwd(), instrument_dir_path, 'instruments_masks')
    if os.path.isdir(images_path):
        return images_path, labels_path
    else:
        raise Exception("Images path doesnt exist")
def get_one_data(indice, index):
    images_path, labels_path = get_file_path(indice)
    
    image = cv2.imread(os.path.join(images_path, "frame{0:0=3d}.jpg".format(index)))
    image = normalized(image)
    image = np.rollaxis(np.array(image), 2)

    label = os.path.join(labels_path, "frame{0:0=3d}.png".format(index))
    label = np.array(cv2.imread(label, cv2.IMREAD_GRAYSCALE))
    label = preprocess_label(label)
    return image, label
def get_more_data(indice=1, sample_size=1):
    if indice>4 or indice<0:
        raise Exception("Indice of instrument folder is invalid")
    img = []
    lbl = []
    images_path, labels_path = get_file_path(indice)
    dataset = sorted(list(os.listdir(images_path))) 
    dataset_size = len(dataset)
    if sample_size>dataset_size:
        raise Exception("Sample size {} cannot be greater than dataset of size {}".format(sample_size, dataset_size))
    if sample_size>0:
        index = random.sample(range(0, dataset_size), sample_size)
    else:
        index = range(0, dataset_size)
    for ind in index:
        image, label = get_one_data(indice, ind)
        img.append(image)
        lbl.append(label)
    return np.array(img), np.array(lbl)


if __name__ == '__main__':
    # img, lbl = get_more_data(1, 5)
    # print(img.shape, lbl.shape)
    divide_images_into_small_parts(np.zeros((3, 1024, 1280)))