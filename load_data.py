import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path

image_height1 = 1024
image_width1 = 1280

def get_all_pixels_labels(label):
    # Since all pixels of labels are in the format of either a pixel value 
    # of multiple of 32 or 0, we seperate the pixels from the label and 
    # feed it in the max branch list to get all the varied variables.
    # input: label of shape [image_height1, image_width1]
    # output: a list with all the different values of the pixel value in input label
    max = []
    for i in range(label.shape[0]):
        for  j in range(label.shape[1]):
            if label[i, j] not in max:
                max.append(label[i, j])
    return sorted(max)
    

    
def sanity_check_for_label(label):
    # for a check wether the label is correctly modified or not. 
    # Since label is one hot encoded, it cannot be displayed easily.
    # Hence it is good to view it using this function. 
    # input: label of dimension [image_height1, image_width1, no_of_channels]
    # output: displays an the label using matplotlib.
    a = np.zeros((image_height1, image_width1))
    for k in range(8):
        for i in range(image_height1):
            for j in range(image_width1):
                if label[i, j, k]!=0:
                    a[i, j] = label[i,j,k]
    plt.imshow(a)
    plt.show()
def sanity_check_for_label_modified(label):
    # for a check wether the label is correctly modified or not. 
    # Since label is one hot encoded, it cannot be displayed easily.
    # Hence it is good to view it using this function. 
    # input: label of dimension [image_height1, image_width1, no_of_channels]
    # output: displays an the label using matplotlib.
    a = np.zeros((image_height1, image_width1))
    for k in range(8):
        for i in range(int(image_height1/2)):
            for j in range(int(image_width1/2)):
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
    # into a shape of [image_height1, image_width1, num_channels]
    # input: a 2D label of normalised pixel value between 0 and 
    #        num_channels
    # output: a modified label with above mentioned dimensions with 8 channels
    cropped_label = np.zeros((image_height1, image_width1, 8))
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

#####################################################      EXPERIMENTAL        ####################################################################
class modify():
    def __init__(self, img, lbl):
        self.image = img
        self.label = lbl
        self.rgb_image = self.modify_image()
        self.rgb_label = self.modify_label()
        self.return_function()
        
    def return_function(self):
        return self.rgb_image, self.rgb_label


    def modify_image(self):
        image = self.image
                
        img = []
        for i in range(3):
            img_ = image[i, ]
            img.append(img_)
        horizontal_split = []
        for item in img:
            a , b = self.split_into_two(item)
            horizontal_split.append(a)
            horizontal_split.append(b)
        total_split = []
        for item in horizontal_split:
            a , b = self.vertical_split(item)
            total_split.append(a)
            total_split.append(b)
        # print("len of total_split is {}".format(len(total_split)))
        rgb_image = []
        for i in range(4):
            a = np.dstack((total_split[i], total_split[i+4], total_split[i+8]))
            rgb_image.append(np.rollaxis(a, 2))
            # rgb_image.append(a)
        return rgb_image

    def vertical_split(self, image):
        return np.hsplit(image, 2)

    def split_into_two(self, image):
        return np.split(image, 2)

    def modify_label(self):
        label = np.rollaxis(self.label, 2)
        # label shape is 1024 * 1280 * 8 
        seperate_layers = []
        for i in range(8):
            array = label[i, ]
            seperate_layers.append(array)
        # print("shape of seperate layers is {} and length of layer list is {}".format(seperate_layers[0].shape, len(seperate_layers)))
        final_list = []
        for item in seperate_layers:
            a, b = self.split_into_two(item)
            a1, a2 = self.vertical_split(a)
            b1, b2 = self.vertical_split(b)
            final_list.append(a1)
            final_list.append(a2)
            final_list.append(b1)
            final_list.append(b2)
        seperate_layers = []
        for i in range(4):
            seperate_layers.append(np.dstack((final_list[i], final_list[i+4], final_list[i+8], final_list[i+12], final_list[i+16], final_list[i+20], final_list[i+24], final_list[i+28])))

    
        # print(seperate_layers[0].shape)
        # sanity_check_for_label_modified(seperate_layers[0])
        # a = np.hstack((seperate_layers[0], seperate_layers[1]))
        # b = np.hstack((seperate_layers[2], seperate_layers[3]))
        # c = np.vstack((a, b))
        # sanity_check_for_label(c)

        return seperate_layers


######################################################################################################################################################
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
        imageArr, labelArr = modify(image, label).return_function()
        for i in range(4):
            img.append(imageArr[i])
            lbl.append(labelArr[i])
    return np.array(img), np.array(lbl)




# if __name__ == '__main__':
#     img, lbl = get_more_data(1, 1)
#     imageArr, labelArr = modify(img[0], lbl[0]).return_function()
    # print("{}, {}, \n {}, {}".format(len(imageArr), len(labelArr), imageArr[0].shape, labelArr[0].shape))
    