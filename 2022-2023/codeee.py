import math
import os
import random
import shutil
import PIL
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import string
import cv2
import tensorflow as tf
from PIL.Image import Image
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
#from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import Concatenate
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.preprocessing import image
#from keras.utils import to_categorical
from keras import initializers
#from keras.engine.topology import Layer
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
#import SciPy
import keras
from tensorflow.keras import metrics
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import confusion_matrix
import shutil
def writer_to_folder(path,newpath):
    dic={}
    for filename in  os.listdir(path):
        writer=filename.split('_')[0]
        if writer in dic:
            dic[writer].append(filename)
        else:
            dic[writer]=[]
            dic[writer].append(filename)

    
    for keyed in dic.items():
        os.makedirs(newpath+'/'+keyed[0])
        for i in keyed[1]:
            try:
                shutil.copy(path+'/'+i, newpath+'/'+keyed[0]+'/')
                print("File copied successfully.")
            except shutil.SameFileError:
                 print("Source and destination represents the same file.")
 
        
    # data = pd.DataFrame(columns=["image", "writer"])

def loadImages(path = "./Dataset"):
    """
    returns a dataframe that saves image name and writer in each row
    """
    data = pd.DataFrame(columns=["image", "writer"])

    for writer in os.listdir(path):
        writer_path = os.path.join(path,writer)

        for image in os.listdir(os.path.join(path,writer)):
            data.loc[len(data)] = [image, writer]

        data.to_csv('Writerset.csv', index=False)

    return data

loadImages("./DatasetTrainWriters")




def base_network(input_dim):
    inputs = Input(shape=input_dim)
    conv_1=Conv2D(64,(5,5),padding="same",activation='relu',name='conv_1')(inputs)
    conv_1=MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2=Conv2D(128,(5,5),padding="same",activation='relu',name='conv_2')(conv_1)
    conv_2=MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3=Conv2D(256,(3,3),padding="same",activation='relu',name='conv_3')(conv_2)
    conv_3=MaxPooling2D(pool_size=(2, 2))(conv_3)
    conv_4=Conv2D(512,(3,3),padding="same",activation='relu',name='conv_4')(conv_3)
    conv_5=Conv2D(512,(3,3),padding="same",activation='relu',name='conv_5')(conv_4)
    conv_5=MaxPooling2D(pool_size=(2, 2))(conv_5)

    dense_1=Flatten()(conv_5)
    dense_1=Dense(512,activation="relu")(dense_1)
    dense_1=Dropout(0.3)(dense_1)
    dense_2=Dense(512,activation="relu")(dense_1)
    dense_2=Dropout(0.5)(dense_2)
    return Model(inputs, dense_2)

input_shape=(64,64,1)
learning_rate=0.000001

base_network = base_network(input_shape)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
processed_a = base_network(input_a)
processed_b = base_network(input_b)

fc6=concatenate([processed_a, processed_b])

fc7 = Dense(4096, activation = 'relu')(fc6)
fc7 = Dropout(0.7)(fc7)
fc8 = Dense(4096, activation = 'relu')(fc7)
fc8 = Dropout(0.8)(fc8)

fc9=Dense(1, activation='sigmoid')(fc8)
model = Model([input_a, input_b], fc9)

def splitData(path = "Writerset.csv"):
    """
    splits data to train and test
    returns test writers for page level functions
    """
    data = pd.read_csv(path)

    writers = set(data["writer"])
    writers = list(writers)
    random.shuffle(writers)
    writers_len = len(list(writers))
    train = writers[:round(writers_len * 0.80)]
    test = writers[round(writers_len * 0.80):]

    train_df = data.loc[data['writer'].isin(train)]
    test_df = data.loc[data['writer'].isin(test)]

    return train_df, test_df, test

def loadImageDict(df, path = "./aug_patches", khatt = 0):
    """
    :param df: df which we will load images from (columns: image, writer)
    :param path: path to load images from
    :return: a dict with image names as keys and loaded images as values
    """
    dict = {}
    for image in df['image'].values:
        if khatt == 0: # regular data
            index = image.find("_", image.find("_")+1) # find index of second _ in order to find folder name(= original image name)
            folder = image[:index]  # original image name
        if khatt == 1:
            if (len(image)==24):
             folder = image[:len(image) - 14] #original image name
            else:
             folder = image[:len(image) - 12] #original image name

            print(folder)
        folder_path = os.path.join(path,folder)

        # load image
        loaded = cv2.resize(cv2.imread(folder_path + "/" + image, cv2.IMREAD_GRAYSCALE), (64, 64))
        thresh, loaded = cv2.threshold(loaded, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        loaded = np.array(loaded, dtype=np.float32)
        #loaded/=255

        # add to dictionary
        dict[image] = loaded

    return dict

def newcreatePairs(df, dict, pairs_num):
    """
    :param df: df (with colums image, writer) to create pairs from
    :param dict: a dict with image names as keys and loaded images as values (from loadImageDict)
    :param pairs_num: pairs_num of pairs for same writer and pairs_num for different writer
    :return: 3 numpy arrays of left, right and labels
    """
    left = []
    right = []
    labels = []

    for index, row in df.iterrows():
        image = row["image"]
        writer = row["writer"]

        writer_images = df.loc[df['writer'] == writer]
        writer_images = writer_images["image"].values
        diff_writer_images = df.loc[df['writer'] != writer]
        diff_writer_images = diff_writer_images["image"].values

        # add pairs for same writer
        filtered_images = np.delete(writer_images, np.where(writer_images == image))
        try:
            right_patches = np.random.choice(filtered_images, size=pairs_num, replace=False)
        except:
            right_patches = np.random.choice(filtered_images, size=len(filtered_images), replace=False)
        for p in right_patches:
            left.append(dict[image])
            right.append(dict[p])
            labels.append(1)

        # add pairs for different writer
        try:
            right_patches = np.random.choice(diff_writer_images, size=pairs_num, replace=False)
        except:
            right_patches = np.random.choice(diff_writer_images, size=len(diff_writer_images), replace=False)
        for p in right_patches:
            left.append(dict[image])
            right.append(dict[p])
            labels.append(0)

        # shuffle
        # temp = list(zip(left, right, labels))
        # random.shuffle(temp)
        # left, right, labels = zip(*temp)
        # left, right, labels = list(left), list(right), list(labels)

    return np.array(left), np.array(right), np.array(labels).astype("float32")


 #training
adam=Adam(learning_rate=learning_rate)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)


    #adam = Adam(learning_rate=learning_rate)
    #model = siamese_model((150, 150, 1))

train_df, test_df, _ = splitData("patches_aug.csv")
train_df = train_df.sample(frac = 1)
test_df = test_df.sample(frac = 1)

train_writers = train_df['writer'].unique()
test_writers = test_df['writer'].unique()

dic = loadImageDict(train_df, "./aug_patches", khatt = 1)
test_dic = loadImageDict(test_df, "./aug_patches", khatt = 1)
print("finished building dictionaries")

img_list = list(dic.values())
print(img_list[0].shape)


left, right, labels = newcreatePairs(train_df, dic, 1)
test_left, test_right, test_labels = newcreatePairs(test_df, test_dic, 1)

print("finished create pairs")




model.compile(loss='binary_crossentropy', optimizer=adam, metrics=tf.keras.metrics.BinaryAccuracy( name="binary_accuracy", dtype=None, threshold=0.497))

model.fit([left, right], labels, epochs=5)

loss , acc = model.evaluate([test_left, test_right], test_labels, verbose=2)
print("accuracy: ", acc)

pred = model.predict([test_left, test_right])
