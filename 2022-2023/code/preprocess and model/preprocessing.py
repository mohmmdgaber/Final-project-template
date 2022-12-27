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
from keras.layers.merge import Concatenate
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


def patchesCsv():
    """
    returns a dataframe that saves patch name and writer in each row
    """
    writerdf = pd.read_csv('Writerset.csv')
    patches = []
    for imageFolder in os.listdir("./patches"):
        print("image: ", imageFolder)
        image = imageFolder + ".tif"
        row = writerdf[writerdf["image"] == image]
        writer = (row.iloc[0])["writer"]

        for patch in os.listdir(os.path.join("./patches",imageFolder)):
            patches.append((patch,writer))

    patchesdf = pd.DataFrame(patches, columns=["image", "writer"])
    patchesdf.to_csv('patches.csv', index=False)

    return patchesdf


def copyImages(dest_dir):
    # copy original images from Dataset to destination folder
    path = "./Dataset"
    for writer in os.listdir(path):
        writer_path = os.path.join(path, writer)
        for image in os.listdir(os.path.join(path, writer)):
            source = os.path.join(writer_path, image)
            dest = os.path.join(dest_dir, image)
            shutil.copy(source, dest)

#copyImages("./All_images")


def convert2tif(path):
    """
    converts pdf to jpg
    :param path:
    :return:
    """
    PATH = "C:\\Users\\rotem\\Downloads\\poppler-0.68.0\\bin"
    for file in os.listdir(path):
        name, extension = os.path.splitext(file)
        #print(name, ", ", extension)
        if extension == ".pdf":
            image = convert_from_path(os.path.join(path, file), poppler_path = PATH, size = (4816,6847))
            image[0].save(os.path.join(path, name) + '.tif', 'TIFF')
        if extension == ".jpg" or extension == ".jpeg":
            new_name = os.path.join(path, name + ".tif")
            os.rename(os.path.join(path, file), new_name)

#convert2tif("./All_images")

def applyAugmentation(img):
    img = img.reshape(1, img.shape[0], img.shape[1], 1)
    datagen = ImageDataGenerator(rotation_range=15, fill_mode='nearest')
    aug_iter = datagen.flow(img, batch_size=1)

    image1 = next(aug_iter)[0].astype('uint8')
    image2 = next(aug_iter)[0].astype('uint8')
    image3 = next(aug_iter)[0].astype('uint8')

    return image1, image2, image3





def generateAugImages(path = "./patches", dest = "./aug_patches"):

    patch_df = pd.read_csv("patches.csv")

    imageList = []
    writerList = []

    for folder in os.listdir(path): # folder name is image name(without patch num)
        print(folder)
        img_dir = os.path.join(path,folder)

        for patch in os.listdir(img_dir):
            print("patch: ", patch)
            image = cv2.imread(img_dir + "/" + patch, cv2.IMREAD_GRAYSCALE)
            img1, img2, img3 = applyAugmentation(image)
            dir_dest = os.path.join(dest,folder)

            if (os.path.isdir(dir_dest)) != True:
                os.mkdir(dir_dest)

            new_img1 = patch[:len(patch) - 4] + "_1.tif"
            new_img2 = patch[:len(patch) - 4] + "_2.tif"
            new_img3 = patch[:len(patch) - 4] + "_3.tif"

            cv2.imwrite(dir_dest + "/" + new_img1, img1)
            cv2.imwrite(dir_dest + "/" + new_img2, img2)
            cv2.imwrite(dir_dest + "/" + new_img3, img3)
            cv2.imwrite(dir_dest + "/" + patch, image)

            imageList.append(new_img1)
            imageList.append(new_img2)
            imageList.append(new_img3)
            imageList.append(patch)

            writer = patch_df.loc[patch_df['image'] == patch, 'writer'].iloc[0]
            for i in range(4):
                writerList.append(writer)

    aug_imgs = pd.DataFrame({"image": imageList, "writer": writerList})
    aug_imgs.to_csv('patches_aug.csv', index=False)







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


def createPairs(data, data_type):
    """"
    Creates a dataframe, columns: left image, right image, label
    Saves the dataframe to a csv file
    """
    patches = pd.read_csv("./patches.csv")
    writers = set(data["writer"])
    writers = list(writers)
    tuples = []  # list of tuples (left image, right image, label)

    for w in writers:
        # for every writer, generate one couple of the same writer
        w_imgs = data.loc[data['writer'] == w]
        diff_w_imgs = data.loc[data['writer'] != w]
        diff_w_imgs = diff_w_imgs["image"]

        for image_name in w_imgs["image"]:
            img, extension = os.path.splitext(image_name)

            for i in range(15):
                # same writer
                # for every patch, create a pair with a different patch from the same writer
                left = img + "_" + str(i) + extension
                if not left in patches.values:  # to make sure this patch exists
                    continue
                random_img = np.random.choice(w_imgs["image"], size=1, replace=False)
                random_img, extension = os.path.splitext(random_img[0])

                r = list(range(15))
                r.remove(i)
                right_patch = np.random.choice(r, size=1, replace=False) # patch number
                right = random_img + "_" + str(right_patch[0]) + extension
                while (right not in patches.values):
                    #print("patch doesnt exist: ", right)
                    right_patch = np.random.choice(r, size=1, replace=False)
                    right = random_img + "_" + str(right_patch[0]) + extension
                # print(right_patch)

                tuples.append((left, right, 1))

                # different writer
                random_img = np.random.choice(diff_w_imgs, replace=False)

                random_img, extension = os.path.splitext(random_img)
                right_patch = np.random.choice(r, size=1, replace=False)
                right = random_img + "_" + str(right_patch[0]) + extension
                while (right not in patches.values):
                    #print("patch doesnt exist: ",right)
                    right_patch = np.random.choice(r, size=1, replace=False)
                    right = random_img + "_" + str(right_patch[0]) + extension

                tuples.append((left, right, 0))

    pairs = pd.DataFrame(tuples, columns=["left", "right", "label"])
    pairs = pairs.sample(frac=1)

    if data_type == "train":
        pairs.to_csv("train_pairs.csv", index=False)
    else:
        pairs.to_csv("test_pairs.csv", index=False)

    return pairs



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
            folder = image[:len(image) - 6] #original image name

        folder_path = os.path.join(path,folder)

        # load image
        loaded = cv2.resize(cv2.imread(folder_path + "/" + image, cv2.IMREAD_GRAYSCALE), (150, 150))
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







def createImageList(data, path = "./patches/"):
    """
    returns a numpy array of pairs of images and a nparray of labels
    """
    pairs = []
    labels = []
    same = 0

    for index, row in data.iterrows():

        if (same % 2 == 0):
            left_img = row[0]
        right_img = row[1]
        label = row[2]

        if (same % 2 == 0):
            index_left = left_img.find("_", len(left_img) - 8, len(left_img))
            dir_left = left_img[:index_left]

        index_right = right_img.find("_", len(right_img) - 8, len(right_img))
        dir_right = right_img[:index_right]

        if (same % 2 == 0):
            print(left_img)
            left_img = cv2.resize(cv2.imread(path + "/" + dir_left + "/" + left_img, cv2.IMREAD_GRAYSCALE), (150, 150))
        print(right_img)
        right_img = cv2.resize(cv2.imread(path + "/" + dir_right + "/" + right_img, cv2.IMREAD_GRAYSCALE), (150, 150))

        # if (same % 2 == 0):
        #   print("img shape: ", left_img.shape)
        same = (same + 1) % 2

        pairs += [[left_img, right_img]]
        labels.append(label)

    return np.array(pairs), np.array(labels).astype("float32")



# ---------------------
# from last year

def convert_document_to_patches(path = "./All_images"):
    document_name_list = os.listdir(path)

    path_dir = './patches'
    if (os.path.isdir(path_dir)) != True:
        os.mkdir(path_dir)
    for doc_name in document_name_list:

        print(doc_name)
        path_to_save_patches = path_dir + '/' + doc_name[:len(doc_name)-4]
        if (os.path.isdir(path_to_save_patches)) != True:
            os.mkdir(path_to_save_patches)
        try:
            img = extractTextbox(doc_name)
            split_into_patches(path_to_save_patches,img,doc_name[:len(doc_name)-4],5)
        except AttributeError:
            print('cannot find text box')


def extractTextbox(document_name):
    print(document_name)
    img = PIL.Image.open("./All_images/" + document_name)
    img = img.resize((w, h))
    img = np.array(img)
    angle, xmin, xmax, ymax = find_squares2(img)

    img = rotate_image(img, angle)
    img = remove_yellow(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    img = np.array(img)
    img = img[1900:ymax - 100, xmin:xmax + 100]
    img = boundText(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def remove_yellow(img):
    # yellow in BRG mode
    yellow = np.uint8([[[0,255,255]]])
    #yellow in hsv
    hsv_yellow = cv2.cvtColor(yellow,cv2.COLOR_BGR2HSV)

    # Convert BGR image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of yellow color in HSV
    lower_yellow = np.array([20,10,10])
    upper_yellow = np.array([40,255,255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    #background to put instead of yellow
    background = np.full(img.shape, 255, dtype=np.uint8)
    # biwise or is performed only in the region of mask, all other values will be set to black in the output
    bk = cv2.bitwise_or(background, background, mask=mask)

    # combine foreground+background
    res = cv2.bitwise_or(img, bk)
    return res



def boundText(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,210,255,cv2.THRESH_BINARY)
    thresh = 255-thresh

    image, contours  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    newContours = []
    x1min = 1000
    y1min = 1000
    x2max = 0
    y2max = 0
    for cnt in image:
        if cv2.contourArea(cnt) > 150:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if x<x1min:
                x1min = x
            if y<y1min:
                y1min = y
            if x+w>x2max:
                x2max = x+w
            if y+h>y2max:
                y2max = y+h
    return im[y1min:y2max,x1min:x2max]


def split_into_patches(path, img, img_name, num_of_patches):
    # set borders to the image
    ymax, xmax = img.shape[0], img.shape[1]
    # for the patch name
    counter = 0
    # set the height and width of the patch size
    h, w = patch_size, patch_size
    # keep track of the patches that created
    x1, x2 = 0, w
    y1, y2 = 0, h

    while (counter < num_of_patches):
        if (x2 > xmax):
            x1 = 0
            x2 = w
            y1 += 240
            y2 += 240
        if (y2 > ymax):
            break
        temp = img[y1:y2, x1:x2]
        if ((x2 - x1) == patch_size and (y2 - y1) == patch_size and checkPatch(temp)):
            cv2.imwrite(path + '/' + img_name + '_' + str(counter) + '.tif', temp)
            counter += 1
        x1 += 400
        x2 += 400
    if counter < num_of_patches:
        split_into_patches_random(path, img, img_name, num_of_patches, counter)


def split_into_patches_random(path, img, img_name, num_of_patches, counter):
    # set borders to the image
    ymax, xmax = img.shape[0], img.shape[1]
    # set the height and width of the patch size
    h, w = patch_size, patch_size
    # keep track of the number of patches created
    # number of rows in the document
    numOfRows = (img.shape[0] // row_size) + 1

    while (counter < num_of_patches):
        # radom start x point of patch
        x_startPatch = random.randrange(0, xmax - patch_size)

        # radom start y point of patch
        if (numOfRows - 4 > 0):
            randomRowNumber = (random.randrange(0, numOfRows - 3))
        else:
            randomRowNumber = 0

        y_startPatch = randomRowNumber * row_size
        # end x point
        x_endPatch = x_startPatch + patch_size
        # end y point
        y_endPatch = y_startPatch + patch_size

        patch = img[y_startPatch:y_endPatch, x_startPatch:x_endPatch]

        if (checkPatch(patch)):
            counter += 1
            cv2.imwrite(path + '/' + img_name + '_' + str(counter) + '.tif', patch)


def checkPatch(img):
    h,w= img.shape[0],img.shape[1]
    threshold = 800000

    orig = img.copy()
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,img = cv2.threshold(img,240,255,cv2.THRESH_BINARY_INV)

    #Split to four quarters
    upLeft = img[0:h//2,0:w//2]
    upRight = img[0:h//2,w//2:]
    downLeft = img[h//2:,0:w//2]
    downRight = img[h//2:,w//2:]

    #claculate the sum of each quarter
    sumUpLeft = np.sum(upLeft)
    sumUpRight = np.sum(upRight)
    sumDownLeft = np.sum(downLeft)
    sumDownRight = np.sum(downRight)

    #check if bigger than the threshold
    boolUpLeft = threshold < sumUpLeft
    boolUpRight = threshold < sumUpRight
    boolDownLeft = threshold < sumDownLeft
    boolDownRight = threshold < sumDownRight

    #check if at least 3 quarters are bigger than thershold
    isValidPatch = (int(boolUpRight) + int(boolUpLeft) + int(boolDownLeft) + int(boolDownRight)) > 2

    return isValidPatch


def find_squares2(img_bgr):
    location = []
    img = img_bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    marker_count, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    segmented = cv2.watershed(img, markers)

    # END of original watershed example

    output = np.zeros_like(img)
    output2 = img.copy()
    xmin, xmax, ymax = 1000000, 0, 0

    # Iterate over all non-background labels
    for i in range(2, marker_count + 1):
        mask = np.where(segmented == i, np.uint8(255), np.uint8(0))
        x, y, w, h = cv2.boundingRect(mask)
        area = cv2.countNonZero(mask[y:y + h, x:x + w])
        location.append([x, y])

        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y > ymax:
            ymax = y

        # Visualize
        color = random.randint(0, 255 + 1)
        output[mask != 0] = color
        cv2.rectangle(output2, (x, y), (x + w, y + h), color, 1)
        cv2.putText(output2, '%d' % i, (x + w // 4, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    cv2.LINE_AA)

    angle = calc_angle([location[0][0], location[1][0]], [location[0][1], location[1][1]])
    return angle, xmin, xmax, ymax


def calc_angle(points_x, points_y):
    # Calculating the angle of the image rotation.
    # according to the squares parameters.
    if points_x[0] < points_x[1]:
        y = points_y[0]
        points_y[0] = points_y[1]
        points_y[1] = y

    a = abs(points_y[1] - points_y[0])
    b = abs(points_x[1] - points_x[0])
    c = math.sqrt(a * a + b * b)
    angle = math.acos(a / c)
    angle = 90 - math.degrees(angle)

    if points_y[1] > points_y[0]:
        return -angle
    else:
        return angle

def rotate_image(img,angle):
    # Rotate a given image.
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    #print("rotate {0}° right".format(angle))
    return rotated

# from last year
# ---------------------




# ----------------------------------------
def siamese_model(input_shape):

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))
    #model.add(Dropout(0.5))

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1, activation='sigmoid')(L1_distance)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net





#global virables
h,w= 6843,4816
patch_size = 900
row_size = 240


#*********************************************************************************

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


if __name__ == "__main__":

    input_shape=(150,150,1)
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
    #model.summary()

    #training
    adam=Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)


    #adam = Adam(learning_rate=learning_rate)
    #model = siamese_model((150, 150, 1))

    train_df, test_df, _ = splitData("patches.csv")
    train_df = train_df.sample(frac = 1)
    test_df = test_df.sample(frac = 1)

    train_writers = train_df['writer'].unique()
    test_writers = test_df['writer'].unique()

    dic = loadImageDict(train_df, "./patches", khatt = 0)
    test_dic = loadImageDict(test_df, "./patches", khatt = 0)
    print("finished building dictionaries")

    img_list = list(dic.values())
    print(img_list[0].shape)


    left, right, labels = newcreatePairs(train_df, dic, 1)
    test_left, test_right, test_labels = newcreatePairs(test_df, test_dic, 1)

    print("finished create pairs")

    # load model
    #model = keras.models.load_model('./Model/model.h5')

    # train the model
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=tf.keras.metrics.BinaryAccuracy( name="binary_accuracy", dtype=None, threshold=0.497))

    model.fit([left, right], labels, epochs=2)

    loss , acc = model.evaluate([test_left, test_right], test_labels, verbose=2)
    print("accuracy: ", acc)

    pred = model.predict([test_left, test_right])

    #model.save("./Model/model.h5")

    tn, fp, fn, tp = confusion_matrix(test_labels, (pred >= 0.49), labels = [0, 1]).ravel()
    print("tn: ", tn, " fp:", fp," fn: ", fn,"tp: ", tp)

    #print(pred)
    print("---------------------")
    hits = 0

     # pairwise
    # for i in range(0, len(pred), 2):
    #     positive = pred[i][0]
    #     negative = pred[i+1][0]
    #     print("positive: ", positive, "  negative: ", negative)
    #     if positive > negative:
    #         hits+=1

    # print("hits: ", hits)
    #print(hits / int(len(pred)/2) * 100, "%")


