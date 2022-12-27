import os
from os.path import exists
from matplotlib.style import context
import pymongo
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.contrib.sessions.models import Session
from sklearn import preprocessing
from .models import Compare, Image
import gridfs
from django.contrib import messages
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow import keras
from django.template import loader
from django.contrib.auth.models import User


def changeModel(request):
    return render(request, 'changeModel.html')

def changeModelSubmit(request):
    try:
        model = request.FILES['model']
    except:
        messages.add_message(request, messages.ERROR, 'Please upload a model in h5 format')
        return render(request, 'changeModel.html')

    modelname = str(model.name)
    if not modelname.endswith(".h5"):
        messages.add_message(request, messages.ERROR, 'Please upload a model in h5 format')
        return render(request, 'changeModel.html')

    # success
    messages.add_message(request, messages.SUCCESS, 'Model has been changed')

    path = os.path.join(settings.MEDIA_ROOT, "model.h5")

    if exists(path):
        os.remove(path) 

    fs = FileSystemStorage() #defaults to MEDIA_ROOT  
    filename = fs.save("model.h5", model)

    return render(request, 'changeModel.html')


def history(request):
    context = {}
    context["compare"] = Compare.objects.filter(user = request.user.username)
    
    return render(request, 'history.html', context=context)


def upload(request):
    return render(request, 'upload.html')

def result(request):
    try:
        img1 = request.FILES['img1']
        img2 = request.FILES['img2']
    except:
        messages.add_message(request, messages.INFO, 'Please upload two images')
        return render(request, 'upload.html')

    image1 = {
        'image': img1,
        'title': img1.name,
        'user': request.user,
    }

    image2 = {
        'image': img2,
        'title': img2.name,
        'user': request.user,
    }

    doc1 = Image(**image1)
    doc1.save()
    doc2 = Image(**image2)
    doc2.save()


    my_client = pymongo.MongoClient(settings.DATABASES['default']['CLIENT']['host'])
    writerid = my_client['writerid']
    img_col = writerid["images"]

    # create an instance of gridfs
    fs = gridfs.GridFS(writerid)

    with open(os.path.join(settings.MEDIA_ROOT, str(img1)) , 'rb') as f:
        contents = f.read()
    fs.put(contents, filename=img1.name)

    with open(os.path.join(settings.MEDIA_ROOT, str(img2)) , 'rb') as f:
        contents = f.read()
    fs.put(contents, filename=img2.name)

    img_1 = cv2.imread(os.path.join("./media" ,str(image1['image'])), cv2.IMREAD_COLOR)
    img_2 = cv2.imread(os.path.join("./media" ,str(image2['image'])), cv2.IMREAD_COLOR)

    patches1, patches2 = preprocess(img_1, str(image1['image']), img_2, str(image2['image']))

    pred = model_predict(patches1, patches2)

    comaprison = {
        'image1': img1,
        'image2': img2,
        'user': request.user,
        'result': pred
    }

    cmpr = Compare(**comaprison)
    cmpr.save()

    context = {
        'pred' : pred,
        'img1' : "/media/" + str(doc1.image), 
        'img2' : "/media/" + str(doc2.image),
    }

    return render(request, 'result.html', context)



def preprocess(img1, imgname1, img2, imgname2):

    # img1 = remove_yellow(img1)
    # img2 = remove_yellow(img2)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    thresh, loaded1 = cv2.threshold(img1, 180, 255, cv2.THRESH_BINARY_INV)
    thresh, loaded2 = cv2.threshold(img2, 180, 255, cv2.THRESH_BINARY_INV)

    cv2.imwrite("./media/1.jpg", loaded1)

    patches1 = split_into_patches("./media", loaded1, imgname1, 15)
    patches2 = split_into_patches("./media", loaded2, imgname2, 15)

    return patches1, patches2



def model_predict(patches1, patches2):

    model = keras.models.load_model('./media/model.h5')
    
    sum = 0
    for i in range(len(patches1)):

        patches1[i] = np.array(patches1[i], dtype = np.float32)
        patches2[i] = np.array(patches2[i], dtype = np.float32)

        np.expand_dims(patches1[i], axis=0)
        np.expand_dims(patches2[i], axis=0)

        patches1[i] = cv2.resize(patches1[i], (150,150))
        patches2[i] = cv2.resize(patches2[i], (150,150))

        # patches1[i]/=255
        # patches2[i]/=255

        patches1[i] = np.reshape(patches1[i], (1,150,150,1))
        patches2[i] = np.reshape(patches2[i], (1,150,150,1))


        pred = model.predict([np.array(patches1[i]),np.array(patches2[i])])

        print(pred[0][0])

        thresh = 0.8

        if pred[0][0] > thresh:
            sum += 1




    if sum > (len(patches1) // 2):
        result = True
    else:
        result = False

    return result





# PREPROCESSING


patch_size = 150
row_size = 240

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

    patches = []

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
            patches.append(temp)
            counter += 1
        x1 += 400
        x2 += 400
    if counter < num_of_patches:
        patches = split_into_patches_random(patches, img, img_name, num_of_patches, counter)

    cv2.imwrite("./media/patchtest.jpg", patches[0])
    return patches

def split_into_patches_random(patches, img, img_name, num_of_patches, counter):
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
            patches.append(patch)
    return patches


def checkPatch(img):
    h,w= img.shape[0],img.shape[1]
    threshold = 200000

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