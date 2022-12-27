# training by patches
# predict by page level
import os.path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import preprocessing


def createTestPairs(dict, test_writers):
    """
    :param dict: a dict with image names as keys and loaded images as values (from loadImageDict)
    :return: 3 numpy arrays of left, right and labels
    """
    df = pd.read_csv("Writerset.csv")

    # filter out writers that are not in test
    boolean_series = df.writer.isin(test_writers)
    df = df[boolean_series]

    left = []
    right = []
    labels = []

    # for every writer, generate one couple of the same writer and one couple of different
    for w in test_writers:

        writer_imgs = df.loc[df['writer'] == w]
        diff_w_imgs = df.loc[df['writer'] != w]
        #diff_w_imgs = diff_w_imgs["image"]

        # create a pair for the same writer
        if len(writer_imgs) > 1:
            random_img = preprocessing.np.random.choice(writer_imgs["image"], size=2, replace=False)

            left_image_folder = os.path.join("patches", (random_img[0])[:len(random_img[0]) - 4])
            right_image_folder = os.path.join("patches", (random_img[1])[:len(random_img[1]) - 4])

            left_patches = os.listdir(left_image_folder)
            right_patches = os.listdir(right_image_folder)

            patch_num = min(len(left_patches), len(right_patches))
            for i in range(patch_num):
                left.append(dict[left_patches[i]])
                right.append(dict[right_patches[i]])
                labels.append(1)

        else: # if the writer has only 1 image
            image = writer_imgs['image'].values[0]
            patches = os.listdir(os.path.join("patches", image[:len(image) - 4]))
            left_patches = patches[:len(patches)//2]
            # for it to have same amount of images
            right_patches = patches[len(patches)//2:len(patches)//2 + len(left_patches)]

            for i in range(len(left_patches)):
                left.append(dict[left_patches[i]])
                right.append(dict[right_patches[i]])
                labels.append(1)

        # create one pair with a different writer
        left_img = preprocessing.np.random.choice(writer_imgs["image"], size=1, replace=False)
        right_img = preprocessing.np.random.choice(diff_w_imgs["image"], size=1, replace=False)

        left_image_folder = os.path.join("patches", (left_img[0])[:len(left_img) - 4])
        right_image_folder = os.path.join("patches", (right_img[0])[:len(right_img) - 4])

        left_patches = os.listdir(left_image_folder)
        right_patches = os.listdir(right_image_folder)

        patch_num = min(len(left_patches), len(right_patches))
        for i in range(patch_num):
            left.append(dict[left_patches[i]])
            right.append(dict[right_patches[i]])
            labels.append(0)

        # shuffle
        # temp = list(zip(left, right, labels))
        # preprocessing.random.shuffle(temp)
        # left, right, labels = zip(*temp)
        # left, right, labels = list(left), list(right), list(labels)

    return np.array(left), np.array(right), np.array(labels).astype("float32")



def createTestPairs2(dict, test_writers):
    df = pd.read_csv("Writerset.csv")

    # filter out writers that are not in test
    boolean_series = df.writer.isin(test_writers)
    df = df[boolean_series]

    left = []
    right = []
    labels = []

    for w in test_writers:
        writer_imgs = df.loc[df['writer'] == w]
        diff_w_imgs = df.loc[df['writer'] != w]

        left_img = preprocessing.np.random.choice(writer_imgs["image"], size=1, replace=False)  # writers image
        right_img = preprocessing.np.random.choice(diff_w_imgs["image"], size=1, replace=False)  # different writer

        left_image_folder = os.path.join("patches", (left_img[0])[:len(left_img[0]) - 4])
        right_image_folder = os.path.join("patches", (right_img[0])[:len(right_img[0]) - 4])

        left_patches = os.listdir(left_image_folder)
        right_patches = os.listdir(right_image_folder)

        patch_num = min(len(left_patches), len(right_patches))
        for i in range(patch_num // 2):
            # add patches for different writer
            left.append(dict[left_patches[i]])
            right.append(dict[right_patches[i]])
            labels.append(0)

            # add patches for same writer
            left.append(dict[left_patches[i]])
            right.append(dict[left_patches[patch_num - i - 1]])
            labels.append(1)

    return np.array(left), np.array(right), np.array(labels).astype("float32")

if __name__ == "__main__":

    # global variables
    h, w = 6843, 4816
    patch_size = 900
    row_size = 240


    input_shape = (150, 150, 1)
    learning_rate = 0.00001
    base_network = preprocessing.base_network(input_shape)
    input_a = preprocessing.Input(shape=input_shape)
    input_b = preprocessing.Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    fc6 = preprocessing.concatenate([processed_a, processed_b])

    fc7 = preprocessing.Dense(4096, activation='relu')(fc6)
    fc7 = preprocessing.Dropout(0.7)(fc7)
    fc8 = preprocessing.Dense(4096, activation='relu')(fc7)
    fc8 = preprocessing.Dropout(0.8)(fc8)

    fc9 = preprocessing.Dense(1, activation='sigmoid')(fc8)
    model = preprocessing.Model([input_a, input_b], fc9)
    # model.summary()

    # training
    adam = preprocessing.Adam(learning_rate=learning_rate)
    loss_fn = preprocessing.keras.losses.BinaryCrossentropy(from_logits=True)


    # adam = preprocessing.Adam(learning_rate=0.005)
    # model = preprocessing.siamese_model((150, 150, 1))

    train_df, test_df, test_writers = preprocessing.splitData("./patches.csv")
    train_df = train_df.sample(frac = 1)
    test_df = test_df.sample(frac = 1)


    train_writers = train_df['writer'].unique()
    test_writers = test_df['writer'].unique()


    # training
    train_dic = preprocessing.loadImageDict(train_df, "./patches", khatt = 0)
    left, right, labels = preprocessing.newcreatePairs(train_df, train_dic, 1)

    test_dic = preprocessing.loadImageDict(test_df, "./patches", khatt = 0)
    print("finished test dictionary")
    test_left, test_right, test_labels = createTestPairs2(test_dic, test_writers)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=preprocessing.tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5))

    model.fit([left, right], labels, epochs=101)

    loss, acc = model.evaluate([test_left, test_right], test_labels, verbose=2)
    #pred = model.predict([x_test_1, x_test_2], t_labels)
    #print(pred)
    print("accuracy: ", acc)

    pred = model.predict([test_left, test_right])

    hits, mini_hits = 0, 0
    sum, count = 0, 0

    for i in range(0, len(test_labels), 2):

        negative = pred[i][0]
        positive = pred[i + 1][0]
        print("positive: ", positive, "  negative: ", negative)
        if positive > negative:
            mini_hits += 1

        count += 1

        # if we finished an image
        if count == 7:
            if mini_hits > 3:
                hits += 1

            count = 0
            mini_hits = 0

    writers_num =  len(test_writers)

    print(hits / writers_num * 100, "%")

