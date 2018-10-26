import numpy as np
import cv2

def img_to_encoding_top(image, model):
    side = min(image.shape[0], image.shape[1])
    image = image[0:side, 0:side]
    img = cv2.resize(image, (96, 96))
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def img_to_encoding_bot(image, model):
    side = min(image.shape[0], image.shape[1])
    s1 = image.shape[0] - side
    s2 = image.shape[1] - side
    image = image[s1:s1+side, s2:s2+side]
    img = cv2.resize(image, (96, 96))
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def img_to_encoding(image, model):
    side = min(image.shape[0], image.shape[1])
    s1 = int((image.shape[0] - side) / 2)
    s2 = int((image.shape[1] - side) / 2)
    image = image[s1:s1+side, s2:s2+side]
    img = cv2.resize(image, (96, 96))
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def img_to_encoding_range(sz, x, y, image, model):
    image = image[x:x+sz,y:y+sz]
    img = cv2.resize(image, (96, 96))
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding
