from keras.applications.imagenet_utils import obtain_input_shape
import numpy as np
import cv2
import os
import sys
import time
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Dense,MaxPool2D,Dropout,Flatten,Conv2D,GlobalAveragePooling2D,Activation
from keras.applications import DenseNet121
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.losses import CategoricalCrossentropy
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf

def data():
    DATA_PATH = "C:/personal/Object" # Path to folder containing data

    #,'scissor':np.array([0.,0.,1.,0.]),'ok':np.array([0.,0.,0.,1.])

    shape_to_label = {'yes':np.array([1.,0.,0.,0.]),'no':np.array([0.,1.,0.,0.]),'alldone':np.array([0.,0.,1.,0.]),'please':np.array([0.,0.,0.,1.])}
    arr_to_shape = {np.argmax(shape_to_label[x]):x for x in shape_to_label.keys()}

    
    imgData = list()
    lable = list()
    for dr in os.listdir(DATA_PATH):
        if dr not in ['yes','no','alldone','please']:
            continue
        print(dr)
        lb = shape_to_label[dr]
        i = 0
        for pic in os.listdir(os.path.join(DATA_PATH,dr)):
            path = os.path.join(DATA_PATH,dr+'/'+pic)
            img = cv2.imread(path)
            imgData.append([img,lb])
            imgData.append([cv2.flip(img, 1),lb]) #horizontally flipped image
            imgData.append([cv2.resize(img[50:250,50:250],(300,300)),lb]) # zoom : crop in and resize
            i+=3
        print(i)

    np.random.shuffle(imgData)

    imgData,labels = zip(*imgData)

    imgData = np.array(imgData)
    labels = np.array(labels)
    return imgData , labels


def get_model():
    model = Sequential([
        SqueezeNet(input_shape=(300, 300, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(4, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model



# densenet = DenseNet121(include_top=False, weights='imagenet', classes=3,input_shape=(300,300,3))
# densenet.trainable= True

# def genericModel(base):
#     model = Sequential()
#     model.add(base)
#     model.add(MaxPool2D())
#     model.add(Flatten())
#     model.add(Dense(3,activation='softmax'))
#     model.compile(optimizer = Adam() , loss='categorical_crossentropy' , metrics=['accuracy'])
#     #model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
#     return model

imgData = list()
labels = list()
imgData , labels = data()
model = get_model()
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# checkpoint = ModelCheckpoint(
#     'model.h5', 
#     monitor='val_acc', 
#     verbose=1, 
#     save_best_only=True, 
#     save_weights_only=True,
#     mode='auto'
# )

# es = EarlyStopping(patience = 3)

model.fit(imgData, labels, epochs=10)

# save the model for later use
model.save("rock-paper-scissors-model.h5")

# history = dnet.fit(
#     x=imgData,
#     y=labels,
#     batch_size = 16,
#     epochs=8,
#     callbacks=[checkpoint,es],
#     validation_split=0.2
# )

# dnet.save_weights('model.h5')

# with open("model.json", "w") as json_file:
#     json_file.write(dnet.to_json())