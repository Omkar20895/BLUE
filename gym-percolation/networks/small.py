import json
import pickle
import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.initializers import glorot_normal, Zeros, RandomNormal


def get_model():
    input1 = Input(shape=(5, 5, 2))
    kernel = RandomNormal(mean=0.0, stddev=0.05, seed=None)

    conv1 = Conv2D(128, kernel_size=2, activation='relu', kernel_initializer=kernel)(input1)
    conv1 = MaxPooling2D((2,2))(conv1)
    #conv1 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer=kernel)(conv1)
    #conv1 = MaxPooling2D((2,2))(conv1)
    conv1 = Flatten()(conv1)

    #final_model = concatenate([conv1, conv2])


    final_model = Dense(256, activation='relu', kernel_initializer=kernel)(conv1)
    final_model = Dense(128, activation='relu', kernel_initializer=kernel)(final_model)
    final_model = Dense(32, activation='relu', kernel_initializer=kernel)(final_model)
    final_model = Dense(1, activation='linear')(final_model)

    model = Model(inputs=input1, outputs=final_model)

    sgd = Adam(lr=0.01)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mse'])

    print(model.summary())
    return model
