#!/usr/bin/env python
# coding: utf-8

import json
import pickle
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_curve, mean_squared_error
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.colors import ListedColormap
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, BatchNormalization, LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop
from keras.initializers import glorot_normal, Zeros, RandomNormal



def get_model_data_stacked(file_path):
    X1_train = []
    X2_train = []
    Y_train = []
    
    print("Loading data...")
    try:
        with open(file_path) as json_file:
            json_data = json.load(json_file)
        for key, value in json_data.items():
            for index in range(len(value)):
                array_board = np.array(value[index]['board_state'])
                array_board = np.reshape(array_board, (5, 5, 1))
                #X1_train.append(array)
                Y_train.append(value[index]['move_quality'])
                array_action = np.zeros((5, 5, 1))
                array_action[value[index]['action']['x']][value[index]['action']['y']] = 1
                #X2_train.append(array)
                stacked_array = np.dstack((array_board, array_action))
                X1_train.append(stacked_array)
        
        print("Done")
    except Exception as e:
        print(e)
        
    X = pd.DataFrame({'X':X1_train, 'Y':Y_train})
    X = shuffle(X)
    print(type(X["X"][0]))
    print(X["X"][0].shape)
    print("Done")
    X, Y_train = X["X"], X["Y"]
    Y_train = (Y_train>0).astype(float)
    return X, list(Y_train)

def encode(observation, red_equals_black = False):
    black, red, blue, green = [0, 1, 2, 3]

    black_layer = observation == black
    red_layer = observation == red
    blue_layer = observation == blue
    green_layer = observation == green

    if red_equals_black:
        new_board = np.dstack((black_layer|red_layer, blue_layer, green_layer))
    else:
        new_board = np.dstack((black_layer, red_layer, blue_layer, green_layer))

    return new_board.astype(int)

def get_model_data_policy_gradient(file_path):

    X1_train = []
    X2_train = []
    Y_train = []
    
    print("Loading data...")
    try:
        with open(file_path) as json_file:
            json_data = json.load(json_file)
        for key, value in json_data.items():
            for index in range(len(value)):
                array_board = np.array(value[index]['board_state'])
                array_board = np.reshape(array_board, (5, 5, 1))
                array_board = encode(array_board, True)

                policy_array = np.zeros((5, 5, 1))
                if value[index]['move_quality'] == 1:
                    policy_array[value[index]['action']['x']][value[index]['action']['y']] = 1
                #else:
                #    policy_array[value[index]['action']['x']][value[index]['action']['y']] = -1
                Y_train.append(list(policy_array.flatten()))
                X1_train.append(array_board)

        print("Done")
    except Exception as e:
        print(e)

    X = pd.DataFrame({'X':X1_train, 'Y':Y_train})
    print(X.shape)
    X = shuffle(X)
    X, Y_train = X["X"], X["Y"]
    return X, list(Y_train)

def get_model_stacked():

    kernel = RandomNormal(mean=0.0, stddev=0.05, seed=None)
    input1 = Input(shape=(5, 5, 2))
    
    conv1 = Conv2D(128, kernel_size=2, activation='relu', kernel_initializer=kernel)(input1)
    conv1 = MaxPooling2D((2,2))(conv1)
    #conv1 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer=kernel)(conv1)
    #conv1 = MaxPooling2D((2,2))(conv1)
    conv1 = Flatten()(conv1)
    
    #final_model = concatenate([conv1, conv2])
    
    
    final_model = Dense(256, activation='relu', kernel_initializer=kernel)(conv1)
    final_model = Dense(128, activation='relu', kernel_initializer=kernel)(final_model)
    final_model = Dense(32, activation='relu', kernel_initializer=kernel)(final_model)
    final_model = Dense(1, activation='sigmoid')(final_model)
    
    model = Model(inputs=input1, outputs=final_model)
    
    sgd = Adam(lr=0.01)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    
    print(model.summary())
    return model

def get_model():

    input1 = Input(shape=(5, 5, 2))
    kernel = RandomNormal(mean=0.0, stddev=0.05, seed=None)
    
    conv1 = Conv2D(128, kernel_size=2, kernel_initializer=kernel)(input1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = MaxPooling2D((2,2))(conv1)
    #conv1 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer=kernel)(conv1)
    #conv1 = MaxPooling2D((2,2))(conv1)
    conv1 = Flatten()(conv1)
    
    #final_model = concatenate([conv1, conv2])
    
    
    final_model = Dense(256, kernel_initializer=kernel)(conv1)
    final_model = LeakyReLU(alpha=0.1)(final_model)
    final_model = Dense(128, kernel_initializer=kernel)(final_model)
    final_model = LeakyReLU(alpha=0.1)(final_model)
    final_model = Dense(32,  kernel_initializer=kernel)(final_model)
    final_model = LeakyReLU(alpha=0.1)(final_model)
    final_model = Dense(1, activation='linear')(final_model)
    
    model = Model(inputs=input1, outputs=final_model)
    
    opti = Adam(lr=0.01)
    model.compile(optimizer=opti, loss='mean_squared_error', metrics=['mse'])
    
    print(model.summary())
    return model

def get_policy_gradient_model():
    input1 = Input(shape=(5, 5, 3))
    kernel = RandomNormal(mean=0.0, stddev=0.05, seed=None)

    conv1 = Conv2D(64, activation='relu', kernel_size=2, kernel_initializer=kernel)(input1)
    conv1 = MaxPooling2D((2,2))(conv1)
    #conv1 = Conv2D(64, activation='relu', kernel_size=2, kernel_initializer=kernel)(conv1)
    conv1 = Flatten()(conv1)

    final_model = Dense(256, activation='relu', kernel_initializer=kernel)(conv1)
    final_model = Dense(128, activation='relu', kernel_initializer=kernel)(final_model)
    final_model = Dense(64,  activation='relu', kernel_initializer=kernel)(final_model)
    final_model = Dense(25, activation='softmax')(final_model)

    model = Model(inputs=input1, outputs=final_model)

    opti = Adam(lr=0.001, clipnorm=1)
    #model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.compile(optimizer=opti, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['categorical_accuracy'])

    print(model.summary())
    return model

def pre_process_data(file_path):

    X, Y = get_model_data_policy_gradient(file_path)
    
    new_df = pd.concat([X, pd.Series(Y)], axis = 1)
    new_df.columns = ['X', 'Y']
    
    unb_class = new_df.loc[new_df['Y'] == 0, :]
    unb_class = unb_class.append([unb_class])
    new_df = new_df.append([unb_class])
    new_df = shuffle(new_df)
    X2 = new_df.X
    Y2 = new_df.Y
    
    
    X_train, X_val, Y_train, Y_val = train_test_split(X2, Y2, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5)
    
    
    X_train = np.reshape(X_train.tolist(), (X_train.shape[0], 5, 5, 2))
    X_val = np.reshape(X_val.tolist(), (X_val.shape[0], 5, 5, 2))
    X_test = np.reshape(X_test.tolist(), (X_test.shape[0], 5, 5, 2))

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def pre_process_data_policy_gradient(file_path):

    X, Y = get_model_data_policy_gradient(file_path)

    new_df = pd.concat([X, pd.Series(Y)], axis = 1)
    new_df.columns = ['X', 'Y']

    #unb_class = new_df.loc[new_df['Y'] == 0, :]
    #unb_class = unb_class.append([unb_class])
    #new_df = new_df.append([unb_class])
    #new_df = shuffle(new_df)
    X2 = new_df.X
    Y2 = new_df.Y
    
    
    X_train, X_val, Y_train, Y_val = train_test_split(X2, Y2, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5)
    
    
    X_train = np.reshape(X_train.tolist(), (X_train.shape[0], 5, 5, 3))
    X_val = np.reshape(X_val.tolist(), (X_val.shape[0], 5, 5, 3))
    X_test = np.reshape(X_test.tolist(), (X_test.shape[0], 5, 5, 3))

    Y_train = np.array(Y_train.to_list())
    Y_val = np.array(Y_val.to_list())
    Y_test  = np.array(Y_test.to_list())
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def train_agent(X_train,X_val, Y_train, Y_val, epochs=50):
    model = get_model()
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=50)

    return model

def train_policy_gradient_agent(X_train,X_val, Y_train, Y_val, epochs=50, network_size='small'):
   
    if network_size == 'small':
        model = get_policy_gradient_model() 
    elif network_size == 'medium':
        import networks.medium
        model = networks.medium.get_policy_gradient_model()
    else:
        import networks.large
        model = networks.large.get_policy_gradient_model()
        
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=50)

    return model, history

def test_agent(agent, X_test, Y_test):
    predictions = agent.predict(X_test)
    print(predictions[0])
 
    import tensorflow as tf 
    cce = tf.keras.losses.CategoricalCrossentropy() 
    print("########################################################") 
    print("Test Results:")
    print("Categorical Cross Entropy: "+str(cce(predictions, Y_test).numpy()))

    #final_predictions = []
    #for x in predictions:
    #    if x < 0.5:
    #        final_predictions.append(0)
    #    else:
    #        final_predictions.append(1)
    #print(np.mean(np.array(final_predictions) == np.array(Y_test)))

    return
    
def save_model(file_path, agent, history=None):

    print("########################################################")
    print("Saving the model to: " + str(file_path))
    
    agent.save(file_path)

    if history:
        file_path = file_path + "/history.json"
        with open(file_path, "w+") as history_file: 
            json.dump(history.history, history_file, indent=4)
   
    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--agent-path', required=True)
    parser.add_argument('--policy-gradient', default=False)
    parser.add_argument('--epochs', '-n', type=int, default = 50)
    parser.add_argument('--agent-size', default='small')

    args = parser.parse_args()

    if args.policy_gradient:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = pre_process_data_policy_gradient(args.data_path)
        agent, history = train_policy_gradient_agent(X_train, X_val, Y_train, Y_val, args.epochs, args.agent_size)
    else:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = pre_process_data(args.data_path)
        agent, history = train_agent(X_train, X_val, Y_train, Y_val, args.epochs)
    
    test_agent(agent, X_test, Y_test)

    save_model(args.agent_path, agent, history)
 
