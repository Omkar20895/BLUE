import gym
import gym_percolation

import json
import random
import numpy as np
import pandas as pd
import codecs
import pickle
import slackbot

import matplotlib.pyplot as plt
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.colors import ListedColormap

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, BatchNormalization
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
                array_board = np.reshape(array_board, (25, 25, 1))
                #X1_train.append(array)
                Y_train.append(value[index]['move_quality'])
                array_action = np.zeros((25, 25, 1))
                array_action[value[index]['action']['x']][value[index]['action']['y']] = 1
                #X2_train.append(array)
                stacked_array = np.dstack((array_board, array_action))
                X1_train.append(stacked_array)
        
        print("Done")
    except Exception as e:
        print(e)
        
    X = pd.DataFrame({'X':X1_train, 'Y':Y_train})
    X = shuffle(X)
    #print(type(X["X"][0]))
    #print(X["X"][0].shape)
    print("Done")
    X, Y_train = X["X"], X["Y"]
    Y_train = (Y_train>0).astype(float)
    return X, list(Y_train)

def get_roc_curves():
    #figure, ax = plt.figure()
    for num in range(1, 5):
        file_path = './experiments/moveData/iteration'+str(num)+'/experiment_0.3_new_heuristic_part1.json'
        X, Y = get_model_data_stacked(file_path)
        file_path = './experiments/moveData/iteration'+str(num)+'/experiment_0.3_new_heuristic_part2.json'
        X2, Y2 = get_model_data_stacked(file_path)
        
        X = pd.concat([X, X2], axis=0)
        Y.extend(Y2)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5)
        
        X_train = np.reshape(X_train.tolist(), (X_train.shape[0], 25, 25, 2))
        X_val = np.reshape(X_val.tolist(), (X_val.shape[0], 25, 25, 2))
        X_test = np.reshape(X_test.tolist(), (X_test.shape[0], 25, 25, 2))

        file_path = "./models/iteration"+str(num)+"/model_0.3_stacked_heuristic.sav"
        print(file_path)
        model = pickle.load(open(file_path, 'rb'))

        predictions = model.predict(X_test)
        predictions = [1 if x >= 0.5 else 0 for x in predictions]
        fpr, tpr, _ = roc_curve(Y_test, predictions)

        plt.plot(fpr, tpr, label="iteration "+str(num))

    plt.legend()
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC curve for each iteration")
    return plt

plot = get_roc_curves()
plot.show()       
