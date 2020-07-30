import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_curve, mean_squared_error
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.colors import ListedColormap


flatui = ["black", "tomato", "deepskyblue", "darkseagreen"]
#flatui = 
sns.set_palette(flatui)
new_cmap = ListedColormap(flatui)


def plot_data(file_path):
    X1_train = []
    X2_train = []
    Y_train = []

    x_list = []

    print("Loading data...")
    try:
        with open(file_path) as json_file:
            json_data = json.load(json_file)
        skip = 0

        x = 0
        for key, value in json_data.items():
            #x += len(action)
                x += 1
            #if x/1 == 1:
                print("###################### Experiment "+str(x)+" ##################")
                for index in range(len(value)):
                    fig, axes = plt.subplots(1, 3)#,figsize=(10,5))
                    fig.set_figwidth(16)
                    axes[0].imshow(np.array(value[index]['board_state']), origin='lower', cmap=new_cmap)
                    array = np.zeros((5,5))
                    for a in value[index]['action_space']:
                        axes[0].plot(a['y'], a['x'], 'bx')
                        axes[1].plot(a['y'], a['x'], 'bx')
                        array[a['x']][a['y']] = a['prob']
                    axes[1].imshow(array, cmap='Blues', origin='lower')
                    axes[0].plot(value[index]['action']['y'], value[index]['action']['x'], 'ro')
                    axes[1].plot(value[index]['action']['y'], value[index]['action']['x'], 'ro')
                    #array = np.reshape(array, (25,))
                    axes[2].hist(array.flatten(), rwidth = 0.5)
                    #axes[2].set_ylim(8, 13)
                    #axes[2].set_xlim(0, 0.5)
                    plt.show()
                    #break

        print("Done")
    except Exception as e:
        print(e)


def plot_average(data_path, iterations):

    iterations += 1

    averages = {i:[0]*10 for i in range(1, iterations)}
    stand_devs = {i:[0]*10 for i in range(1, iterations)}
    
    for i in range(1, iterations):
        file_path = data_path + '/iteration'+str(i)+'/data.json'
    
        with open(file_path) as json_file:
            json_data = json.load(json_file)
        
        x = []
        
        json_data = {int(k):v for k, v in json_data.items()}
        
        sorted_items = sorted(json_data.items())
        
        iterator = 0
        average = []
        board_states = []
        for key, value in sorted_items:
            averages[i][iterator] += len(value)
            average.append(len(value))
            
            
            if key%100 == 0:
                board_states.append(value[0]['board_state'])
                averages[i][iterator] /= 100
                stand_devs[i][iterator] = np.array(average).std()
                average = []
                iterator += 1
        
    flatui = ["black", "tomato", "deepskyblue", "darkseagreen"]
    sns.set_palette(flatui)
    new_cmap = ListedColormap(flatui)
    
    
    x = list(range(1, iterations))
    for i in range(1, 11):
        fig, axes = plt.subplots(1, 3)
        fig.set_figwidth(16)
        
        avg = []
        std = []
        for item in x:
            #print(averages[item])
            avg.append(averages[item][i-1])
            std.append(stand_devs[item][i-1])
            
        axes[1].errorbar(x, avg, yerr=std, color='orange')
        axes[1].scatter(x, avg, color='red')
        #plt.title("Plot of mean and standard deviation in each iteration")
        #plt.xlabel("Iteration")
        #plt.ylabel("Average of actions to win")
        #plt.show()
        axes[0].plot(x, avg, color='orange')
        axes[0].scatter(x, avg, color='red')
        #axes[0].title("Plot of mean in each iteration")
        #axes[0].xlabel("Iteration")
        #axes[0].ylabel("Average of actions to win")
        axes[2].imshow(np.array(board_states[i-1]).T, origin='lower', cmap=new_cmap)
        plt.show()    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--iterations', '-n', type=int, default=10)

    args = parser.parse_args()

    plot_average(args.data_path, args.iterations) 

if __name__ == "__main__":
    main()


