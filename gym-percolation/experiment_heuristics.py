import gym
import gym_percolation
import getopt, sys

import json
import random
import numpy as np
import codecs
import pickle
import slackbot

import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class ExperimentLogger(object):

    def __init__(self, filename, overwrite=False):
        self.filename = filename
        if overwrite:
            open(filename, 'w').close()

    def addRecord(self, record):
        with open(self.filename, 'a') as fl:
            fl.write('{}\n'.format(json.dumps(record)))

def get_model_action(actionSpace, observation, model):
   
    maxi = 0
    board_state = np.reshape(np.array(observation), (5, 5, 1))
    probs = []
    for action in actionSpace:
        encoded_action = np.zeros((5, 5, 1)) 
        encoded_action[action['x']][action['y']] = 1
        stacked_array = np.dstack((board_state, encoded_action))
        #pred = model.predict([np.reshape(np.array(observation),(1, 25, 25, 1)), np.reshape(encoded_action,(1, 25, 25, 1))])[0][0]
        pred = model.predict(np.reshape(stacked_array, (1, 5, 5, 2)))
        action["prob"] = str(pred[0][0])
        probs.append(pred[0][0])
        if pred[0][0] >= maxi:
            maxi  = pred[0][0]
            return_action = action
    
    rand = random.uniform(0, 1)
    probs = np.array(probs)
    probs = probs/probs.sum()
    if rand <= 0.1:
        return_action = np.random.choice(actionSpace, p=probs)

    return return_action, actionSpace

def run_gym_environment(params):
    env = gym.make("gym_percolation:Percolation-mode0-v0", grid_size=params['grid_size'], p=params['p'])

    observation = env.reset()
    done = False
    stepCounter = 0
    json_data = []
    while not done:
        stepCounter += 1
        env.render()

        actionSpace = env.action_space
        if len(actionSpace) > 0:
            if params['random'] == True:
                action = random.sample(actionSpace, 1)[0]
            else:
                action, actionSpace = get_model_action(actionSpace, observation, params['model'])
            action_step = {}
            observation, reward, done, info = env.step(action)
            action_step["board_state"] = observation.tolist()
            action_step["action_space"] = actionSpace
            action_step["action"] = action
            json_data.append(action_step)
        else:
            done = True

    env.render()

    return json_data, plt

def calculate_par(data, n_experiments):
    secondary_steps = 0

    for key, value in data.items():
        steps = len(value)
        secondary_steps += len(value)
        for index in range(steps):
            value[index]["steps_from_win"] = steps
            steps -= 1

    avg_steps = secondary_steps/n_experiments
    print("Average Steps: "+str(avg_steps))

    return data, avg_steps

def calc_move_quality(data, avg_steps):

    for key, value in data.items():
        steps = len(value)
        for index in range(steps):
            if value[index]["steps_from_win"] > avg_steps:
                move_quality = -1
            else:
                move_quality = 1
            value[index]["move_quality"] = move_quality

    return data

if __name__ == "__main__":

    elogger = ExperimentLogger('experiments/experiment_connection-select.json')
    short_options = "hm:f:"
    long_options = ["help","random", "model"]
    full_cmd_arguments = sys.argv

    # Keep all but the first
    argument_list = full_cmd_arguments[1:]

    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        # Output error, and return with an error code
        print (str(err))
        sys.exit(2)

    if len(arguments) == 0:
        print("Error: Missing required arguments -f (or) --file, please refer to help using -h")
        exit()

    rand = True
    file_path = None 
    for opt, val in arguments:
        print(str(opt)+"  "+str(val))
        if opt in ('-h', '--help'):
            print("Arguments: ")
            print("[optional] -h (or) --help: help menu")
            print("[optional] -m (or) --model: model path from which we want to select actions")
            print("[required] -f (or) --file: file path to store the generated data")
            exit()
        if opt in ('-f', '--file'):
            file_path = val
        if opt in ('-m', '--model'):
            rand = False
            model_path = val
            model = pickle.load(open(model_path, 'rb'))

    if file_path is None:
        raise Exception("Missing -f (or) --file argument to store the generated data")

    #for p in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    avg_step = []
    for i in range(0, 1):
        p = 0.3
        # path to the model using which you want to pick actions
        #model_path = './mini_experiments/iteration11/model_exp.sav'
        #model = pickle.load(open(model_path, 'rb'))

        parameters = {'grid_size': (5,5), 'p':p, 'random': rand, 'model':model}
        nExperiment = 500
        json_data = {}
        for r in range(nExperiment):
            try:
                print('Creating gym [{}/{}]'.format(r, nExperiment))
                results, plt = run_gym_environment(parameters)
                json_data[str(r)] = results
                #elogger.addRecord({**parameters, **results})
            except Exception as e:
                print(e)

    #file_path = './mini_experiments/iteration12/data_exp_2.json'
    print("Calculating steps from win for each board state observation...")
    json_data, avg_steps = calculate_par(json_data, nExperiment)
    avg_step.append(avg_steps)
    print("Calculating move quality for each step and board state observation pair...")
    json_data = calc_move_quality(json_data, avg_steps)
    print("Writing to the json file...")
    json.dump(json_data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    #slackbot.send_message(["Done with the experiments", "i = "+str(i)])
