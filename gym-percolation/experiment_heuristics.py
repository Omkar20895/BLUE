import gym
import gym_percolation

import json
import random
import numpy as np
import codecs
import pickle
import slackbot

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
    board_state = np.reshape(np.array(observation), (25, 25, 1))
    for action in actionSpace:
        encoded_action = np.zeros((25, 25, 1)) 
        encoded_action[action['x']][action['y']] = 1
        stacked_array = np.dstack((board_state, encoded_action))
        #pred = model.predict([np.reshape(np.array(observation),(1, 25, 25, 1)), np.reshape(encoded_action,(1, 25, 25, 1))])[0][0]
        pred = model.predict(np.reshape(stacked_array, (1, 25, 25, 2)))
        action["prob"] = str(pred[0][0])
        if pred[0][0] >= maxi:
            maxi  = pred[0][0]
            return_action = action
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
    return json_data

def calc_steps(data, n_experiments):
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

    #for p in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    #for p in [0.3]:
    avg_step = []
    for i in range(0,4):
        p = 0.3
        # path to the model using which you want to pick actions
        model_path = '/Users/omkarreddy/Desktop/RAWork/ProjectBlue/battle-perc-master/gym-percolation/models/iteration4/model_0.3_stacked_heuristic.sav'
        model = pickle.load(open(model_path, 'rb'))

        parameters = {'grid_size': (25,25), 'p':p, 'random': True, 'model':model}
        nExperiment = 100
        json_data = {}
        for r in range(nExperiment):
            try:
                print('Creating gym [{}/{}]'.format(r, nExperiment))
                results = run_gym_environment(parameters)
                json_data[str(r)] = results
                #elogger.addRecord({**parameters, **results})
            except Exception as e:
                print(e)
        #print(json_data.keys())
        #print(json.dumps(json_data, indent=4, sort_keys=True))
       
        #file_path = './experiments/moveData/iteration1/'+str(p)+'.json' 
        file_path = './experiments/moveData/iteration4/experiment_'+str(p)+'_new_heuristic_part2.json'
        print("Calculating steps from win for each board state observation...")
        json_data, avg_steps = calc_steps(json_data, nExperiment)
        avg_step.append(avg_steps)
        print("Calculating move quality for each step and board state observation pair...")
        json_data = calc_move_quality(json_data, avg_steps)

        print("Writing to the json file...")
        #json.dump(json_data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        slackbot.send_message(["Done with the experiments", "i = "+str(i)])
    print(avg_step)
