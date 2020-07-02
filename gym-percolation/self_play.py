import gym
import gym_percolation
import getopt, sys
import argparse
import multiprocessing

import json
import random
import numpy as np
import codecs
import pickle
#import slackbot

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

def get_model_action(actionSpace, observation, model, exp_rate=0.1):
   
    maxi = -10000
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
    probs = (probs - probs.min())/(probs.max()-probs.min())
    probs = probs/probs.sum()
    if rand <= exp_rate:
        return_action = np.random.choice(actionSpace, p=probs)

    return return_action, actionSpace

def run_gym_environment(params):
    env = gym.make("gym_percolation:Percolation-mode0-v0", grid_size=params['grid_size'], p=params['p'], np_seed=params['np_seed'], 
                    enable_render=params["enable_render"])

    new_observation = np.array([[0, 3, 3, 0, 3],
                       [2, 0, 3, 3, 0],
                       [0, 3, 3, 3, 3],
                       [3, 3, 3, 0, 3],
                       [3, 0, 3, 3, 0]]).T

    #observation = env.reset(observation=new_observation)
    observation = env.reset()
    done = False
    stepCounter = 0
    json_data = []
    while not done:
        stepCounter += 1
        if params["enable_render"]:
            env.render()

        actionSpace = env.action_space
        if len(actionSpace) > 0:
            if params['random'] == True:
                action = random.sample(actionSpace, 1)[0]
            else:
                action, actionSpace = get_model_action(actionSpace, observation, params['model'], params["exp_rate"])
            action_step = {}
            observation, reward, done, info = env.step(action)
            action_step["board_state"] = observation.tolist()
            action_step["action_space"] = actionSpace
            action_step["action"] = action
            action_step["seed"] = params["np_seed"]
            json_data.append(action_step)
        else:
            done = True

    if params["enable_render"]:
        env.render()
    #else:
    #    params["queue"].put(json_data)

    return json_data

def generate_experiments(num_games, params):
    workers = []

    for i in range(num_games):
        workers.append(multiprocessing.Process(
            target=run_gym_environment,
            args=(params)))

    for worker in workers:
            worker.start() 

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
        move_quality = avg_steps - steps
        for index in range(steps):
            #move_quality = avg_steps - value[index]["steps_from_win"]
            #if value[index]["steps_from_win"] > avg_steps:
            #    move_quality = -1 
            #else:
            #    move_quality = 1
            value[index]["move_quality"] = move_quality

    return data

def generate_experiments(parameters, args):

    nExperiment = args.num_games
    json_data = {}
    exp_num = parameters["exp_num"]
    for r in range(args.num_games):
        try:
            print('Creating gym [{}/{}]'.format(r, nExperiment))
            results = run_gym_environment(parameters)
            json_data[str(exp_num)] = results
            exp_num += 1
            #elogger.addRecord({**parameters, **results})
        except Exception as e:
            print(e)

    return json_data, exp_num


def generate_experiments_parallel(parameters, args):
    workers = []
    queue = multiprocessing.Queue()
    parameters['queue'] = queue
    nExperiment = args.num_games
    json_data = {}

    for i in range(args.num_games):
        print('Creating gym [{}/{}]'.format(i, nExperiment))
        worker = multiprocessing.Process(
            target=run_gym_environment,
            args=([parameters]))
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()

    try:
        n_exp = parameters["exp_num"]
        while not queue.empty():
           result = queue.get()
           json_data[n_exp] = result
           n_exp += 1
    except Exception as error:
        print(error)
        raise Exception
    finally:
        queue.close()
        queue.join_thread()

    return json_data, n_exp 


def main():

    elogger = ExperimentLogger('experiments/experiment_connection-select.json')

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=False, default=None)
    parser.add_argument('--num-games', '-n', type=int, default=100)
    parser.add_argument('--file-out', required=True)
    parser.add_argument('--explore-rate', type=float, default=0.1)
    parser.add_argument('--run-parallel', type=bool, default=False)

    args = parser.parse_args()

    model_path = args.agent
    model = None
    if model_path:
        model = pickle.load(open(model_path, 'rb'))
    rand = True if model is None else False
    file_path = args.file_out

    import time
    start_time = time.time()
    #for p in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    avg_step = []
 
    for i in range(0, 1):
        p = 0.3

        parameters = {'grid_size': (5,5), 'p':p, 'random':rand, 'model':model, "exp_rate":args.explore_rate}
        nExperiment = args.num_games
        json_data = {}

        exp_num = 1
        for i in range(1, 11):
            temp_json = {}
            parameters["np_seed"] = i
            parameters["exp_num"]  = exp_num
            if args.run_parallel:
                parameters["enable_render"] = False
                temp_json, exp_num = generate_experiments_parallel(parameters, args)
            else:
                parameters["enable_render"] = False
                temp_json, exp_num = generate_experiments(parameters, args)

            temp_json, avg_steps = calculate_par(temp_json, nExperiment)
            temp_json = calc_move_quality(temp_json, avg_steps)
            json_data.update(temp_json)

    #file_path = './mini_experiments/iteration12/data_exp_2.json'
    print("Calculating steps from win for each board state observation...")
    #json_data, avg_steps = calculate_par(json_data, nExperiment)
    #avg_step.append(avg_steps)
    print("Calculating move quality for each step and board state observation pair...")
    #json_data = calc_move_quality(json_data, avg_steps)
    print("Writing to the json file...")
    json.dump(json_data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    #slackbot.send_message(["Done with the experiments", "i = "+str(i)])
    print(time.time()-start_time)

if __name__ == "__main__":
    main()
