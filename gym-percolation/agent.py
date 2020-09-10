import numpy as np 
import json
import os
eps = 1e-6

class ExperimentLogger(object):

    def __init__(self, filename, overwrite=False):
        self.filename = filename
      
        if overwrite:
            open(filename, 'w').close()
        
        self.active_game={"metadata":{},"history":[]}
        
    def flush(self):
        if not os.path.exists(self.filename):
            json_data = {'experiments':[]}
            json_data['experiments'].append(self.active_game)
            with open(self.filename, 'w') as fl:
                fl.write(json.dumps(json_data, separators=(',', ':'), sort_keys=True, indent=4))

        else:
            with open(self.filename, 'r+') as fl:
                existing_json = json.load(fl)
                existing_json['experiments'].append(self.active_game)
                fl.seek(0)
                fl.write(json.dumps(existing_json, separators=(',', ':'), sort_keys=True, indent=4))

        self.active_game["history"]=[]
    
    def record_step(self,record):
        self.active_game["history"].append(record)
        
    def update_metadata(self,**kwargs):
        self.active_game["metadata"].update(**kwargs)

"""
   Base agent class
"""
class Agent(object):
    def __init__(self, model, exp_rate=0.1, board_size=5):
        self.model = model
        self.exp_rate = exp_rate
        self.board_size = board_size
        self.logger = None
        
    def set_logger(self,logger):
        self.logger=logger

    def record_step(self,record):
        if self.logger:
            self.logger.record_step(record)
    
    def flush_logger(self):
        if self.logger:
            self.logger.flush()
            
    def scores_to_probs(self,scores):
        probs = np.array(scores)
        probs = (probs - probs.min())/(probs.max()-probs.min())
        probs = probs/probs.sum()
        probs = np.clip(probs,eps,1-eps)
        probs = probs/probs.sum()
        return probs
        
    def serialize(self,ofname):
        self.model.save(ofname)
        #pickle.dump(self.model, open(ofname, 'wb'))


"""
   Agent implementing Q Learning 
"""
class QAgent(Agent):

    def encode(self,observation,action):
        board_state = np.reshape(np.array(observation), (self.board_size, self.board_size, 1))
        encoded_action = np.zeros((self.board_size, self.board_size, 1))
        encoded_action[action['x']][action['y']] = 1
        stacked_array = np.dstack((board_state, encoded_action))
        return np.reshape(stacked_array, (1, self.board_size, self.board_size, 2))

    def select_move(self, env):
        action_space=env.action_space
        action_scores = []
        for action in action_space:
            model_input = self.encode(env.state,action)
            pred = self.model.predict(model_input)
            action["prob"] = float(pred[0][0])
            action_scores.append(pred[0][0])

        probs = self.scores_to_probs(action_scores)
        if np.random.rand() <= self.exp_rate:
            return_action = np.random.choice(actionSpace)
        else:
            return_action = max(zip(action_scores, action_space))[1]

        observation, reward, done, info = env.step(return_action)
        step_record = dict(board_state = observation.tolist(), action_space=action_space,action= return_action)
        self.record_step(step_record)
        return env, action_step

"""
   Agent implementing Policy Gradient Learning 
"""
class PolicyAgent(Agent):

    def encode(self, observation, red_equals_black = False):
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

    def select_move(self, env):
        action_space=env.action_space
        action_scores = []

        board_state = np.reshape(np.array(env.state), (5, 5, 1))
        board_state = self.encode(board_state, red_equals_black=True)
        pred = self.model.predict(np.reshape(board_state, (1, 5, 5, 3)))
        policy = np.reshape(pred[0], (5, 5))

        for action in action_space:
            action_scores.append(policy[action['x']][action['y']])
            action['prob'] = float(policy[action['x']][action['y']])
            
        probs = self.scores_to_probs(action_scores)
        if np.random.rand() <= self.exp_rate:
            return_action = np.random.choice(actionSpace)
        else:
            return_action = max(zip(action_scores, action_space))[1]

        observation, reward, done, info = env.step(return_action)
        step_record = dict(board_state = observation.tolist(), action_space=action_space,action= return_action)
        self.record_step(step_record)
        return env

def load_agent(ifname, policy_gradient=False):
    from tensorflow.keras.models import load_model
    model = load_model(ifname)

    if policy_gradient:
        agent = PolicyAgent(model, board_size = model.input_shape[1])
    else:
        agent = QAgent(model, board_size = model.input_shape[1])

    return agent
    
 
