import numpy as np 
import pickle


class ExperimentLogger(object):

    def __init__(self, filename, overwrite=False):
        self.filename = filename
        if overwrite:
            open(filename, 'w').close()

    def add_record(self, record):
        with open(self.filename, 'a') as fl:
            fl.write('{}\n'.format(json.dumps(record)))


class Agent(object):
    def __init__(self, model, exp_rate=0.1, board_size=5):
        self.model = model
        self.exp_rate = exp_rate
        self.board_size = board_size
        self.logger = None
        
    def set_logger(self,logger):
        self.logger=logger

    def encode(self,observation,action):
        board_state = np.reshape(np.array(observation), (self.board_size, self.board_size, 1))
        encoded_action = np.zeros((self.board_size, self.board_size, 1))
        encoded_action[action['x']][action['y']] = 1
        stacked_array = np.dstack((board_state, encoded_action))
        return np.reshape(stacked_array, (1, self.board_size, self.board_size, 2))
    
    def scores_to_probs(self,scores):
        probs = np.array(scores)
        probs = (probs - probs.min())/(probs.max()-probs.min())
        probs = probs/probs.sum()
        probs = np.clip(probs,eps,1-eps)
        probs = probs/probs.sum()
        return probs
        
    def select_move(self, env):
        observation=env.state
        action_space=env.action_space
        action_scores = []
        for action in action_space:
            model_input = self.encode(board_state,action)
            pred = self.model.predict(model_input)
            action["prob"] = pred[0][0]
            action_scores.append(pred[0][0])
            
        probs = self.scores_to_probs(action_scores)
        if random.uniform(0, 1) <= exp_rate:
            return_action = np.random.choice(actionSpace)
        else:
            return_action = max(zip(action_scores,action_space))[0]
        observation, reward, done, info = env.step(return_action)
        self.collector.add_record(
                    dict(board_state = observation.tolist(), 
                         action_space=action_space, 
                         action= return_action) )
        return env
    
    def serialize(self,ofname):
        self.model.save(ofname)
        #pickle.dump(self.model, open(ofname, 'wb'))

def load_agent(ifname):
    from tensorflow.keras.models import load_model
    model = load_model(ifname)
    return Agent(model,board_size = model.input_shape[1])
    
        
