import numpy as np 

class Agent(object):
    def __init__(self, model, exp_rate=0.1, board_size=5):
        self.model = model
        self.exp_rate = exp_rate
        self.board_size = board_size

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
        if rand <= exp_rate and not True in np.isnan(probs):
            return_action = np.random.choice(actionSpace, p=probs)

        return return_action, actionSpace
    
