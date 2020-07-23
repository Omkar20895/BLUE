
import gym
import argparse
from agent import Agent,ExperimentLogger,load_agent

from itertools import product as iterproduct
from copy import deepcopy


env = gym.make("gym_percolation:Percolation-mode0-v0", grid_size=(args.board_size,args.board_size), p=args.p, np_seed=seed, 
            enable_render=False)        
observation = env.reset()
env.seed()
agent.logger.update_metadata(np_seed=seed, game_number=game_no)
while len(env.action_space)>0:
    env = agent.select_move(env)
agent.flush_logger()


class GameTreeNode(object):

    def __init__(self,game_state,action,parent):
        self.children = []
        self.parent = parent
        self.moves_to_end = np.inf
        self.game_state = deepcopy(game_state)
        self.apply_action(action)

    def apply_action(self,action):
        self.game_state.step(action)
        if len(self.game_state.action_space) == 0:
            self.moves_to_end = 0
        current_parent = self.parent
        current_child = self
        while current_parent is not None:
            current_parent.moves_to_end = current_child.moves_to_end + 1
            current_child = current_parent
            current_parent = current_parent.parent
            


    def populate_children(self):
        for action in self.game_state.action_space:
            self.children.append(GameTreeNode(self.game_state, self.action, self))

    def grow_tree(self):
        queue = [self]
        while np.isinf(self.moves_to_end):
            front = queue.pop()
            front.populate_children()
            queue += front.children

    def best_child(self):
        self.children = sorted(self.children,key=lambda x: x.moves_to_end)
        return self.children[0]
            