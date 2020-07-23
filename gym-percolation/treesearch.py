import numpy as np
import gym
from itertools import product as iterproduct

def deepcopy(old_env):

    env = gym.make("gym_percolation:Percolation-mode0-v0", grid_size=old_env.state.shape, 
            enable_render=False)        
        
    env.grid_view.grid.states = old_env.grid_view.grid.states
    env.grid_view.grid.alive = old_env.grid_view.grid.alive
    env.grid_view.grid.visited = old_env.grid_view.grid.visited
    env.grid_view.grid.groups = old_env.grid_view.grid.groups
    env.state = env.grid_view.grid.states.copy()
    return env

class GameTreeNode(object):

    def __init__(self,game_state,action,parent):
        self.children = []
        self.parent = parent
        self.moves_to_end = np.inf
        self.action = action
        self.game_state = deepcopy(game_state)
        self.full_tree = False
        if self.action is not None:
            self.apply_action(action)

    def apply_action(self,action):
        self.game_state.step(action)
        #print(len(self.game_state.action_space))
        if len(self.game_state.action_space) == 0:
            print(self.game_state.state)
            self.moves_to_end = 0
            current_parent = self.parent
            current_child = self
            while current_parent is not None:
                print(current_child.moves_to_end)
                current_parent.moves_to_end = min(current_parent.moves_to_end,current_child.moves_to_end + 1)
                current_child = current_parent
                current_parent = current_parent.parent
            
    def populate_children(self):
        for action in self.game_state.action_space:
            self.children.append(GameTreeNode(self.game_state, action, self))

    def grow_tree(self):
        queue = [self]
        while queue:#(self.full_tree and queue) or (not self.full_tree and np.isinf(self.moves_to_end)):
            front = queue.pop(0)
            front.populate_children()
            queue += front.children

    def best_child(self):
        self.children = sorted(self.children,key=lambda x: x.moves_to_end)
        return self.children[0]
            