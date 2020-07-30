import numpy as np
import gym
from itertools import product as iterproduct
import heapq

class DummyEnv(object):
    def __init__(self,a_s):
        self.action_space=a_s
    def __sub__(self,action):
        return DummyEnv(set(self.action_space) -set([action]))
    
def deepcopy(old_env):
    if isinstance(old_env,DummyEnv):
        return DummyEnv(old_env.action_space)

    env = gym.make("gym_percolation:Percolation-mode0-v0", grid_size=old_env.state.shape, 
            enable_render=False)        
    env.load(old_env.state.copy())
    return env
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
        self.depth = 0 if self.parent is None else self.parent.depth+1
        if self.action is not None:
            self.apply_action(self.action)

    def apply_action(self,action):
        self.game_state.step(action)
        #print(len(self.game_state.action_space))
        if len(self.game_state.action_space) == 0:
            #print(self.game_state.state)
            self.moves_to_end = 0
            current_parent = self.parent
            current_child = self
            while current_parent is not None:
                #print(f"Child {current_child.depth} - {current_child.moves_to_end}")
                #print(f"Parent {current_parent.depth} - {current_parent.moves_to_end}")
                current_parent.moves_to_end = min(current_parent.moves_to_end,current_child.moves_to_end + 1)
                current_child = current_parent
                current_parent = current_parent.parent
            
    def populate_children(self):
        for action in self.game_state.action_space:
            self.children.append(GameTreeNode(self.game_state, action, self))
        #print(f"At depth {self.depth}")

    def grow_tree(self):
        import uuid
        queue = []
        heapq.heapify(queue)
        heapq.heappush(queue, (self.depth,uuid.uuid1().int,self))
        depth=self.depth
        lastdepth=depth
        while True:
            if not queue:
                break
            if not self.full_tree:
                if not np.isinf(self.moves_to_end) and depth > self.moves_to_end:
                    break
            
            depth,_,front = heapq.heappop(queue)
            #print(len(queue))
            if depth>lastdepth:
                print(f"exploring depth {depth}")
                lastdepth=depth
            front.populate_children()
            for c in front.children:
                heapq.heappush(queue,(c.depth,uuid.uuid1().int,c))

    def split_children_good_bad(self):
        if not self.children:
            return [],[]
        import bisect
        self.children = sorted(self.children,key=lambda x: x.moves_to_end)
        split_idx = bisect.bisect_right([i.moves_to_end for i in self.children], self.children[0].moves_to_end)
        return self.children[:split_idx], self.children[split_idx:]

    def best_child(self):
        self.children = sorted(self.children,key=lambda x: x.moves_to_end)
        return self.children[0] if self.children else None
    
    def generate_training_data(self):
        good,bad=self.split_children_good_bad()
        if good or bad:
            Y_good = [c.action for c in good]
            Y_bad = [c.action for c in bad]
            return self.game_state.state.tolist(),Y_good,Y_bad
        else:
            return [],[],[]
