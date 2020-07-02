import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_percolation.envs.percolation_grid import PercolationGrid

import numpy as np
import math
 
class PercolationEnv(gym.Env):  
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    
    @property
    def action_space(self):
        actions = list()

        '''
        # Very naive heuristic => select available cells
        for i in range(self.grid_view.grid.states.shape[0]):
            for j in range(self.grid_view.grid.states.shape[1]):
                if self.grid_view.grid.states[i,j] == self.grid_view.grid.STATES['Empty']:
                    actions.append({'x':i, 'y':j})
        '''

        # A little better one => pick cells that has potential to bridge other cells
        cellByClusters = dict() 
        for i in range(self.grid_view.grid.states.shape[0]):
            for j in range(self.grid_view.grid.states.shape[1]):
                if self.grid_view.grid.states[i,j] == self.grid_view.grid.STATES['Empty']:
                    # Get angle around the cell and look clusters by sorting them
                    angleDict = dict()
                    for m,n in [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1, -1), (0, -1), (1, -1)]:
                        if 0 <= i+m < self.grid_view.grid_size[0] and 0 <= j+n < self.grid_view.grid_size[1]:
                            angle = math.atan2(n, m) 
                            angleDict[angle] = self.grid_view.grid.states[i+m,j+n]

                    # Keep coordinates by the number of disjoint clusters
                    emptyNum = self.grid_view.grid.STATES['Empty']
                    stateVec = [d[1] for d in sorted(angleDict.items(), key=lambda x: x[0])]
                    nc = 1 if stateVec[0] != emptyNum else 0
                    for c in range(1,len(stateVec)):
                        if stateVec[c] != emptyNum and stateVec[c-1] == emptyNum:
                            nc += 1
                    if len(stateVec) == 8 and stateVec[0] == stateVec[-1] and stateVec[0] != emptyNum:
                        nc -= 1

                    if nc not in cellByClusters:
                        cellByClusters[nc] = list()
                    cellByClusters[nc].append({'x':i, 'y':j})

        # Return the largest ones
        if len(cellByClusters) > 0:
            #maxC = max(list(cellByClusters.keys()))
            #actions = cellByClusters[maxC]
            actions = list()
            for key in cellByClusters.keys():
                if key == 0: continue
                actions.extend(cellByClusters[key])
        else:
            actions = list()

        return actions

    @property
    def observation_space(self):
        observations = list()
        return observations

    def __init__(self, gridObject=None, enable_render=True, np_seed=None):

        self.viewer = None
        self.enable_render = enable_render

        # Simulation related variables.
        self.seed()

        self.grid_view = gridObject
        self.grid_size = self.grid_view.grid_size

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self.reset()
        self.configure()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __del__(self):
        if self.enable_render is True:
            self.grid_view.quit_game()
 
    def is_game_over(self):
        return self.grid_view.game_over

    def step(self, action):
        self.state = self.grid_view.grid.states.copy()
        self.grid_view.grid.register_move(action['x'], action['y'])

        reward = 0
        done = self.is_game_over()
        info = {}

        return self.state, reward, done, info
 
    def reset(self, observation=np.array([])):
        self.grid_view.restart()
        self.grid_view.grid.fill_affected()
        if observation.any():
            self.state = observation
            self.grid_view.grid.states = observation
        else:
            self.state = self.grid_view.grid.states # added by Omkar
        #self.state = np.zeros(2) changes made by Omkar
        self.steps_beyond_done = None
        self.done = False
        return self.state
 
    def render(self, mode="human", close=False):
        if close:
            self.grid_view.quit_game()

        return self.grid_view.update(mode)


class PercolationEnvMode0(PercolationEnv):

    def __init__(self, grid_size=(25, 25), p=0.38, zero_thres=0.05, enable_render=True, np_seed=None):
        pgrid = PercolationGrid(grid_size=grid_size, p=p, zero_thres=zero_thres, np_seed=np_seed, enable_render=enable_render)
        super(PercolationEnvMode0, self).__init__(gridObject=pgrid, enable_render=enable_render, np_seed=np_seed)

