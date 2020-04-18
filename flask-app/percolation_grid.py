import time
import random
import numpy as np
import os, sys
import uuid

import logging

# create logger
logger = logging.getLogger('game_logger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


################################
#                              #
#         GRID OBJECTS         #
#                              #
################################

class Grid(object):

    STATES = {
        'Initially-Attacked': 2,
        'Attacked': 3,
        'Affected': 1,
        'Empty': 0,
    }

    def __init__(self, grid_size=(50,50), p=0.38, zero_thres=0.05, mode='0', seed=None):
        self.grid_size = grid_size
        self.probability = p
        self.zero_threshold = zero_thres
        self.game_mode = mode

        if seed == None:
            self.seed = str(uuid.uuid1())
        else:
            self.seed = seed
        
        self.randomizer = random
        self.randomizer.seed(self.seed)

        self.states = np.ones(self.grid_size) * self.STATES['Empty']
        self.alive = np.ones(self.grid_size)
        self.visited = np.zeros(self.grid_size)
        self.groups = list()

        self.make_randomized_alive()

    def make_all_alive(self):
        self.alive = np.ones(self.grid_size)
        self.states = np.ones(self.grid_size) * self.STATES['Empty']

    def make_randomized_alive(self, p=None):
        if p == None:
            p = self.probability

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.alive[i,j] = 0 if self.randomizer.random() < p else 1
                self.states[i,j] = self.STATES['Empty'] if self.alive[i,j] else self.STATES['Initially-Attacked']
        self.get_gcc_membership()
        logger.info('Grid with size {} created'.format(self.grid_size))

    def get_gcc_membership(self):
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.visited[i,j] = 1 if self.alive[i,j] == 0 else 0

        grps = list()
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.visited[i,j] == 0:
                    grps.append(self.run_bfs(i,j))

        grps.sort(key=len, reverse=True)
        self.groups = grps

    def run_bfs(self, i,j):
        currentGroup = list()

        queue = [(i,j)]
        self.visited[i,j] = 1
        while len(queue) > 0:
            xc, yc = queue.pop()
            currentGroup.append((xc, yc))

            for xn, yn in self.get_adjacent_cells(xc, yc):
                if self.visited[xn,yn] == 0:
                    queue.append((xn, yn))
                    self.visited[xn,yn] = 1
        return currentGroup


    def get_adjacent_cells(self, x, y):
        adjList = list()
        for i,j in [(-1,0), (1,0), (0,1), (0,-1)]:
            if 0 <= x+i < self.grid_size[0] and 0 <= y+j < self.grid_size[1]:
                adjList.append((x+i, y+j)) 
        return adjList

    # Function called as generate_status_vector_from_groups in JS version
    def update_grid_status(self):
        return 'Implement logic of this grid here'

    def is_complete(self):
        return not np.any(self.states == self.STATES['Empty'])

    @property
    def GRID_W(self):
        return int(self.grid_size[0])

    @property
    def GRID_H(self):
        return int(self.grid_size[1])




class GridMode0(Grid):

    def __init__(self, grid_size=(50,50), p=0.38, zero_thres=0.05, mode='0', seed=None):
        self.grid_size = grid_size
        self.probability = p
        self.zero_threshold = zero_thres
        self.game_mode = mode

        if seed == None:
            self.seed = str(uuid.uuid1())
        else:
            self.seed = seed
        
        self.randomizer = random
        self.randomizer.seed(self.seed)

        self.states = np.ones(self.grid_size) * self.STATES['Empty']
        self.alive = np.ones(self.grid_size)
        self.visited = np.zeros(self.grid_size)
        self.groups = list()

        self.make_randomized_alive()
        self.states, self.groups = self.get_gcc_membership()


    def run_bfs(self, x, y):
        currentGroup = list()
        queue = [(x,y)]

        availableCells = np.array(self.visited)
        availableCells[x,y] = 1
        while(len(queue)) > 0:
            xc, yc = queue.pop()
            currentGroup.append((xc, yc))

            for xn, yn in self.get_adjacent_cells(xc, yc):
                if availableCells[xn,yn] == 0:
                    queue.append((xn, yn))
                    availableCells[xn,yn] = 1
        return currentGroup

    def get_components(self):
        self.visited = np.zeros(self.grid_size)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.states[i,j] == self.STATES['Initially-Attacked'] or self.states[i,j] == self.STATES['Attacked']:
                    self.visited[i,j] = 1

        groups = list()
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if (self.states[i,j] == self.STATES['Empty'] or self.states[i,j] == self.STATES['Affected']) and self.visited[i,j] == 0:
                    groups.append(self.run_bfs(i,j))
                    for c in groups[-1]:
                        self.visited[c[0],c[1]] = 1

        # Sort components by size
        groups.sort(key=len, reverse=True)

        return groups

    def get_gcc_membership(self): 

        newStates = np.array(self.states)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if newStates[i,j] == self.STATES['Empty']:
                    newStates[i,j] = self.STATES['Affected']

        components = self.get_components()

        for x, y in components[0]:
            newStates[x,y] = self.STATES['Empty']

        return newStates, components

    def register_move(self, x, y):
        if self.states[x,y] != self.STATES['Empty']:
            return []

        logger.info('Attack on site ({}, {})'.format(x,y))
        self.states[x,y] = self.STATES['Attacked']
        self.alive[x,y] = 0
        self.visited[x,y] = 1

        cutoffs = list()
        if not self.is_complete():
            newStates, newcomponents = self.get_gcc_membership()
            print([len(c) for c in newcomponents])

            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    if self.states[i,j] == self.STATES['Empty'] and newStates[i,j] != self.STATES['Empty']:
                        self.states[i,j] = newStates[i,j]
                        cutoffs.append((i,j))
        
        if self.is_complete():        
            logger.info('Completed!')
    
        return cutoffs



if __name__ == "__main__":

    grid = PercolationGrid(screen_size= (750, 750), grid_size=(50,50))
    while True:
        grid.update()
    input("Enter any key to quit.")