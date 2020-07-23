import time
import pygame
import random
import numpy as np
import os, sys
import uuid

import logging

# create logger
logger = logging.getLogger('game_logger')
logger.setLevel(logging.WARNING)
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class PercolationGrid:

    def __init__(self, game_name="Percolation Grid", seed_value=None, p=0.38, zero_thres=0.05,
                 grid_size=(30, 30), screen_size=(600, 600), enable_render=True, np_seed=None):

        self.__game_over = False
        self.__enable_render = enable_render

        self.__grid = GridMode0(grid_size, p=p, zero_thres=zero_thres, seed=seed_value, np_seed=np_seed) # Update P value

        self.grid_size = self.__grid.grid_size
        if self.__enable_render is True:
            # PyGame configurations
            pygame.init()
            pygame.display.set_caption(game_name)
            self.clock = pygame.time.Clock()

            # to show the right and bottom border
            self.screen = pygame.display.set_mode(screen_size)
            self.__screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

        if self.__enable_render is True:
            # Create a background
            self.background = pygame.Surface(self.screen.get_size()).convert()
            self.background.fill((255, 255, 255))

            # Create a layer for the maze
            self.grid_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
            self.grid_layer.fill((0, 0, 0, 0,))

            self.__draw_grid()


    def restart(self):
        self.__grid.restart()

    def update(self, mode="human"):
        try:
            img_output = self.__view_update(mode)
            self.__controller_update()

            if self.__grid.is_complete():
                self.__game_over = True

        except Exception as e:
            self.__game_over = True
            self.quit_game()
            raise e
        else:
            return img_output

    def quit_game(self):
        try:
            self.__game_over = True
            if self.__enable_render is True:
                pygame.display.quit()
            pygame.quit()
            #sys.exit()
        except Exception:
            pass

    def __controller_update(self):
        if not self.__game_over:
            pos = pygame.mouse.get_pos()
            pressed1, pressed2, pressed3 = pygame.mouse.get_pressed()
    
            if pressed1:
                idx_x = int(pos[0] / self.CELL_W)
                idx_y = int(pos[1] / self.CELL_H)

                self.__grid.register_move(idx_x, idx_y)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()

    def __view_update(self, mode="human"):
        if not self.__game_over:
            # update visual components
            self.__draw_grid()

            # update the screen
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.grid_layer,(0, 0))

            if mode == "human":
                pygame.display.flip()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))


    def __draw_grid(self):
        
        if self.__enable_render is False:
            return
        
        color_dict = {
            0: (0, 0, 0, 255),  # Black
            1: (255, 80, 0, 255), # Orange
            2:(20, 165, 195, 255), # Blue
            3:(160, 185, 115, 255) # Green
        } 

        #pygame.draw.rect(self.grid_layer, line_colour, (0, 0, self.CELL_W, self.CELL_H), 0)
        nMargin = 0.90
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                pygame.draw.rect(self.grid_layer, color_dict[self.__grid.states[i,j]],
                    (self.CELL_W*i, self.CELL_H*j, self.CELL_W*nMargin, self.CELL_H*nMargin), 0)



    @property
    def grid(self):
        return self.__grid

    @property
    def game_over(self):
        return self.__game_over

    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)

    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.grid.GRID_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.grid.GRID_H)



################################
#                              #
#         GRID OBJECTS         #
#                              #
################################

class Grid(object):

    STATES = {
        'Initially-Attacked': 0,
        'Attacked': 1,
        'Affected': 2,
        'Empty': 3,
    }

    MODES = {
        '0': 0,
        '1': 1,
        '2': 2,
    }

    def __init__(self, grid_size=(50,50), p=0.38, zero_thres=0.05, mode='0', seed=None, np_seed=None):
        self.grid_size = grid_size
        self.probability = p
        self.zero_threshold = zero_thres
        self.game_mode = mode

        if seed == None:
            self.seed = str(uuid.uuid1())
        else:
            self.seed = seed

        self.np_seed = None
        
        self.randomizer = random
        self.randomizer.seed(self.seed)

        self.states = np.ones(self.grid_size) * self.STATES['Empty']
        self.alive = np.ones(self.grid_size)
        self.visited = np.zeros(self.grid_size)
        self.groups = list()

        self.make_randomized_alive(np_seed)

    def restart(self):
        self.states = np.ones(self.grid_size) * self.STATES['Empty']
        self.alive = np.ones(self.grid_size)
        self.visited = np.zeros(self.grid_size)
        self.groups = list()
        self.make_randomized_alive(np_seed = self.np_seed)

    def make_all_alive(self):
        self.alive = np.ones(self.grid_size)
        self.states = np.ones(self.grid_size) * self.STATES['Empty']

    def make_randomized_alive(self, p=None, np_seed=None):
        if p == None:
            p = self.probability

        if np_seed:
            np.random.seed(np_seed)

        size =  self.grid_size[0]*self.grid_size[1]
        zeros = np.zeros(size)
        proportion = size - int(p*size) 
        zeros[:proportion] = 1
        np.random.shuffle(zeros)
        zeros = np.reshape(zeros, (self.grid_size[0], self.grid_size[1]))
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                #self.alive[i,j] = 0 if self.randomizer.random() < p else 1
                #self.states[i,j] = self.STATES['Empty'] if self.alive[i,j] else self.STATES['Initially-Attacked']
                self.states[i,j] = self.STATES['Empty'] if zeros[i,j] else self.STATES['Initially-Attacked']
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

    def __init__(self, grid_size=(50,50), p=0.38, zero_thres=0.05, mode='0', seed=None, np_seed=None):
        self.grid_size = grid_size
        self.probability = p
        self.zero_threshold = zero_thres
        self.game_mode = mode
        self.maximum_blue_size = 0

        if seed == None:
            self.seed = str(uuid.uuid1())
        else:
            self.seed = seed

        self.np_seed = np_seed
        
        self.randomizer = random
        self.randomizer.seed(self.seed)

        self.states = np.ones(self.grid_size) * self.STATES['Empty']
        self.alive = np.ones(self.grid_size)
        self.visited = np.zeros(self.grid_size)
        self.groups = list()

        self.make_randomized_alive(np_seed = np_seed)
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

        if self.MODES[self.game_mode] == 0:
            if len(components) > 1:
                self.maximum_blue_size = max(self.maximum_blue_size,len(components[1]))
            switch_to_state = self.STATES['Empty'] if len(components[0]) > self.maximum_blue_size else self.STATES['Affected']
            for x, y in components[0]:
                newStates[x,y] = switch_to_state


        return newStates, components

    def fill_affected(self):
        newStates, newComponents = self.get_gcc_membership()
                    
        #print([len(c) for c in newcomponents])
        cutoffs = list()

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.states[i,j] == self.STATES['Empty'] and newStates[i,j] != self.STATES['Empty']:
                    self.states[i,j] = newStates[i,j]
                    cutoffs.append((i,j))
        
        return  cutoffs

    def register_move(self, x, y):
        if self.states[x,y] != self.STATES['Empty']:
            return []

        logger.info('Attack on site ({}, {})'.format(x,y))
        self.states[x,y] = self.STATES['Attacked']
        self.alive[x,y] = 0
        self.visited[x,y] = 1

        if not self.is_complete():
            cutoffs = self.fill_affected()
        else:
            logger.info('Completed!')
    
        return cutoffs



if __name__ == "__main__":

    grid = PercolationGrid(screen_size= (750, 750), grid_size=(15,15))
    while True:
        grid.update()
    input("Enter any key to quit.")
