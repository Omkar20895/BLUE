# Project BLUE

## Introduction:

One of the major challenges network science faces is making the networks more resilient and as we are progressing into the future of more connectivity than ever, there is a high demand for resilient and robust networks. In the study of network resilience, various toy models are used to represent the network’s response to perturbation. We propose to utilize these models, to develop a network resilience game "BLUE" which can then be learned by a reinforcement learning agent. The game is inspired from AlphaGo, a computer program with an AI agent playing the game Go.


## Experimental Setup:

All the experiments were performed on a 25x25 board. The green cells represent available cells to attack, blue cells represent affected cells, orange cells represent attacked cells and black cells represent the pre-occupied cells. The goal in the game is to find minimum number of cells which when attacked turn the board blue. We also use different percolation probabilities to fill the board with black cells during board  initialization. Approximately p\*N cells are pre-occupied, i.e marked as black. We will be dealing with percolation probabilities in the range of [0.5, 5] and the probability of 0.3 is of particular interest as it is closely related to real world networks.

At each step in the game we track the board state, the action space that consists of all the possible actions available in that board state and also the action that resulted in the current board state. We generate the action space using spatial heuristics that consider the immediate neighboring cells for each cell to determine if attacking that cell is a legitimate move.


Table 1: Example game play on a 15x15 grid. Player aims to select cells (orange) at each step to span blocked regions (black) and maximize occupied (blue) connected regions.

STEP 1 | STEP 2 | STEP 3 | STEP 4
------ | ------ | ------ | ------
<img width="350" height="250" src="https://i.imgur.com/qElnLKi.png"> | <img width="350" height="250" src="https://i.imgur.com/SsnGeOe.png"> | <img width="350" height="250" src="https://i.imgur.com/UywgEpp.png"> | <img width="350" height="250" src="https://i.imgur.com/6k0VxEM.png">


We generate action space for a given board state using two types of heuristics described as follows:

**one-or-more neighbors:**
For each green cell in a given board state we consider the 8 neighboring cells and include the green cell with one or more neighbors colored blue, black or orange into the action space.

**maximum neighbors:**
For each green cell in a given board state we consider the 8 neighboring cells and include the green cell with maximum number of neighbors colored blue, black or orange into the action space.

## Report of Execution:

### Methodology:

After intializing a board game state, for each pair of board state and action space we select a random action from the action space, play that action on the board state and repeat this until we reach end of the game. This is considered to be one game experiment. We generate multiple game experiments using this process and this would serve as our dataset.


We pre-process the dataset before using it to train the neural network by calculating the average number of steps taken to win among all the game experiments, referred to as **par** of the dataset. We assign all the actions in a game experiment with a -1 if the number of steps-to-win in that game was more than the par and 1 otherwise. This is our mechanism of penalizing and rewarding game experiments that are above(greater than) and below(lesser than) par respectively. 

Our main goal is to:

- predict the quality of a action, on a scale of 0 to 1, given a board state and action pair
- reduce the number of steps taken by the agent to win

We try to do this by iteratively training a convolutional neural network architecture. The CNN takes the observation and action pair as inputs and predicts a probability of taking that action. We generate game experiments using the CNN to select the best action from an action space for a given action space and board state rather than selecting random action from the action space as done initially while creating the random action dataset. The game experiments generated in this way act as a dataset for the next iteration of training a new CNN. We intend to run multiple iterations of training a CNN and generating new dataset as shown in Figure 1. ***We expect the average number of steps-to-win in the game experiments to decrease with each iteration.***

<p align="center">
  <img width="350" height="550" src="https://i.imgur.com/iVz3RM1.png"><br>
  Figure 2: percolation probability trend
</p>

We explore two types of architectures as follows:

1. Convoluting the board state separately and introducing convoluted board state and action vector as input to a feed forward neural network
2. Stacking the board state and the action vector on each other and then using this combined vector as an input to the convolutional neural network

Before using the action as an input to the neural network we one hot encode it into a 25x25 vector. 

### Results:

We explore how the average number of steps-to-win in game experiments with random action selection vary for each percolation probability. From Figure 2 we can see that the average number of steps decrease with increasing probability. This is due to increasing number of pre-occupied(black) cells with increasing probability and hence less number of steps-to-win.

```python
   from gym_percolation.utils import calc_avg_steps
   
   p_value = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
   p_avg_steps = calc_avg_steps(p_list)

   plt.scatter(p_value, p_avg_steps)
```

<p align="center">
  <img width="350" height="250" src="https://i.imgur.com/4xZ3Qxr.png"><br>
  Figure 2: percolation probability trend
</p>

Using maximum neighbors heuristic to generate action space the average number of steps-to-win decreased for each iteration rather than increasing, and we believe that using this heuristic results in a more constrained action space for each board state leaving very less room for exploration on the board state. This limits the learning ability of the neural network and results in an increase in the average steps-to-win. Also we did not see much difference in performance between the two CNN architectures, hence we stick with the stacked input CNN architecture.  

Hence we developed one-or-more neighbors heuristic which results in a relaxed action space for a give board state. After training the CNN architecture in an iteration we generate a game experiment by initializing a board state, generating an action space for this board state and selecting the action with the highest probability. We get the probabilities using the CNN trained on the previous iteration's data. We repeat the process till the end of the game experiment. The CNN architecture trained at each iteration in this project had **accuracy** around **96%** and the ROC curves are as shown in Figure 3.

```python
   from gym_percolation.utils import get_roc_curves

   plot = get_roc_curves()
   plot.show()
```

<p align="center">
  <img width="350" height="250" src="https://i.imgur.com/Sl3LoJS.png"><br>
  Figure 3: Plot of average steps-to-win over iterations
</p>


Using the one-or-more neighbor heuristic we generate 5 samples of 100 game experiments each for every iteration and plot the average steps-to-win amongst these samples. Since this is a more relaxed heuristic we expect the average steps to decrease with each iteration.

<p align="center">
  <img width="350" height="250" src="https://i.imgur.com/fqKzxEj.png"><br>
  Figure 4: Plot of average steps-to-win over iterations
</p>

From Figure 4, we can see that the average number of moves decrease after each iteration but start increasing after 4th iteration. We also see a really peculiar scenario where the agent(CNN) favours selecting the move that is just adjacent to a bunch of attacked cells rather than attacking the bridging cells that form two clusters. We also see that there is a rapid change in the probability distributions after taking an inefficient step on the board as show in Figure 5. The intensity of the blues represents the probability with which that action is recommended by the CNN and the red cell represents the action chosen for the board state.

<p align="center">
  <img width="550" height="250" src="https://i.imgur.com/V1O39iz.png"><br>
  <img width="550" height="250" src="https://i.imgur.com/OQt2hpz.png"><br>
  Figure 5: Change in probability distributions from one board state to the next one.
</p>



## Conclusion

Our research is headed in the right direction and we are making good progress so far. Further, we are studying why there is a sudden change in probability distributions from one board state to the next one even though the action chosen was not an efficient one. We are also analyzing why the CNN favours the cells close to a bunch of attacked(orange) cells.