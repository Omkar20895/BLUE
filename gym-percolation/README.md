# BLUE
Repository for CCNR network percolation game BLUE


## Introduction:

One of the major challenges network science faces is making the networks more resilient and as we are progressing into the future of more connectivity than ever, there is a high demand for resilient and robust networks. In the study of network resilience, various toy models are used to represent the network’s response to perturbation. We propose to utilize these models, to develop a network resilience game "BLUE" which can then be learned by a reinforcement learning agent. The game is inspired by AlphaGo, a computer program with an AI agent playing the game Go.

## GamePlay:

We perform experiments over varied board sizes. The green cells represent available cells to attack, blue cells represent affected cells, orange cells represent attacked cells and black cells represent the pre-occupied cells. The goal in the game is to find the minimum number of cells which when attacked turn the board blue. We also use different percolation probabilities to fill the board with black cells during board initialization. Approximately p\*N cells are pre-occupied, i.e marked as black. We will be dealing with percolation probabilities in the range of [0.5, 5] and the probability of 0.3 is of particular interest as it is closely related to real-world networks.

At each step in the game, we track the board state, the action space that consists of all the possible actions available in that board state, and also the action that resulted in the current board state. We generate the action space using spatial heuristics that consider the immediate neighboring cells for each cell to determine if attacking that cell is a legitimate move.


Table 1: Example of gameplay on a 15x15 grid. The Player aims to select cells (orange) at each step to span blocked regions (black) and maximize occupied (blue) connected regions.

STEP 1 | STEP 2 | STEP 3 | STEP 4
------ | ------ | ------ | ------
<img width="350" height="250" src="https://i.imgur.com/qElnLKi.png"> | <img width="350" height="250" src="https://i.imgur.com/SsnGeOe.png"> | <img width="350" height="250" src="https://i.imgur.com/UywgEpp.png"> | <img width="350" height="250" src="https://i.imgur.com/6k0VxEM.png">


We generate action space for a given board state using heuristics described as follows:

**maximum neighbors:**
For each green cell in a given board state we consider the 8 neighboring cells and include the green cell with maximum number of neighbors colored blue, black or orange into the action space.


## Approach 1:


All the experiments were performed on a 5x5 board. The main idea is to create and train an agent that performs better with each iteration. The agent is basically a Convolutional Neural Network assigning a numerical reward value for each pair of board state and action space. 


In the first iteration we use 10 random seeds to generate 10 specific board game initializations. We also generate 100 experiments for each seed selecting random actions in each experiment and playing the experiment till the end. A total of 1000 experiments are generated and this will be our first dataset to train a CNN. 

We pre-process the dataset before using it to train the neural network by calculating the average number of steps taken to win among all the game experiments, referred to as **par** of the dataset. For each game experiment we calculate reward using the formula (average_steps-present_steps)/average_steps, this type of reward system gives a higher reward to the games that win the game in lesser number of steps than the par and lower reward otherwise. We assign all the action, boardstate pairs in a game experiment with this reward. This is our mechanism of penalizing and rewarding game experiments that are above(greater than) and below(lesser than) par respectively. 


<p align="center">
  <img width="350" height="550" src="https://i.imgur.com/MWCB2rz.png"><br>
  Figure 1: Structure of the CNN used for training the agent
</p>


In the subsequent iterations we use the same 10 random seeds to generate board state initializations and use CNN to chose the action for us, instead of selecting a random action. The CNN takes each board state and action from the action space of the board state and picks he action with the highest reward. In subsequent iterations we follow the cycle of generating data from the previous iteration's CNN and training a CNN with this generated data. We track the average number of steps taken to win for each seed. Ideally, the average steps to win must decrease with each subsequent iteration. 



The following python code show the training regime over 18 iterations: 

```python
import os

os.makedirs("./omkar_experiments")
for index in range(1, 19):
    os.makedirs("./omkar_experiments/iteration"+str(index))
    if index == 1:
        os.system("python3 self_play.py --file-out './omkar_experiments/iteration"+str(index)+"/data.json'")
        #os.system("python3 self_play.py --file-out './omkar_experiments/iteration1/data.json'")
    else:
        os.system('python3 self_play.py --file-out "./omkar_experiments/iteration'+str(index)+'/data.json" --agent "./omkar_experiments/iteration'+str(index-1)+'/model.sav"')
    os.system('python3 train.py --data-path "./omkar_experiments/iteration'+str(index)+'/data.json" --agent-path "./omkar_experiments/iteration'+str(index)+'/model.sav"')


os.system('python3 plot.py --data-path "./omkar_experiments" --iterations 18')
```

The following were the results of the experiments: 

<p align="center">
  <img width="1000" height="350" src="https://i.imgur.com/Od4xhoa.png"><br>
  Seed 1:
</p>

<p align="center">
  <img width="1000" height="350" src="https://i.imgur.com/uHXdApG.png"><br>
  Seed 2:
</p>

<p align="center">
  <img width="1000" height="350" src="https://i.imgur.com/G2l0ysA.png"><br>
  Seed 3:
</p>

<p align="center">
  <img width="1000" height="350" src="https://i.imgur.com/QShEIf0.png"><br>
  Seed 4:
</p>

<p align="center">
  <img width="1000" height="350" src="https://i.imgur.com/RtlvPg1.png"><br>
  Seed 5:
</p>

<p align="center">
  <img width="1000" height="350" src="https://i.imgur.com/nMVhakb.png"><br>
  Seed 6:
</p>

<p align="center">
  <img width="1000" height="350" src="https://i.imgur.com/ZmKUDga.png"><br>
  Seed 7:
</p>

<p align="center">
  <img width="1000" height="350" src="https://i.imgur.com/XMucDEP.png"><br>
  Seed 8:
</p>

<p align="center">
  <img width="1000" height="350" src="https://i.imgur.com/dwaCHWG.png"><br>
  Seed 9:
</p>

<p align="center">
  <img width="1000" height="350" src="https://i.imgur.com/RRt8RIX.png"><br>
  Seed 10:
</p>


As we can see from the graphs, we got mixed results and the average steps to win was not continuosly decreasing. We are currently working on a new approach called Policy Gradient learning and the results will be upadated here shortly.

## Helpful resources

- [Gym-Maze code using pygame](https://github.com/MattChanTK/gym-maze/blob/master/gym_maze/envs/maze_view_2d.py)
- [Reinforcement learning - Part 3: Creating your own gym environment](https://www.novatec-gmbh.de/en/blog/creating-a-gym-environment/)



# Appendix:

## Training Details:

Files:

self_play.py: file used to generate experiments

Arguments:

--agent (optional):  path to the model using which we select actions

--num-games (optional): number of games to be generated

--file-out (required): file to store the generated data

--explore-rate (optional)

--run-parallel (optional) (boolean)



train.py: used to train a model using the current iteration’s data

Arguments:

--data-path (required): location of the json file on which we train the model

--agent-path (required): file path to save the model

--epochs(optional)


The following file paths are relative and can be adjusted to your own local file paths for convenience

### Iteration1:

Generating data:

```python
python3 self_play.py --file-out "./mini_experiments/iteration1/data_exp_1.json"
```

Training the agent:

```python
python3 train.py --data-path "./mini_experiments/iteration1/data_exp_1.json" --agent-path "./mini_experiments/iteration1/model.sav"
```


### Iteration2: 

Generating data:

```python
python3 self_play.py --file-out "./mini_experiments/iteration2/data_exp_1.json" --agent "./mini_experiments/iteration1/model.sav"
```

Training the agent:

```python
python3 train.py --data-path "./mini_experiments/iteration2/data_exp_1.json" --agent-path "./mini_experiments/iteration2/model.sav"
```

### Iteration3: 

Generating data:

```python
python3 self_play.py --file-out "./mini_experiments/iteration3/data_exp_1.json" --agent "./mini_experiments/iteration2/model.sav"
```

Training the agent:

```python
python3 train.py --data-path "./mini_experiments/iteration3/data_exp_1.json" --agent-path "./mini_experiments/iteration3/model.sav"
```

Subsequent iterations are trained in similar fashion as above. 


