# BLUE
Repository for CCNR network percolation game BLUE

## Helpful resources

- [Gym-Maze code using pygame](https://github.com/MattChanTK/gym-maze/blob/master/gym_maze/envs/maze_view_2d.py)
- [Reinforcement learning - Part 3: Creating your own gym environment](https://www.novatec-gmbh.de/en/blog/creating-a-gym-environment/)


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


