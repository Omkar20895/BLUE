import gym
import gym_percolation

import numpy as np
import uuid

RAND_SEED = str(uuid.uuid1())

env = gym.make("gym_percolation:Percolation-mode0-v0", grid_size=(25,25), p=0.35)

observation = env.reset()
for _ in range(1000):
  env.render()
  #action = env.action_space.sample() # your agent here (this takes random actions)
  action = {'x': np.random.randint(25), 'y': np.random.randint(25)}
  observation, reward, done, info = env.step(action)
  print(reward, done, info)

  if done:
    _ = input('Finished! Press any key to exit...')
    observation = env.reset()
    break
env.close()
