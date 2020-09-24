import gym
import argparse
from agent import Agent,ExperimentLogger,load_agent

from itertools import product as iterproduct


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True, default=None)
    parser.add_argument('--input-data')
    parser.add_argument('--num-samples', '-n', type=int, default=100)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--indices', nargs="+", type=int)
    parser.add_argument('--board-size',type=int,default=5)
    parser.add_argument('--policy-gradient', required=False, default=False)
    
    args = parser.parse_args()

    agent = load_agent(args.agent, policy_gradient=args.policy_gradient)
    
    #load_data

    #select random samples from input data
    # OR
    #plot specific indices

    #run agent.fit() on selected samples

    #run plot_data() from ipynb file on these sets of X, Y, Yhat

    #save as image with index included in filename