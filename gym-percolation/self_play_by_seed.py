import gym
import argparse
from agent import Agent,ExperimentLogger,load_agent

from itertools import product as iterproduct


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=False, default=None)
    parser.add_argument('--num-games', '-n', type=int, default=100)
    parser.add_argument('--file-out', required=True)
    parser.add_argument('--explore-rate', type=float, default=0.1)
    parser.add_argument('--seeds', nargs="+", type=int)
    parser.add_argument('-p',type=float,default=0.3)
    parser.add_argument('--board-size',type=int,default=5)
    parser.add_argument('--policy-gradient', required=False, default=False)
    
    args = parser.parse_args()

    agent = load_agent(args.agent, policy_gradient=args.policy_gradient)
    agent.exp_rate = args.explore_rate
    agent.set_logger(ExperimentLogger(args.file_out))
    agent.logger.update_metadata(explore_rate=agent.exp_rate, grid_size=(args.board_size,args.board_size), p=args.p)
    
    for seed, game_no in iterproduct(args.seeds,range(args.num_games)):
        env = gym.make("gym_percolation:Percolation-mode0-v0", grid_size=(args.board_size,args.board_size), p=args.p, np_seed=seed, 
                    enable_render=False)        
        observation = env.reset()
        env.seed()
        agent.logger.update_metadata(np_seed=seed, game_number=game_no)
        
        while len(env.action_space)>0:
            env = agent.select_move(env)
            
        agent.flush_logger()


if __name__ == "__main__":
    main()
