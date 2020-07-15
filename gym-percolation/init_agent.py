import json
import h5py
import argparse
import networks
import numpy as np
from agent import Agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=5)
    parser.add_argument('--network', default='small')
    parser.add_argument('--output-file')
    args = parser.parse_args()

    network = getattr(networks, args.network)
    board_size = args.board_size
    model = network.get_model(board_size)

    new_agent = Agent(model, args.board_size)
    new_agent.serialize(args.output_file) 


if __name__ ==  '__main__':
    main()
