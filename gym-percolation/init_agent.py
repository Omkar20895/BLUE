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
    parser.add_argument('output_file')
    args = parser.parse_args()

    network = getattr(networks, args.network)
    input_shape = args.board_size
    model = network.get_model(input_shape)

    new_agent = Agent(model, args.board_size)

    with h5py.File(args.output_file, 'w') as outf:
        new_agent.serialize(outf) 


if __name__ ==  '__main__':
    main()
