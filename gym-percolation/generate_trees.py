import gym
from treesearch import GameTreeNode,DummyEnv
import argparse
import json
from itertools import product as iterproduct

def plot_data(tree,recurse=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    flatui = ["black", "tomato", "deepskyblue", "darkseagreen"]
    #sns.set_palette(flatui)
    new_cmap = ListedColormap(flatui)
    
    for idx,child in enumerate(tree.split_children_good_bad()[0]):
        if not child:
            continue
        ay,ax = child.action["y"], child.action["x"]
    
        fig, axes = plt.subplots(1, 2)#,figsize=(10,5))
        fig.set_figwidth(8)
        axes[0].imshow((tree.game_state.state), origin='lower', cmap=new_cmap,vmin=0,vmax=3)
        axes[1].imshow((child.game_state.state), origin='lower', cmap=new_cmap,vmin=0,vmax=3)
        axes[0].plot(ay,ax, 'ro')
        axes[1].plot(ay,ax, 'ro')
        plt.savefig(f"/tmp/{tree.depth}-{child.depth}-{idx}.png")
        if recurse:
            return plot_data(child,recurse=True)
    return

    
def whole_tree_training_data(root):
    output=root.generate_training_data()
    for c in root.split_children_good_bad()[0]:
        output+=whole_tree_training_data(c)
    return output

def format_json_data(json_data):

    result_list = []
    
    for index in range(len(json_data['X'])):
        temp_dict = {}
        if len(json_data['X'][index]) != 0:
            for i in range(len(json_data['Y_good'][index])):
                 temp_dict = {}
                 temp_dict['board_state'] = json_data['X'][index]
                 temp_dict['action'] = json_data['Y_good'][index][i]
                 temp_dict['move_quality'] = 1
                 result_list.append(temp_dict)

            for i in range(len(json_data['Y_bad'][index])):
                 temp_dict = {}
                 temp_dict['board_state'] = json_data['X'][index]
                 temp_dict['action'] = json_data['Y_bad'][index][i]
                 temp_dict['move_quality'] = -1
                 result_list.append(temp_dict)

    json_data = {'1': result_list}
             
    #file_path = './omkar.json'
    #json.dump(json_data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    print("Done")
    return json_data

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', '-n', type=int, default=100)
    parser.add_argument('--file-out', required=True)
    parser.add_argument('--make-plots',default=False,type=bool)
    parser.add_argument('-p',type=float,default=0.3)
    parser.add_argument('--board-size',type=int,default=5)
    
    args = parser.parse_args()
    trees=[]
    training_data=[]
    for n in range(args.num_games):
        
        env = gym.make("gym_percolation:Percolation-mode0-v0", grid_size=(args.board_size,args.board_size), p=args.p, 
            enable_render=False)        
        observation = env.reset()
        print(f"building tree {n}")
        tree = GameTreeNode(env, None, None)
        tree.full_tree = False
        print(f"growing tree {n}")
        tree.grow_tree()
        print(sorted([i.moves_to_end for i in tree.children]))
        trees.append(tree)
        training_data += whole_tree_training_data(tree)

    json_dict = dict(X=training_data[::3],Y_good=training_data[1::3],Y_bad=training_data[2::3])

    json_data = format_json_data(json_dict)
    
    json.dump(json_data, open(args.file_out,"w"),  separators=(',', ':'), sort_keys=True, indent=4)
    if args.make_plots:
        plot_data(trees[0],recurse=True)
    return trees
    


if __name__ == "__main__":
    trees = main()
