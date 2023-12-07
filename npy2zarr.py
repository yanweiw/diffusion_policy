
import argparse
import numpy as np
import zarr
from tqdm import tqdm
import os

pos_dims_before = {
    "ee_pos": list(range(10, 13)),  # 3 pos
    "gripper": list(range(8, 9)),  # 1 pos
    "red": list(range(13, 16)),  # 3 pos
    "green": list(range(20, 23)),
    "blue": list(range(27, 30)),
    "yellow": list(range(34, 37)),
    "cyan": list(range(41, 44)),
    "magenta": list(range(48, 51)),
}

pos_dims= {
    "ee_pos": list(range(0, 3)),  # 3 pos
    "gripper": list(range(3, 4)),  # 1 pos
    "red": list(range(4, 7)),  # 3 pos
    "green": list(range(7, 10)),
    "blue": list(range(10, 13)),
    "yellow": list(range(13, 16)),
    "cyan": list(range(16, 19)),
    "magenta": list(range(19, 22)),
}

def main(datasource='logged_plans_03'):
    # read from folder 
    data_folder  = os.path.join(os.path.abspath('../interactpolicy/ipolicy/'), 'data', datasource)
    data_files = sorted(os.listdir(data_folder))
    episode_end = 0
    episode_ends = []
    action_list = []
    state_list = []
    plan_list = []

    for file in tqdm(data_files):
        if '.txt' in file:
            continue
        plan = np.load(os.path.join(data_folder, file))
        # plan = np.concatenate([plan[0::10], plan[[-1]]], axis=0) # select every 10th step, making sure the last step is included
        next_plan = np.concatenate([plan[1:], plan[[-1]]], axis=0)
        
        action = np.concatenate([next_plan[:, pos_dims_before['ee_pos']], next_plan[:, pos_dims_before['gripper']]], axis=1) # end effector position and gripper state
        ee_gripper_cube = []
        for s in ["ee_pos", "gripper", "red", "green", "blue", "yellow", "cyan", "magenta"]:
            ee_gripper_cube.append(plan[:, pos_dims_before[s]])
        state = np.concatenate(ee_gripper_cube, axis=1) # ee pos, gripper, red, green, blue, yellow, cyan, magenta
        action_list.append(action)
        state_list.append(state)
        plan_list.append(plan)
        episode_end += len(plan)
        episode_ends.append(episode_end) 

    data_root = zarr.open_group('data/' + datasource + '.zarr', mode='w')
    data = data_root.create_group('data')
    data.create_dataset('action', data=np.concatenate(action_list, axis=0), dtype='float32')
    data.create_dataset('state', data=np.concatenate(state_list, axis=0), dtype='float32')
    data.create_dataset('plan', data=np.concatenate(plan_list, axis=0), dtype='float32')
    meta = data_root.create_group('meta')
    meta.create_dataset('episode_ends', data=np.array(episode_ends))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasource', required=True, help='Path to data file')
    args = parser.parse_args()
    main(args.datasource)
