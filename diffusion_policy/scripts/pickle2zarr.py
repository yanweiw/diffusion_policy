
import argparse
import numpy as np
import zarr
from tqdm import tqdm
import os, pickle
import cv2


def main(datasource):
    # read from folder 
    data_folder  = os.path.join(os.path.abspath('/mnt/data'), datasource)
    data_files = sorted(os.listdir(data_folder))
    data_files = [f for f in data_files if f.endswith('pkl')]

    episode_end = 0
    episode_ends = []
    state_list = []
    action_list = []
    img_list = []

    for file in tqdm(data_files):
        filepath = os.path.join(data_folder, file)
        print('loading file: ', filepath)
        with open(filepath, 'rb') as handle:
                (xyz, quat, ee, jointpos, scenergb, wristrgb, scenedep, wristdep) = pickle.load(handle)
        # plan = np.concatenate([plan[0::10], plan[[-1]]], axis=0) # select every 10th step, making sure the last step is included
        state = np.concatenate([xyz, quat, ee], axis=1) # end effector position and gripper state
        print('state shape: ', state.shape)
        next_xyz = np.concatenate([xyz[1:], xyz[[-1]]], axis=0)
        next_quat = np.concatenate([quat[1:], quat[[-1]]], axis=0)
        next_ee = np.concatenate([ee[1:], ee[[-1]]], axis=0)
        action = np.concatenate([next_xyz, next_quat, next_ee], axis=1) # end effector position and gripper state
        print('action shape: ', action.shape)
        assert action.shape[0] == len(wristrgb), f"action shape: {action.shape}, wristrgb shape: {wristrgb.shape}"
        img = []
        for rgb in wristrgb:
            rgb = cv2.resize(rgb, (320, 240))
            img.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)) 
        img = np.stack(img)
        print('obs shape: ', img.shape)
        
        state_list.append(state)
        action_list.append(action)
        img_list.append(img)
        episode_end += len(action)
        episode_ends.append(episode_end) 

    save_path = os.path.join('/home/rss/diffusion_policy/data/kitchen', datasource) + '.zarr'
    data_root = zarr.open_group(save_path, mode='w')
    data = data_root.create_group('data')
    data.create_dataset('state', data=np.concatenate(state_list, axis=0), dtype='float32')
    data.create_dataset('action', data=np.concatenate(action_list, axis=0), dtype='float32')
    data.create_dataset('img', data=np.concatenate(img_list, axis=0), dtype='uint8')
    meta = data_root.create_group('meta')
    meta.create_dataset('episode_ends', data=np.array(episode_ends))
    print('data saved to: ', os.path.abspath(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasource', required=True, help='Path to data file')
    args = parser.parse_args()
    main(args.datasource)