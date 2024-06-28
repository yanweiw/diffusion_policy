
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
    wristrgb_list = []
    scenergb_list = []

    for file in tqdm(data_files):
        filepath = os.path.join(data_folder, file)
        print('loading file: ', filepath)
        with open(filepath, 'rb') as handle:
                (xyz, quat, ee, jointpos, scenergb, wristrgb, scenedep, wristdep) = pickle.load(handle)
        
        # end effector position and gripper state
        state = np.concatenate([xyz, quat, ee], axis=1) 
        print('state shape: ', state.shape)
        action = np.concatenate([state[1:], state[[-1]]], axis=0)
        print('action shape: ', action.shape)
        
        assert action.shape[0] == len(wristrgb), f"action shape: {action.shape}, wristrgb shape: {wristrgb.shape}"
        assert action.shape[0] == len(scenergb), f"action shape: {action.shape}, scenergb shape: {scenergb.shape}"
        wrist = []
        for rgb in wristrgb:
            rgb = cv2.resize(rgb, (320, 240))
            wrist.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)) 
        wrist = np.stack(wrist)
        print('obs shape: ', wrist.shape)

        scene = []
        for rgb in scenergb:
            rgb = cv2.resize(rgb, (320, 240))
            scene.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        scene = np.stack(scene)
        print('scene shape: ', scene.shape)
        
        state_list.append(state)
        action_list.append(action)
        wristrgb_list.append(wrist)
        scenergb_list.append(scene)
        episode_end += len(action)
        episode_ends.append(episode_end) 

    save_path = os.path.join('/home/rss/diffusion_policy/data/kitchen', datasource) + '.zarr'
    data_root = zarr.open_group(save_path, mode='w')
    data = data_root.create_group('data')
    data.create_dataset('state', data=np.concatenate(state_list, axis=0), dtype='float32')
    data.create_dataset('action', data=np.concatenate(action_list, axis=0), dtype='float32')
    data.create_dataset('wrist', data=np.concatenate(wristrgb_list, axis=0), dtype='uint8')
    data.create_dataset('scene', data=np.concatenate(scenergb_list, axis=0), dtype='uint8')
    meta = data_root.create_group('meta')
    meta.create_dataset('episode_ends', data=np.array(episode_ends))
    print('data saved to: ', os.path.abspath(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasource', required=True, help='Path to data file')
    args = parser.parse_args()
    main(args.datasource)