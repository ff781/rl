import glob
import os
import posixpath
import h5py
import numpy as np
import torch
from torch import Tensor
from tensordict import TensorDict


def load_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        action = torch.tensor(f['action'][:], dtype=torch.float32)
        # discount = torch.tensor(f['discount'][:], dtype=torch.float32)
        observation = torch.tensor(f['observation'][:], dtype=torch.uint8)
        reward = torch.tensor(f['reward'][:], dtype=torch.float32)
    
    bs = action.shape[0]//501, 501
    action = action.reshape(*bs, *action.shape[1:])
    observation = observation.reshape(*bs, *observation.shape[1:])
    reward = reward.reshape(*bs, *reward.shape[1:])

    td = TensorDict(dict(
        action=action,
        observation=observation,
        reward=reward,
        terminal=torch.zeros(bs)
    ), batch_size=bs)
    
    return td

def load_npz_file(file_path):
    data = np.load(file_path)
    action = torch.tensor(data['action'], dtype=torch.float32)
    observation = torch.tensor(data['image'], dtype=torch.uint8).permute(0,3,1,2)
    reward = torch.tensor(data['reward'], dtype=torch.float32)
    bs = *reward.shape,
    return TensorDict(dict(
        action=action,
        observation=observation,
        reward=reward,
        terminal=torch.zeros(bs)
    ), batch_size=bs).unsqueeze(0)

def load_file(file_path):
    if file_path.endswith(".hdf5"):
        return load_hdf5(file_path)
    elif file_path.endswith(".npz"):
        return load_npz_file(file_path)
    else:
        assert False, file_path
def load_deer(deer, num_episodes=None):
    r = []
    t = 0
    for f in glob.glob(posixpath.join(deer, '*')):
        r.append(load_file(f))
        # t += r[-1].shape[0]
        t += r[-1].shape[0]
        if num_episodes is not None and t >= num_episodes:
            break
    r = torch.cat(r)
    _ = r
    if num_episodes is not None:
        r = r[:num_episodes]

    # Unsqueeze all tensors with shape = batch_size along the last dim
    for key, value in r.items():
        if value.shape == r.batch_size:
            r[key] = value.unsqueeze(-1)

    return r



def get_path_key(path):
    parts = path.split(os.sep)
    relation = None
    for _ in ["main", "distracting", "multitask"]:
        if _ in parts:
            relation = _
            break
    else:
        assert False, parts
    env = None
    map = dict(
        cheetah="cheetah_run",
        walker="walker_walk",
        humanoid="humanoid_walk",
    )
    for _ in ["cheetah", "walker", "humanoid"]:
        if _ in path:
            env = map.get(_, _)
            break
    else:
        assert False, path
    collection_type = None
    for _ in ["random", "medium_expert", "medium_replay", "medium", "expert"]:
        if _ in path:
            collection_type = _
            break
    else:
        assert False, path
    px = None
    for _ in ["64px", "84px"]:
        if _ in parts:
            px = int(_[:-2])
            break
    else:
        assert False, parts
    distraction_level = None
    if relation == "distracting":
        for _ in ["easy", "medium", "hard"]:
            if _ in parts:
                distraction_level = _
                break
        else:
            assert False, parts
    return dict(relation=relation, env=env, collection_type=collection_type, px=px, distraction_level=distraction_level)

def enumerate_file_structure(root=posixpath.join("$PROJ_HOME", "data/vd4rl")):
    root = os.path.expandvars(os.path.expanduser(root))
    root = os.path.normpath(root)
    if not os.path.exists(root):
        raise FileNotFoundError(f"The specified root directory does not exist: {root}")

    for dirpath, dirnames, filenames in os.walk(root):
        if filenames:
            key = get_path_key(dirpath)
            # print(f"Path: {dirpath}, Key: {key}")
            yield (key | dict(path=dirpath))
            # data = load_deer(dirpath, num_episodes=5)
            # print(f"Data: {data}")

def group_data_by_attributes():
    grouped = {}
    for item in enumerate_file_structure():
        key = (
            item['relation'],
            item['env'],
            item['collection_type'],
            item['px'],
            item['distraction_level']
        )
        grouped.setdefault(key, []).append(item['path'])
    return grouped

def filter_data(grouped_data=group_data_by_attributes(),**filters):
    results = []
    for key, paths in grouped_data.items():
        if all(filters.get(attr) is None or filters.get(attr) == value 
               for attr, value in zip(['relation', 'env', 'collection_type', 'px', 'distraction_level'], key)):
            results.extend(paths)
    return results


def main():
    a = list(enumerate_file_structure())
    grouped_data = {}
    for item in a:
        key = (item['relation'], item['px'])
        grouped_data.setdefault(key, []).append(item)

    for (relation, px), group in grouped_data.items():
        print(f"\nRelation: {relation}, Pixel: {px}")
        for item in group:
            print(f"  env: {item['env'][:10]}, col: {item['collection_type']}" + (f", dlv: {item['distraction_level']}" if item['distraction_level'] else ""))

if __name__ == "__main__":
    main()
