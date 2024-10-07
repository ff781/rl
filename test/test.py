import nets
import env
import data
import torch
import torch.nn as nn
from env import DMCEnv
from actor import DummyActor
from visuals import play_video
import numpy as np
from data import load_deer, enumerate_file_structure


def test_mlp():
    print("\nTesting MLP:")
    mlp = nets.MLP(input_dim=10, output_dim=5, layer_num=3, layer_size=32)
    print(mlp)
    print("MLP parameters:", sum(p.numel() for p in mlp.parameters()))
    x = torch.randn(2, 10)
    y = mlp(x)
    print(f"MLP input shape: {x.shape}, output shape: {y.shape}")

def test_stochastic_mlp():
    print("\nTesting StochasticMLP:")
    smlp = nets.StochasticMLP(input_dim=10, output_dim=5, layer_num=3, layer_size=32)
    print(smlp)
    print("StochasticMLP parameters:", sum(p.numel() for p in smlp.parameters()))
    x = torch.randn(2, 10)
    y = smlp(x)
    print(f"StochasticMLP input shape: {x.shape}")
    print(f"StochasticMLP output shapes: sample {y['sample'].shape}, mean {y['mean'].shape}, var {y['var'].shape}")

def test_conv_encoder():
    print("\nTesting ConvEncoder:")
    conv_encoder = nets.ConvEncoder(input_size=(3, 64, 64), output_size=128, base_depth=32)
    print(conv_encoder)
    print("ConvEncoder parameters:", sum(p.numel() for p in conv_encoder.parameters()))
    x = torch.randn(2, 3, 64, 64)
    y = conv_encoder(x)
    print(f"ConvEncoder input shape: {x.shape}, output shape: {y.shape}")

def test_conv_decoder():
    print("\nTesting ConvDecoder:")
    conv_decoder = nets.ConvDecoder(input_size=128, output_size=(3, 64, 64), base_depth=32)
    print(conv_decoder)
    print("ConvDecoder parameters:", sum(p.numel() for p in conv_decoder.parameters()))
    x = torch.randn(2, 128)
    y = conv_decoder(x)
    print(f"ConvDecoder input shape: {x.shape}, output shape: {y.shape}")

def test_env_actor(env_name, obs_type, num_steps=100):
    env = DMCEnv(env_name=env_name, obs_type=obs_type)
    actor = DummyActor(env.observation_space, env.action_space)
    
    obs, _ = env.reset()
    images = []
    from tqdm import tqdm
    for _ in tqdm(range(num_steps), desc=f"Simulating {env_name} with {obs_type} observations"):
        action, _ = actor.sample(torch.from_numpy(obs).float())
        obs, reward, terminated, truncated, info = env.step(action.numpy())
        images.append(env.render())
        if terminated or truncated:
            break
    
    return np.stack(images).transpose(0, 3, 1, 2)

def test_env_from_data(data, num_steps=100):
    images = data['observation'][0, :num_steps].numpy()
    return images

def test_envs():
    num_steps = 100
    grid_size = (6, 2)
    
    videos = []
    
    def process_env(env_name):
        # Real env
        videos.append(test_env_actor(f"{env_name}", "proprio", num_steps))
        videos.append(test_env_actor(f"{env_name}", "img", num_steps))
        
        # Offline datasets
        datasets = data.filter_data(env=env_name, px=64, relation="main")
        for dataset in datasets:
            dataset_data = load_deer(dataset, num_episodes=1)
            videos.append(test_env_from_data(dataset_data, num_steps))
    
    # Process each environment
    for env in ["walker_walk", "humanoid_walk", "cheetah_run"]:
        process_env(env)

    videos = np.stack(videos)
    play_video(videos, fps=60, grid_size=grid_size)

def test_all_groups():
    grouped_data = data.group_data_by_attributes()
    videos = []
    i = 0
    for key, paths in grouped_data.items():
        relation, env, collection_type, px, distraction_level = key
        if px != 64:
            continue
        i += 1
        print(f"{i}. env: {env[:10]}, col: {collection_type}" + (f", dlv: {distraction_level}" if distraction_level else ""))
        # Load one example from this group
        dataset = paths[0]  # Take the first path in the group
        dataset_data = data.load_deer(dataset, num_episodes=1)
        
        num_frames = min(500, dataset_data['observation'].shape[1])
        video = dataset_data['observation'][0, :num_frames].numpy()
        
        videos.append(video)
    # Stack all videos and play
    videos = np.stack(videos)
    play_video(videos, fps=30, grid_size=(5, 4))  # Adjust grid size as needed


def test_rssm_env(env_name="walker_walk", num_steps=100):
    # Create DMC environment with flattened observations
    dmc_env = DMCEnv(env_name, obs_type='img', img_size=64)
    # flattened_dmc_env = FlattenObservationWrapper(dmc_env)
    
    # Create RSSM environment
    rssm_env = env.create_rssm_env(dmc_env)
    actor = DummyActor(rssm_env.markov_state_space, rssm_env.action_space)

    # Test RSSM environment
    obs, _ = rssm_env.reset()
    frames = []
    for _ in range(num_steps):
        action, _ = actor.sample(rssm_env.markov_state().unsqueeze(0))
        obs, reward, done, _, _ = rssm_env.step(action)
        frame = rssm_env.render()
        frames.append(frame)
    
    # Play the video
    play_video(np.array(frames), fps=30)

if __name__ == "__main__":
    test_all_groups()
