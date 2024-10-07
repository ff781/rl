import torch
import torch.optim as optim
import meth
from env import create_rssm_env
from nets import Cat
from actor import TanhActor
from critic import EnsembleQCritic, DoubleQCritic
from logger import SimpleLogger
from trainer import DreamTrainer, ModelThenPolicyScheduler
from trainer.objective import RSSMEnvObjective, SACActorObjective, SACCriticObjective
from replay import ReplayBuffer
from env import DMCEnv
import math
import numpy as np
import data
from trainer import ClippedAdam




def train(
    actor,
    critic,
    model,
    scheduler,
    num_epochs,
    num_batches,
    batch_size,
    seq_len,
    horizon,
    env_name,
):
    # Create logger
    logger = SimpleLogger(project_name="odv2_example")

    # Create replay buffer
    replay = ReplayBuffer()
    
    # Filter and load data
    filtered_paths = data.filter_data(env=env_name, collection_type='medium', relation='main', px=64)
    if not filtered_paths:
        raise ValueError(f"No data found for environment: {env_name}")
    
    loaded_data = data.load_deer(filtered_paths[0], num_episodes=5)
    replay.add(loaded_data)

    # Create objectives
    model_objective = RSSMEnvObjective(
        env=model,
        scales=dict(
            reward=1.0,
            kl=1.0,
            reconstruction=1.0,
            regularization=1.0,
        ),
    )
    actor_objective = SACActorObjective(
        actor=actor,
        critic=critic,
        env=model,
        scales=dict(
            actor=1.0,
            entropy=0.0,
        ),
    )
    critic_objective = SACCriticObjective(
        actor=actor,
        critic=critic,
        env=model,
        scales=dict(
            critic=1.0,
        ),
    )
    objectives = dict(
        model=model_objective,
        actor=actor_objective,
        critic=critic_objective,
    )

    # Create optimizers
    model_optimizer = ClippedAdam(model.parameters(), lr=3e-4, eps=1e-5, max_grad_norm=1e+3)
    actor_optimizer = ClippedAdam(actor.parameters(), lr=8e-5, eps=1e-5, max_grad_norm=1e+2)
    critic_optimizer = ClippedAdam(critic.parameters(), lr=8e-5, eps=1e-5, max_grad_norm=1e+2)
    optimizers = dict(
        model=model_optimizer,
        actor=actor_optimizer,
        critic=critic_optimizer,
    )

    # Create trainer
    trainer = DreamTrainer(
        model=model,
        actor=actor,
        critic=critic,
        replay=replay,
        scheduler=scheduler,
        optimizers=optimizers,
        objectives=objectives,
        horizon=horizon,
        logger=logger,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer.to(device)

    # Train
    trainer.train(
        epochs=num_epochs,
        batches=num_batches,
        bs=batch_size,
        sl=seq_len,
    )

    logger.finish()


if __name__ == "__main__":
    meth.seed_all(42)

    # Create environment
    env_name = "walker_walk"
    dmc_env = DMCEnv(env_name, obs_type='img', img_size=64)
    rssm_env = create_rssm_env(dmc_env)

    print(f"state_space: {rssm_env.state_space}")
    print(f"hidden_space: {rssm_env.hidden_space}")
    print(f"markov_state_space: {rssm_env.markov_state_space}")
    print(f"action_space: {rssm_env.action_space}")

    # Create actor and critic
    actor = Cat(
        TanhActor(rssm_env.markov_state_space, rssm_env.action_space, hidden_size=256, layer_num=3),
        keys=['hidden', 'state'],
        target_keys=['action', 'log_prob'],
    )
    critic = DoubleQCritic(rssm_env.markov_state_space, rssm_env.action_space, hidden_size=256, layer_num=3)

    num_epochs = 6
    num_batches = 4
    batch_size = 3
    seq_len = 2
    horizon = 10

    # Create scheduler
    scheduler = ModelThenPolicyScheduler(epochs=num_epochs, f=0.5)

    # Train
    train(
        actor=actor,
        critic=critic,
        model=rssm_env,
        scheduler=scheduler,
        num_epochs=num_epochs,
        num_batches=num_batches,
        batch_size=batch_size,
        seq_len=seq_len,
        horizon=horizon,
        env_name=env_name,
    )