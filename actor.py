import torch
import torch.distributions
import gymnasium as gym
import torch.nn as nn
import math
import numpy as np
import nets
from tensordict import TensorDict

dists = torch.distributions





class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

    def get_dist(self, state):
        raise NotImplementedError

    def sample(self, state):
        dist = self.get_dist(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return TensorDict(dict(action=action, log_prob=log_prob), batch_size=action.shape[:-1])
    
    def forward(self, state):
        return self.sample(state)

class DummyActor(Actor):
    def __init__(self, state_space, action_space):
        super().__init__(state_space, action_space)

    def get_dist(self, state):
        batch_shape = state.shape[:-len(self.state_space.shape)]
        return torch.distributions.Uniform(
            low=torch.tensor(self.action_space.low).expand(batch_shape + self.action_space.shape),
            high=torch.tensor(self.action_space.high).expand(batch_shape + self.action_space.shape)
        )

class GaussianActor(Actor):
    def __init__(self, state_space, action_space, hidden_size=256, layer_num=2, init_std=1.0, min_std=1e-1, mean_scale_factor=2.0):
        super().__init__(state_space, action_space)
        self.det_net = nets.MLP(
            input_dim=np.prod(state_space.shape),
            output_dim=2 * np.prod(action_space.shape),
            layer_num=layer_num,
            layer_size=hidden_size,
            activation=nn.ELU(inplace=True)
        )
        self.std_act = nets.softplus_activation(init_std, min_std)
        self.mean_scale_factor = mean_scale_factor

    def get_dist(self, state):
        raw_mean, raw_std = torch.chunk(self.det_net(state), 2, dim=-1)
        mean = self.mean_scale_factor * torch.tanh(raw_mean / self.mean_scale_factor)
        std = self.std_act(raw_std)
        return torch.distributions.Normal(mean, std)

class TanhActor(Actor):
    def __init__(self, state_space, action_space, hidden_size=256, layer_num=2, init_std=1.0, min_std=1e-1, mean_scale_factor=2.0):
        super().__init__(state_space, action_space)
        self.det_net = nets.MLP(
            input_dim=np.prod(state_space.shape),
            output_dim=2 * np.prod(action_space.shape),
            layer_num=layer_num,
            layer_size=hidden_size,
            activation=nn.ELU(inplace=True)
        )
        self.std_act = nets.softplus_activation(init_std, min_std)
        self.mean_scale_factor = mean_scale_factor

    def get_dist(self, state):
        raw_mean, raw_std = torch.chunk(self.det_net(state), 2, dim=-1)
        mean = self.mean_scale_factor * torch.tanh(raw_mean / self.mean_scale_factor)
        std = self.std_act(raw_std)
        return _SquashedNormal(mean, std)

class WeightedActor(Actor):
    def __init__(self, actors, weights):
        super().__init__(actors[0].state_space, actors[0].action_space)
        self.actors = nn.ModuleList(actors)
        self.weights = torch.tensor(weights) / sum(weights)

    def sample(self, state):
        batch_shape = state.shape[:-len(self.state_space.shape)]
        num_samples = np.prod(batch_shape)
        
        # Flatten the batch dimensions
        flat_state = state.view(-1, *self.state_space.shape)
        
        # Sample actor indices based on weights
        actor_indices = torch.multinomial(self.weights, num_samples, replacement=True)
        
        actions = []
        log_probs = []
        
        for i, actor in enumerate(self.actors):
            mask = (actor_indices == i)
            if mask.any():
                actor_state = flat_state[mask]
                _ = actor.sample(actor_state)
                actor_action, actor_log_prob = _['action'], _['log_prob']
                actions.append(actor_action)
                log_probs.append(actor_log_prob)
        
        # Combine actions and log_probs
        combined_actions = torch.cat(actions, dim=0)
        combined_log_probs = torch.cat(log_probs, dim=0)
        
        # Reorder to match the original batch shape
        reorder_indices = torch.argsort(actor_indices)
        actions = combined_actions[reorder_indices]
        log_probs = combined_log_probs[reorder_indices]
        
        # Reshape to match the original batch shape
        actions = actions.view(*batch_shape, *self.action_space.shape)
        log_probs = log_probs.view(*batch_shape)
        
        return TensorDict(dict(action=actions, log_prob=log_probs), batch_size=batch_shape)


class TanhTransform(torch.distributions.transforms.Transform):
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y.clamp(-0.99999997, 0.99999997))

    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - nn.functional.softplus(-2. * x))

class _SquashedNormal(torch.distributions.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = torch.distributions.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

# Test code
if __name__ == "__main__":
    # Mock state and action spaces
    state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    # Initialize actors
    gaussian_actor = GaussianActor(state_space, action_space, init_std=0.5, min_std=1e-3, mean_scale_factor=1.5)
    tanh_actor = TanhActor(state_space, action_space, init_std=0.5, min_std=1e-3, mean_scale_factor=1.5)
    weighted_actor = WeightedActor([gaussian_actor, tanh_actor], weights=[0.7, 0.3])

    # Test with mock state
    mock_state = torch.randn(1000, 4)  # Batch of 10 states

    print("Testing Gaussian Actor:")
    gaussian_output = gaussian_actor(mock_state)
    print(f"Action shape: {gaussian_output['action'].shape}")
    print(f"Log prob shape: {gaussian_output['log_prob'].shape}")
    print(f"Action range: [{gaussian_output['action'].min().item():.2f}, {gaussian_output['action'].max().item():.2f}]")

    print("\nTesting Tanh Actor:")
    tanh_output = tanh_actor(mock_state)
    print(f"Action shape: {tanh_output['action'].shape}")
    print(f"Log prob shape: {tanh_output['log_prob'].shape}")
    print(f"Action range: [{tanh_output['action'].min().item():.2f}, {tanh_output['action'].max().item():.2f}]")

    print("\nTesting Weighted Actor:")
    weighted_output = weighted_actor(mock_state)
    print(f"Action shape: {weighted_output['action'].shape}")
    print(f"Log prob shape: {weighted_output['log_prob'].shape}")
    print(f"Action range: [{weighted_output['action'].min().item():.2f}, {weighted_output['action'].max().item():.2f}]")
