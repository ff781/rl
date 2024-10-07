import gymnasium as gym
import numpy as np
from tensordict import TensorDict
import torch
import nets
from torch import nn





class Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = None
        self.action_space = None

    def step(self, action):
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        pass

    def render(self):
        pass


class DMCEnv(Env):
    def __init__(self, env_name, obs_type='proprio', img_size=64, action_repeat=1, enable_rendering=True):
        super().__init__()
        from dm_control import suite
        domain, task = env_name.split('_', 1)
        self._env = suite.load(domain_name=domain, task_name=task)
        self.env_name = env_name
        self.obs_type = obs_type
        self.img_size = img_size
        self.action_repeat = action_repeat
        self.enable_rendering = enable_rendering
        
        # Normalize action space to [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=self._env.action_spec().shape, dtype=np.float32
        )
        self.observation_space = self._get_obs_space()

    def _get_obs_space(self):
        if self.obs_type == 'proprio':
            obs_spec = self._env.observation_spec()
            spaces = {}
            for key, value in obs_spec.items():
                if value.shape == ():
                    shape = (1,)
                else:
                    shape = value.shape
                
                # Normalize observations to [-1, 1]
                spaces[key] = gym.spaces.Box(
                    low=-1, high=1, shape=shape, dtype=np.float32
                )
            return gym.spaces.Dict(spaces)
        elif self.obs_type == 'img':
            return gym.spaces.Box(low=0, high=255, shape=(3, self.img_size, self.img_size), dtype=np.uint8)
        else:
            raise ValueError(f"Invalid obs_type: {self.obs_type}")

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.action_repeat):
            time_step = self._env.step(action)
            total_reward += time_step.reward
            if time_step.last():
                break
        next_obs = self._get_obs(time_step)
        done = time_step.last()
        return next_obs, total_reward, done, False, {}

    def reset(self, seed=None, options=None):
        time_step = self._env.reset()
        return self._get_obs(time_step), {}

    def _get_obs(self, time_step):
        if self.obs_type == 'proprio':
            obs = {}
            for key, value in time_step.observation.items():
                obs[key] = 2 * (value - self.observation_space[key].low) / (self.observation_space[key].high - self.observation_space[key].low) - 1
            return obs
        elif self.obs_type == 'img':
            return self._env.physics.render(height=self.img_size, width=self.img_size, camera_id=0).transpose(2, 0, 1)
        else:
            raise ValueError(f"Invalid obs_type: {self.obs_type}")

    def render(self):
        return self._env.physics.render(height=self.img_size, width=self.img_size, camera_id=0)


class RSSMEnv(Env, nn.Module):
    def __init__(self, obs_space, action_space, obs_encoder, encoded_obs_to_hidden_model, det_state_model, stoch_state_model, update_state_model, obs_decoder, reward_model, obs_type='proprio'):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = action_space
        self.obs_encoder = obs_encoder
        self.encoded_obs_to_hidden_model = encoded_obs_to_hidden_model
        self.det_state_model = det_state_model
        self.stoch_state_model = stoch_state_model
        self.update_state_model = update_state_model
        self.obs_decoder = obs_decoder
        self.reward_model = reward_model
        self.state_size = self.det_state_model.state_size
        self.hidden_size = self.det_state_model.hidden_size
        self.state_space = gym.spaces.Box(low=-1, high=1, shape=(self.state_size,), dtype=np.float32)
        self.hidden_space = gym.spaces.Box(low=-1, high=1, shape=(self.hidden_size,), dtype=np.float32)
        self.markov_state_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.state_space.shape[0] + self.hidden_space.shape[0],),
            dtype=np.float32
        )
        self.markov_state_space_size = self.markov_state_space.shape[0]
        self.hidden = None
        self.obs_type = obs_type

    def preprocess_obs(self, obs):
        if self.obs_type == 'img':
            # return obs.float() / 255.0 - 0.5
            return obs.float() / 127.5 - 1.0
        return obs

    def postprocess_obs(self, obs):
        if self.obs_type == 'img':
            return ((obs + 1.0) * 127.5).to(torch.uint8)
        return obs

    def reset(self, seed=None, options=None, batch_size=None):
        if batch_size is None:
            batch_size = 1
        self.hidden = torch.zeros(batch_size, self.hidden_size).to(next(self.parameters()).device)
        obs = self.obs_decoder(TensorDict(
            dict(
                hidden=self.hidden,
                state=self.stoch_state_model(self.hidden),
            ), batch_size=batch_size
        ))
        return self.postprocess_obs(obs), {}

    def step(self, action):
        _ = self.simulate(action.unsqueeze(0), initial_hidden=self.hidden)[0]
        return self.postprocess_obs(_['obs']), _['reward'], _['terminal'], False, {}

    def markov_state_space(self):
        return self.markov_state_space

    def markov_state(self, batch_size=None, hidden=None):
        hidden = hidden if hidden is not None else self.hidden
        return TensorDict(
            dict(
                hidden=hidden,
                state=self.stoch_state_model.forward_all(hidden),
            ), batch_size=batch_size or hidden.shape[:-1]
        )
    
    def initial_markov_state(self, batch_size=None):
        return self.markov_state(batch_size=batch_size, hidden=torch.zeros(batch_size, self.hidden_size))

    def encode_observations(self, observations):
        encoded_obs = self.obs_encoder(self.preprocess_obs(observations))
        hidden = self.encoded_obs_to_hidden_model(encoded_obs)
        return self.markov_state(hidden=hidden)

    def simulate(self, actions, steps=None, batch_size=None, initial_hidden=None, initial_state=None, observations=None):
        if steps is None:
            steps = actions.shape[0]
        batch_size = batch_size
        if batch_size is None:
            if hasattr(actions, 'shape'):
                batch_size = actions.shape[1:-1]
            elif hasattr(observations, 'shape'):
                batch_size = observations.shape[1:-1]
            elif hasattr(initial_hidden, 'shape'):
                batch_size = initial_hidden.shape[:-1]
            elif hasattr(initial_state, 'shape'):
                batch_size = initial_state.shape[:-1]
            else:
                raise ValueError("Batch size not specified and could not be inferred from actions, observations, initial_hidden, or initial_state.")
        batch_size = (batch_size,) if isinstance(batch_size, int) else batch_size

        if initial_hidden is None:
            initial_hidden = torch.zeros(*batch_size, self.hidden_size).to(next(self.parameters()).device)

        initial_state = initial_state if initial_state is not None else self.stoch_state_model.forward_all(initial_hidden)

        hidden_list = [initial_hidden]
        state_list = [initial_state]
        prior_state_list = []
        if observations is not None:
            prior_state_list.append(state_list.pop())
        action_list = []
        obs_list = []
        reward_list = []
        terminal_list = []
        encoded_obs_list = []

        hidden = initial_hidden
        state = initial_state
        if not isinstance(state, TensorDict):
            state = TensorDict(
                dict(
                    sample=state,
                    mean=state,
                    std=torch.ones_like(state),
                ), batch_size=batch_size,
            )
        
        for step in range(steps):
            action = actions(nets.cat(TensorDict(
                dict(
                    hidden=hidden,
                    state=state["sample"],
                ), batch_size=batch_size,
            ))) if callable(actions) else actions[step]

            if observations is not None:
                prior_state_list.append(state)
                preprocessed_obs = self.preprocess_obs(observations[step])
                encoded_obs = self.obs_encoder(preprocessed_obs)
                state = self.update_state_model.forward_all(torch.cat([hidden, encoded_obs], dim=-1))
                encoded_obs_list.append(encoded_obs)

            hidden_state_action = TensorDict(
                dict(
                    hidden=hidden,
                    state=state["sample"],
                    action=action,
                ), batch_size=batch_size,
            )
            next_hidden = self.det_state_model(hidden_state_action)
            obs = self.obs_decoder(hidden_state_action.select('hidden', 'state'))
            reward, log_dict = self.reward_model(hidden_state_action.select('hidden', 'state'))

            hidden_list.append(hidden)
            state_list.append(state)
            action_list.append(action)
            obs_list.append(obs)
            reward_list.append(reward)
            
            hidden = next_hidden
            state = self.stoch_state_model.forward_all(next_hidden)
        
        if observations is not None:
            state_list.append(torch.zeros_like(state_list[0]))

        result = TensorDict({
            'hidden': torch.stack(hidden_list),
            'state': torch.stack(state_list),
            'action': torch.cat([torch.stack(action_list), torch.zeros_like(action_list[0]).unsqueeze(0)]),
            'obs': torch.cat([torch.stack(obs_list), torch.zeros_like(obs_list[0]).unsqueeze(0)]),
            **torch.cat([torch.stack(reward_list), torch.zeros_like(reward_list[0]).unsqueeze(0)]),
            'terminal': torch.cat([torch.zeros(steps, *batch_size), torch.zeros(1, *batch_size)]).to(action).unsqueeze(-1),
        }, batch_size=[steps+1, *batch_size])

        if observations is not None:
            result['prior_state'] = torch.cat([torch.stack(prior_state_list),], dim=0)
            result['encoded_obs'] = torch.cat([torch.stack(encoded_obs_list), torch.zeros_like(encoded_obs_list[0]).unsqueeze(0)], dim=0)

        return result

    def render(self):
        if self.obs_decoder is None:
            raise NotImplementedError("Render method not implemented for this environment.")
        
        with torch.no_grad():
            obs = self.obs_decoder(self.markov_state(hidden=self.hidden))

        if isinstance(obs, torch.Tensor):
            img = self.postprocess_obs(obs).squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = img.astype(np.uint8)  # Convert to 0-255 range
            return img
        elif isinstance(obs, dict) and 'image' in obs:
            img = self.postprocess_obs(obs['image']).squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = img.astype(np.uint8)  # Convert to 0-255 range
            return img
        else:
            raise ValueError("Unexpected observation format for rendering")

def create_rssm_env(
    dmc_env,
    hidden_size=256,
    state_size=30,
    base_depth=32,
    uncertainty_predictor=None,
    uncertainty_scale=None,
    dynamic_factor=1.0,
    dynamic_uncertainty_model=None
):
    obs_space = dmc_env.observation_space
    action_space = dmc_env.action_space
    obs_type = dmc_env.obs_type if hasattr(dmc_env, 'obs_type') else 'proprio'

    # Determine input size for encoder
    if isinstance(obs_space, gym.spaces.Box):
        input_size = obs_space.shape
        img_pixels = np.prod(input_size[1:])
        obs_size_flat = np.prod(input_size)
    else:
        raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

    hidden_layer_size = 256
    encoded_obs_size = 1024

    sizes = dict(
        state=state_size,
        hidden=hidden_size,
        encoded_obs=encoded_obs_size,
        action=action_space.shape[0],
        reward=1,
    )

    obs_encoder = nets.ConvEncoder(input_size, encoded_obs_size, base_depth)
    
    encoded_obs_to_hidden_model = nn.Linear(encoded_obs_size, hidden_size)

    det_state_model = nets.DeterministicStateModel(hidden_size, state_size, action_space.shape[0])
    stoch_state_model = nets.StochasticMLPLearnableStd(
        input_dim=hidden_size,
        output_dim=state_size,
        layer_num=2,
        layer_size=hidden_layer_size
    )
    update_state_model = nets.StochasticMLPLearnableStd(
        input_dim=hidden_size + encoded_obs_size,
        output_dim=state_size,
        layer_num=2,
        layer_size=hidden_layer_size
    )
    
    obs_decoder = nets.Cat(nets.ConvDecoder(
        input_size=hidden_size + state_size,
        output_size=input_size,
        base_depth=base_depth
    ), keys=['hidden', 'state'], target_keys=['obs'])
    
    if uncertainty_predictor is None:
        uncertainty_predictor = nets.UncertaintyPredictor()
    else:
        uncertainty_predictor = uncertainty_predictor(sizes=sizes)
    
    if dynamic_uncertainty_model is None:
        dynamic_uncertainty_model = nets.RewardEnsemble(
            sizes=sizes,
            layer_num=2,
            layer_size=hidden_layer_size,
            ensemble_size=5,
        )
    else:
        dynamic_uncertainty_model = dynamic_uncertainty_model(sizes=sizes)
    
    reward_model = nets.RewardModelWithUncertainty(
        reward_model=nets.Cat(nets.StochasticMLPFixedStd(
            input_dim=hidden_size + state_size,
            output_dim=1,
            layer_num=2,
            layer_size=hidden_layer_size,
            activation=nn.ELU(inplace=True)
        ), keys=['hidden', 'state'], target_keys=['reward']) if 0 else nets.Cat(nets.MLP(
            input_dim=hidden_size + state_size,
            output_dim=1,
            layer_num=2,
            layer_size=hidden_layer_size,
            activation=nn.ELU(inplace=True)
        ), keys=['hidden', 'state'], target_keys=['reward']),
        uncertainty_predictor=uncertainty_predictor,
        uncertainty_scale=uncertainty_scale,
        dynamic_factor=dynamic_factor,
        dynamic_uncertainty_model=dynamic_uncertainty_model,
    )
    # Create RSSM environment
    rssm_env = RSSMEnv(
        obs_space=obs_space,
        action_space=action_space,
        obs_encoder=obs_encoder,
        encoded_obs_to_hidden_model=encoded_obs_to_hidden_model,
        det_state_model=det_state_model,
        stoch_state_model=stoch_state_model,
        update_state_model=update_state_model,
        obs_decoder=obs_decoder,
        reward_model=reward_model,
        obs_type=obs_type
    )

    return rssm_env



class VRKNEnv(Env):
    pass
