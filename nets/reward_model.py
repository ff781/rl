import torch
from torch import nn
from tensordict import TensorDict
import meth
import nets




class RewardModelWithUncertainty(nn.Module):
    def __init__(self, reward_model, uncertainty_predictor, uncertainty_scale=None, dynamic_factor=None, dynamic_uncertainty_model=None):
        super().__init__()
        self.reward_model = reward_model
        self.uncertainty_predictor = uncertainty_predictor or UncertaintyPredictor()

        self.uncertainty_scale = uncertainty_scale or 0
        self.dynamic_factor = dynamic_factor
        if not self.dynamic_factor:
            dynamic_uncertainty_model = None
        elif dynamic_uncertainty_model is None:
            dynamic_uncertainty_model = self.uncertainty_predictor
        self.dynamic_uncertainty_model = dynamic_uncertainty_model
        self.running_uncertainty = meth.M(sum=None, track_variance=True)
        self.running_reward = meth.M(sum=None, track_variance=True)
        self.warmup_steps = 100
        self.step_count = 0

    def forward(self, x):
        reward_unpenalized = self.reward_model(x)
        uncertainty = self.uncertainty_predictor(x).to(reward_unpenalized)
        uncertainty_reduced = uncertainty.mean(dim=-1, keepdim=True)
        uncertainty_scale = self.get_uncertainty_scale(x)
        reward = reward_unpenalized - uncertainty_scale * uncertainty_reduced
        
        return TensorDict(
            dict(
                reward=reward,
                reward_unpenalized=reward_unpenalized,
            ),
            batch_size=x.shape
        ), dict(
            uncertainty=uncertainty.mean().item(),
            uncertainty_scale=float(uncertainty_scale),
            reward=reward.mean().item(),
            reward_unpenalized=reward_unpenalized.mean().item(),
        )

    def get_uncertainty_scale(self, x):
        if self.dynamic_factor:
            with torch.no_grad():
                _ = self.running_uncertainty.std().mean()
                if _ == 0 or self.step_count < self.warmup_steps:
                    return 1.0
                return self.dynamic_factor * self.running_reward.std().mean() / _
        return self.uncertainty_scale

    def loss(self, x, targets):
        reward_loss = self.reward_model.loss(x, targets)
        uncertainty_loss = self.uncertainty_predictor.loss(x, targets).to(reward_loss)
        dynamic_uncertainty_loss = self.dynamic_uncertainty_model.loss(x, targets) if self.dynamic_uncertainty_model else torch.tensor(0.0).to(reward_loss)
        return torch.stack([reward_loss, uncertainty_loss, dynamic_uncertainty_loss]).sum(), dict(
            reward_loss=reward_loss.item(),
            uncertainty_loss=uncertainty_loss.item(),
            dynamic_uncertainty_loss=dynamic_uncertainty_loss.item(),
        )

    def update(self, x: TensorDict):
        if self.dynamic_factor:
            with torch.no_grad():
                uncertainty = self.dynamic_uncertainty_model(x)
                uncertainty = uncertainty.view(-1, uncertainty.size(-1))
                self.running_uncertainty.add_batch(uncertainty)
                reward = x['reward'].view(-1, x['reward'].size(-1))
                self.running_reward.add_batch(reward)
                self.step_count += 1
                return dict(
                    running_uncertainty_std=self.running_uncertainty.std().mean().item(),
                    running_reward_std=self.running_reward.std().mean().item(),
                )
        return {}

class UncertaintyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.zeros(x.shape).unsqueeze(-1)

    def loss(self, x, targets):
        return torch.zeros([], requires_grad=True)

class UncertaintyEnsemble(nets.Ensemble, UncertaintyPredictor):
    def __init__(self, module_f, ensemble_size):
        UncertaintyPredictor.__init__(self)
        nets.Ensemble.__init__(self, module_f=module_f, ensemble_size=ensemble_size, reduction='var')

class RewardEnsemble(UncertaintyEnsemble):
    def __init__(self, sizes, layer_num, layer_size, ensemble_size):
        module_f = lambda:nets.Cat(nets.MLP(
            input_dim=sizes['hidden'] + sizes['state'],
            output_dim=sizes['reward'],
            layer_num=layer_num,
            layer_size=layer_size
        ), keys=['hidden', 'state'], target_keys=['reward'])
        super().__init__(module_f, ensemble_size)

    # def loss(self, x, targets):
    #     return super().loss(x, targets['reward'])

class DecodeToEmbeddedEnsemble(UncertaintyEnsemble):
    def __init__(self, sizes, layer_num, layer_size, ensemble_size):
        module_f = lambda:nets.Cat(nets.MLP(
            input_dim=sizes['hidden'] + sizes['state'],
            output_dim=sizes['encoded_obs'],
            layer_num=layer_num,
            layer_size=layer_size
        ), keys=['hidden', 'state'], target_keys=['encoded_obs'])
        super().__init__(module_f, ensemble_size)

    # def loss(self, x, targets):
    #     return super().loss(x, targets['observation'].flatten(start_dim=x.ndim-1))