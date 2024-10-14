from typing import Mapping
import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDict
import math



class softplus_activation(nn.Module):
    def __init__(self, init_val, min_val):
        super().__init__()
        self.shift = math.log(math.expm1(init_val - min_val))
        self.min_val = min_val

    def forward(self, x):
        return torch.nn.functional.softplus(x + self.shift) + self.min_val



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer_num, layer_size, activation=nn.ReLU(inplace=True)):
        super().__init__()
        act_fn = activation if callable(activation) else getattr(nn, activation.pop('_'))(**activation)
        
        layers = [nn.Linear(input_dim, layer_size), act_fn]
        for _ in range(layer_num - 1):
            layers.extend([nn.Linear(layer_size, layer_size), act_fn])
        layers.append(nn.Linear(layer_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def loss(self, x, targets):
        predictions = self.forward(x)
        return nn.functional.mse_loss(predictions, targets)

class StochasticMLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer_num, layer_size, activation=nn.ELU(alpha=1.0, inplace=True)):
        super().__init__()
        act_fn = activation if callable(activation) else getattr(nn, activation.pop('_'))(**activation)
        
        if layer_num > 0:
            layers = [nn.Linear(input_dim, layer_size), act_fn]
            for _ in range(layer_num - 1):
                layers.extend([nn.Linear(layer_size, layer_size), act_fn])
            self.net = nn.Sequential(*layers)
            self.mean = nn.Linear(layer_size, output_dim)
        else:
            self.net = nn.Identity()
            self.mean = nn.Linear(input_dim, output_dim)

    def forward_dist(self, x):
        x = self.net(x)
        mean = self.mean(x)
        std = self.get_std(x, mean)
        return mean, std
    
    def get_std(self, x, mean):
        return torch.zeros_like(mean)

    def forward(self, x):
        return self.forward_all(x)["sample"]

    def forward_all(self, x):
        mean, std = self.forward_dist(x)
        dist = Normal(mean, std)
        sample = dist.rsample()
        return TensorDict(dict(mean=mean, std=std, sample=sample), batch_size=x.shape[:-1])
    
    def forward_sample(self, x):
        return self.forward_all(x)['sample']

    def loss(self, inputs, targets, mask=None):
        outputs = self.forward_all(inputs)
        dist = Normal(outputs['mean'], outputs['std'])
        sample_wise_ll = dist.log_prob(targets).sum(dim=-1)
        
        if mask is None:
            loss = -sample_wise_ll.mean()
        else:
            loss = -(sample_wise_ll * mask).sum() / mask.sum()
        
        return loss

    def get_std(self, mean):
        raise NotImplementedError("Subclasses must implement this method")


class StochasticMLPLearnableStd(StochasticMLP):
    def __init__(self, input_dim, output_dim, layer_num, layer_size, min_std=0.1, activation=nn.ELU(alpha=1.0, inplace=True)):
        super().__init__(input_dim, output_dim, layer_num, layer_size, activation)
        if layer_num > 0:
            self.std = nn.Linear(layer_size, output_dim)
        else:
            self.std = nn.Linear(input_dim, output_dim)
        self.std_activation = softplus_activation(init_val=1.0, min_val=min_std)
        self.min_std = min_std

    def get_std(self, x, mean):
        raw_std = self.std(x)
        return self.std_activation(raw_std)


class StochasticMLPFixedStd(StochasticMLP):

    def __init__(self, input_dim, output_dim, layer_num, layer_size, activation=nn.ELU(alpha=1.0, inplace=True), std=1.0):
        super().__init__(input_dim, output_dim, layer_num, layer_size, activation)
        self.std = std

    def get_std(self, x, mean):
        return torch.full_like(mean, fill_value=self.std)
    
    def loss(self, inputs, targets, mask=None):
        outputs = self.forward_all(inputs)
        dist = Normal(outputs['mean'], outputs['std'])
        sample_wise_ll = dist.log_prob(targets).sum(dim=-1)
        
        if mask is None:
            loss = -sample_wise_ll.mean()
        else:
            loss = -(sample_wise_ll * mask).sum() / mask.sum()

        return loss


class ConvEncoder(nn.Module):
    def __init__(self, input_size, output_size, base_depth, activation=nn.ReLU(inplace=True)):
        super(ConvEncoder, self).__init__()
        
        in_channels, height, width = input_size
        img_size = height
        
        act_fn = activation if callable(activation) else getattr(nn, activation.pop('_'))(**activation)
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.Flatten(),
        )
        
    def forward(self, x):
        # Input shape: (..., in_channels, height, width)
        # Output shape: (..., output_size)
        original_shape = x.shape[:-3]
        x = x.view(-1, *x.shape[-3:])
        x = self.net(x)
        return x.view(*original_shape, -1)

class ConvDecoder(nn.Module):
    def __init__(self, input_size, output_size, base_depth, activation=nn.ReLU(inplace=True)):
        super(ConvDecoder, self).__init__()
        
        out_channels, height, width = output_size
        img_size = height
        
        act_fn = activation if callable(activation) else getattr(nn, activation.pop('_'))(**activation)
        
        # layers = []
        # current_size = 4
        # current_channels = base_depth * (img_size // current_size)
        
        # layers.extend([
        #     nn.Linear(input_size, current_channels * current_size * current_size),
        #     nn.Unflatten(-1, (current_channels, current_size, current_size)),
        # ])
        
        # while current_size < img_size:
        #     next_channels = max(current_channels // 2, out_channels)
        #     layers.extend([
        #         nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1),
        #         act_fn,
        #     ])
        #     current_channels = next_channels
        #     current_size *= 2
        
        # layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=1, padding=1))

        # self.net = nn.Sequential(*layers)
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Unflatten(-1, (1024, 1, 1)),
            nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=out_channels, kernel_size=6, stride=2),
            nn.Tanh(),
        )
        # self.final_layer = StochasticMLPFixedStd(out_channels * img_size * img_size, out_channels * img_size * img_size, 0, None, std=1e-16)

    def forward(self, x):
        # Input shape: (..., input_size)
        # Output shape: (..., out_channels, height, width)
        original_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        x = self.net(x)
        chw_shape = x.shape[len(original_shape):]
        return x
        # x = x.view(*original_shape, -1)
        # x_ = self.final_layer(x)
        # x_.batch_size = x.shape
        # x_ = x_.view(*original_shape, *chw_shape).to_tensordict()  # Change to (batch, channels, height, width)
        # x_.batch_size = x_.batch_size[:1]
        # return x_.view(*original_shape)

    def loss(self, inputs, targets, mask=None):
        original_shape = inputs.shape[:-1]
        outputs = self.forward(inputs)
        outputs = outputs.view(*original_shape, -1)
        targets = targets.view(*original_shape, -1)
        return nn.functional.mse_loss(outputs, targets)
        return self.final_layer.loss(outputs, targets, mask)

class DeterministicStateModel(nn.GRUCell):
    def __init__(self, hidden_size, state_size, action_size):
        super().__init__(state_size + action_size, hidden_size)
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.action_size = action_size

    def forward(self, x):
        hidden = x['hidden']
        state = x['state']
        action = x['action']
        
        # Flatten batch dimensions
        original_shape = hidden.shape[:-1]
        hidden_flat = hidden.view(-1, self.hidden_size)
        state_flat = state.view(-1, self.state_size)
        action_flat = action.view(-1, self.action_size)
        
        combined = torch.cat([state_flat, action_flat], dim=-1)
        new_hidden_flat = super().forward(combined, hidden_flat)
        
        # Restore original batch dimensions
        new_hidden = new_hidden_flat.view(*original_shape, self.hidden_size)
        return new_hidden

class Det(nn.Module):
    def forward(self, x):
        if isinstance(x, Mapping):
            flattened = {}
            self._flatten_tensordict(x, '', flattened)
            return TensorDict(flattened, batch_size=x.batch_size)
        return x

    def _flatten_tensordict(self, td, prefix, result):
        for key, value in td.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, Mapping):
                if 'sample' in value and isinstance(value['sample'], torch.Tensor):
                    result[new_key] = value['sample']
                else:
                    self._flatten_tensordict(value, new_key, result)
            else:
                result[new_key] = value
det = Det()

class Cat(nn.Module):
    def __init__(self, m=lambda a:a, keys=None, target_keys=None):
        super().__init__()
        self.m = m
        self.keys = keys
        self.target_keys = target_keys

    def _cat_tensors(self, x, keys):
        if not isinstance(x, TensorDict):
            return x
        keys = keys or list(x.keys())
        if isinstance(keys, Mapping):
            return TensorDict({k: x[k]['sample'] if isinstance(x[k], Mapping) else x[k] for k in keys}, batch_size=x.batch_size)
        else:
            tensors = [x[k]['sample'] if isinstance(x[k], Mapping) else x[k] for k in keys]
            return torch.cat(tensors, dim=-1)

    def forward(self, x: TensorDict):
        _ = self._cat_tensors(x, self.keys)
        return self.m(_)

    def loss(self, x, target, *args, **kwargs):
        return self.m.loss(self._cat_tensors(x, self.keys), self._cat_tensors(target, self.target_keys), *args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.m, name)
cat = Cat()


class Ensemble(nn.Module):
    def __init__(self, module_f, ensemble_size, reduction='mean'):
        super().__init__()
        self.models = nn.ModuleList([module_f() for _ in range(ensemble_size)])
        self.ensemble_size = ensemble_size
        self.reduction = reduction

    def forward(self, *args, **kwargs):
        outputs = torch.stack([m(*args, **kwargs) for m in self.models], dim=0)
        return self._reduce(outputs)

    def _reduce(self, tensor):
        if self.reduction == 'mean':
            return torch.mean(tensor, dim=0)
        elif self.reduction == 'min':
            return torch.min(tensor, dim=0)[0]
        elif self.reduction == 'max':
            return torch.max(tensor, dim=0)[0]
        elif self.reduction == 'sum':
            return torch.sum(tensor, dim=0)
        elif self.reduction == 'var':
            mean = torch.mean(tensor, dim=0)
            return torch.sum((tensor - mean) ** 2, dim=0)
        else:
            raise ValueError(f"Unknown reduction strategy: {self.reduction}")

    def loss(self, *args, **kwargs):
        losses = torch.stack([m.loss(*args, **kwargs) for m in self.models], dim=0)
        return torch.mean(losses, dim=0)
