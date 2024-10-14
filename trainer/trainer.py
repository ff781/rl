import copy
import time
import torch
from torch import nn
from torch.optim import Optimizer
from tensordict import TensorDict
import contextlib
import numpy as np
import random
import meth

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@contextlib.contextmanager
def freeze_params(*modules):
    original_states = []
    for module in modules:
        original_states.append([param.requires_grad for param in module.parameters()])
        for param in module.parameters():
            param.requires_grad = False
    try:
        yield
    finally:
        for module, original_state in zip(modules, original_states):
            for param, requires_grad in zip(module.parameters(), original_state):
                param.requires_grad = requires_grad


class AlternatingScheduler:

    def __call__(self, epochs, epoch, batches, batch, iterations, iteration):
        return 'model' if iteration % 2 == 0 else 'policy'

class ModelThenPolicyScheduler:
    def __init__(self, f=.5):
        self.f = f

    def __call__(self, epochs, epoch, batches, batch, iterations, iteration):
        if iteration < iterations * self.f:
            return 'model'
        else:
            return 'policy'





class MBTrainer(nn.Module):
    def __init__(self, env, model, actor, critic, replay, scheduler, optimizers, objectives, logger):
        super().__init__()
        self.env = env
        self.model = model
        self.actor = actor
        self.critic = critic
        self.replay = replay
        self.scheduler = scheduler
        self.optimizers = optimizers
        self.objectives = objectives
        self.i_epoch = 0
        self.logger = logger
        self.best = dict()

    def eval(self, epochs=1):
        returns = []
        device = next(self.model.parameters()).device
        for _ in range(epochs):
            total_reward = 0
            obs, _ = self.env.reset()
            obs = torch.as_tensor(obs.copy(), device=device)
            self.model.reset()
            done = False
            while not done:
                with torch.no_grad():
                    markov_state = self.model.markov_state()
                    action = self.actor(markov_state)['action'].cpu().numpy()
                    self.model.step(torch.as_tensor(action.copy(), device=device), observation=obs.unsqueeze(0))
                obs, reward, done, _, _ = self.env.step(action)
                obs = torch.as_tensor(obs.copy(), device=device)
                total_reward += reward
            returns.append(total_reward)
        return torch.tensor(returns, device=device)

    def train(self, epochs, batches, bs, sl, eval_epochs, new_best_model_cb=None, profiler=None):
        best_eval_return = float('-inf')
        best = self.best

        for i_epoch in range(epochs):
            start_time = time.time()

            log_dicts = []
            mean_log_dict = {}
            for i_batch, batch in enumerate(self.replay.get_dataloader(batches, bs, sl, None)):
                mode = self.scheduler(
                    epochs=epochs,
                    epoch=i_epoch,
                    batches=batches,
                    batch=i_batch,
                    iterations=epochs * batches,
                    iteration=i_epoch * batches + i_batch,
                )
                batch = self.get_batch(batch, bs, sl, mode)
                if mode == 'model':
                    log_dict = self.train_model(batch)
                elif mode == 'policy':
                    log_dict = self.train_policy(batch)
                
                with torch.no_grad():
                    if i_batch == 0:
                        for key, value in log_dict.items():
                            if not isinstance(value, float):
                                quantiles = meth.quantile(value.detach().cpu(), torch.tensor([0, 0.25, 0.5, 0.75, 1.0]))
                                mean_log_dict[key] = quantiles
                    log_dict = {k: v for k, v in log_dict.items() if isinstance(v, float)}
                    log_dicts.append(log_dict)

            with torch.no_grad():
                # Calculate mean of log_dicts
                all_keys = set().union(*log_dicts)
                for key in all_keys:
                    values = [d[key] for d in log_dicts if key in d]
                    if values:
                        if all(isinstance(v, float) for v in values):
                            mean_log_dict[key] = sum(values) / len(values)

                # Run evaluation every eval_epochs
                if i_epoch % eval_epochs == 0:
                    eval_returns = self.eval()
                    mean_eval_return = eval_returns.mean().item()
                    mean_log_dict['eval/return'] = mean_eval_return

                    # Check if this is the best evaluation so far
                    if mean_eval_return > best_eval_return:
                        best_eval_return = mean_eval_return
                        best['model'] = copy.deepcopy(self.model)
                        best['actor'] = copy.deepcopy(self.actor)
                        best['critic'] = copy.deepcopy(self.critic)
                        best['env'] = dict(
                            env_name=self.env.env_name,
                            obs_type=self.env.obs_type,
                            img_size=self.env.img_size,
                            action_repeat=self.env.action_repeat,
                        )
                        print(f"New best model found at epoch {i_epoch} with return {best_eval_return:.2g}")
                        # if new_best_model_cb:
                        #     mean_log_dict.update({f"eval/{k}": v for k, v in new_best_model_cb().items()})
                    if new_best_model_cb:
                        mean_log_dict.update({f"eval/{k}": v for k, v in new_best_model_cb().items()})

                # Add epoch time to log_dict
                epoch_time = time.time() - start_time
                mean_log_dict['time/epoch'] = epoch_time

                mean_log_dict = dict(sorted(mean_log_dict.items()))

                self.logger.log(mean_log_dict, step=self.i_epoch, log_quantiles=self.i_epoch%50==0)
                self.i_epoch += 1

                if profiler:
                    profiler.step()

        print(f"Final best eval return: {best_eval_return:.2g}")
        return best_eval_return

    def get_batch(self, batch, bs, sl, mode):
        if mode == 'policy':
            return dict(
                real=batch,
                dream=None,
            )
        return batch

    def train_model(self, batch):
        loss, log_dict = self.objectives['model'](batch)
        log_dict = {f"model/{k}": v for k, v in log_dict.items()}
        self.optimizers['model'].zero_grad()
        loss.backward()
        
        # Calculate and log the gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
        log_dict['model/grad_norm'] = grad_norm.item()
        
        self.optimizers['model'].step()

        return log_dict

    def train_policy(self, batch):
        log_dict = {}

        actor_loss, _ = self.objectives['actor'](batch)
        log_dict |= {f"actor/{k}": v for k, v in _.items()}
        self.optimizers['actor'].zero_grad()
        actor_loss.backward()
        
        # Calculate and log the gradient norm for actor
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'))
        log_dict['actor/grad_norm'] = actor_grad_norm.item()
        
        self.optimizers['actor'].step()

        critic_loss, _ = self.objectives['critic'](batch)
        log_dict |= {f"critic/{k}": v for k, v in _.items()}
        self.optimizers['critic'].zero_grad()
        critic_loss.backward()
        
        # Calculate and log the gradient norm for critic
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float('inf'))
        log_dict['critic/grad_norm'] = critic_grad_norm.item()
        
        self.optimizers['critic'].step()

        return log_dict


class DreamTrainer(MBTrainer):
    def __init__(
            self, 
            env, 
            model, 
            actor, 
            critic, 
            replay, 
            scheduler, 
            optimizers, 
            objectives, 
            horizon, 
            logger, 
            variable_horizon=False, 
            max_horizon=69, 
            dream_ratio=1.0
        ):
        super().__init__(env, model, actor, critic, replay, scheduler, optimizers, objectives, logger)
        self.horizon = horizon
        self.variable_horizon = variable_horizon
        self.max_horizon = max_horizon if max_horizon is not None else horizon
        self.dream_ratio = dream_ratio

    def get_batch(self, batch, bs, sl, mode):

        if mode == 'policy':

            with freeze_params(self.model, self.actor, self.critic):
                total_samples = batch.shape[0] * batch.shape[1]
                dream_batch_size = int(batch.shape[0] * self.dream_ratio / (1 - self.dream_ratio)) if self.dream_ratio < 1 else total_samples
                
                # Randomly select dream_batch_size samples along the first dimension
                collapsed_batch = batch.reshape(total_samples, *batch.shape[2:])
                # collapsed_batch = collapsed_batch[torch.randperm(collapsed_batch.shape[0])[:dream_batch_size]]
                
                # Determine the simulation steps
                if self.variable_horizon:
                    steps = min(int(random.expovariate(1 / self.horizon)) + 1, self.max_horizon)
                else:
                    steps = self.horizon

                # Simulate the batch over the horizon
                actions = lambda state: self.actor(state)['action']
                simulated_batch = self.model.simulate(
                    actions=actions,
                    steps=steps,
                    batch_size=collapsed_batch.shape,
                )
                dream = simulated_batch

                real = None
                if self.dream_ratio < 1:
                    real = batch
                    markov_states = self.env.initial_markov_state(batch_size=real.shape)
                    real['state'] = markov_states['state']
                    real['hidden'] = markov_states['hidden']

            return dict(
                real=real,
                dream=dream,
            )
        
        return batch

