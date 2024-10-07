from typing import Mapping
import torch
import torch.nn as nn
import torch.distributions as dists
import gymnasium as gym
from tensordict import TensorDict
import trainer



def gaussian_rssm_kl_loss(lhs_mean, lhs_std, rhs_mean, rhs_std, free_nats=2.0, balanced=True, alpha=0.8):
    def kl(lhs_mean, lhs_std, rhs_mean, rhs_std):
        # Compute KL divergence between two Gaussian distributions
        # KL(p||q) = log(σ_q/σ_p) + (σ_p^2 + (μ_p - μ_q)^2) / (2σ_q^2) - 1/2
        r = dists.kl_divergence(
            dists.Normal(loc=lhs_mean, scale=lhs_std),
            dists.Normal(loc=rhs_mean, scale=rhs_std)
        )
        return r.sum(dim=-1)
    
    if balanced:
        # Compute balanced KL divergence
        # This helps stabilize training by balancing gradients between encoder and decoder
        lhs_kl = kl(lhs_mean.detach(), lhs_std.detach(), rhs_mean, rhs_std)
        rhs_kl = kl(lhs_mean, lhs_std, rhs_mean.detach(), rhs_std.detach())

        # Weighted sum of forward and reverse KL divergences
        # KL_balanced = α * KL(post||prior) + (1-α) * KL(prior||post)
        kl_loss = alpha * lhs_kl.clamp(min=free_nats).mean() + (1 - alpha) * rhs_kl.clamp(min=free_nats).mean()
    else:
        # Compute standard KL divergence
        lhs_kl = kl(lhs_mean, lhs_std, rhs_mean, rhs_std)
        kl_loss = lhs_kl.clamp(min=free_nats).mean()
    
    # Free nats: Allows the model to ignore small differences in distributions
    # max(KL, free_nats)
    
    with torch.no_grad():
        log_dict = dict(kl=lhs_kl.mean().item())

    return kl_loss, log_dict


class RSSMEnvObjective(nn.Module):
    def __init__(
        self,
        env,
        actor,
        critic,
        scales=None,
    ):
        super().__init__()
        self.rssm_env = env
        self.actor = actor
        self.critic = critic
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        if scales is None:
            scales = dict(
                reward=1.0,
                kl=1.0,
                reconstruction=1.0,
                regularization=1.0,
            )
        self.scales = scales

    i = 0
    def forward(self, batch):
        batch = batch.flatten()
        actions = batch['action']
        batch['obs'] = self.rssm_env.preprocess_obs(batch['observation'])

        # markov_state = self.rssm_env.encode_observations(
        #     observations=batch['observation']
        # )

        model_batch = self.rssm_env.simulate(
            actions=actions.unsqueeze(0),
            # initial_state=markov_state['state'],
            # initial_hidden=markov_state['hidden'],
            observations=batch['observation'].unsqueeze(0),
        ).squeeze(0)

        model_batch_without_last = model_batch[0]

        if self.i % 50 == 0:
            1
            # import text;print(text.std(dict(model=model_batch_without_last), collapse_arrays='b'))
        self.i += 1

        combined_batch = TensorDict({
            **batch,
            'encoded_obs': model_batch_without_last['encoded_obs']
        }, batch_size=batch.batch_size)
        reward_loss = self.rssm_env.reward_model.loss(model_batch_without_last, combined_batch)
        reward_running_log_dict =self.rssm_env.reward_model.update(model_batch_without_last)

        kl_loss, kl_log_dict = gaussian_rssm_kl_loss(
            lhs_mean=model_batch_without_last["state"]["mean"],
            lhs_std=model_batch_without_last["state"]["std"],
            rhs_mean=model_batch_without_last["prior_state"]["mean"],
            rhs_std=model_batch_without_last["prior_state"]["std"],
            free_nats=3.0,
            balanced=True,
            alpha=0.8
        )

        reconstruction_loss = self.rssm_env.obs_decoder.loss(model_batch_without_last, batch)

        regularization_loss = torch.zeros(1).to(kl_loss)

        loss = (
            self.scales["reward"] * reward_loss +
            self.scales["kl"] * kl_loss +
            self.scales["reconstruction"] * reconstruction_loss +
            self.scales["regularization"] * regularization_loss
        )

        with torch.no_grad():
            reward_mse = ((model_batch_without_last['reward'] - batch['reward']) ** 2).mean()
            reconstruction_mse = ((model_batch_without_last['obs'] - batch['obs']) ** 2).mean()

            x = model_batch_without_last
            log_dict = {
                "loss": loss.item(),
                "reward_loss": reward_loss.item(),
                "kl_loss": kl_loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "regularization_loss": regularization_loss.item(),
                **{f"{k}": v for k, v in reward_running_log_dict.items()},
                **{f"{k}": v for k, v in kl_log_dict.items()},
                "reward_mse": reward_mse.item(),
                "reconstruction_mse": reconstruction_mse.item(),
                **{f"model_batch/{k}": v for k, v in TensorDict({k: x[k]['sample'] if isinstance(x[k], Mapping) else x[k] for k in x.keys()}, batch_size=x.batch_size).items()},
            }

        return loss, log_dict