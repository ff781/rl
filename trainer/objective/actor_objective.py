import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
from tensordict import TensorDict
import trainer



class SACActorObjective(nn.Module):
    def __init__(
        self,
        actor,
        critic,
        env,
        scales=None,
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.env = env
        if scales is None:
            scales = dict(
                actor=1.0,
                entropy=1.0,
            )
        self.scales = scales

    def forward(self, batch):

        with trainer.freeze_params(self.critic, self.env):
            actor_loss = 0
            entropy_loss = 0

            for batch_type in ['real', 'dream']:

                if batch.get(batch_type) is not None:

                    states = batch[batch_type]['state']["sample"]
                    action_log_probs = self.actor(batch[batch_type])
                    actions, log_probs = action_log_probs['action'], action_log_probs['log_prob']

                    acted_batch = batch[batch_type].select('hidden', 'state')
                    acted_batch['action'] = actions
                    acted_batch['reward'], reward_model_log_dict = self.env.reward_model(acted_batch)
                    acted_batch['reward'] = acted_batch['reward']['reward']
                    
                    target_values = self.critic.actor_values(acted_batch)
                    
                    # L(π) = 𝔼[-Q(s,a) + α * log π(a|s)]
                    actor_loss += -target_values.mean()
                    entropy_loss += log_probs.mean()

            # Total loss
            loss = (
                self.scales["actor"] * actor_loss + 
                self.scales["entropy"] * entropy_loss
            )

        with torch.no_grad():
            log_dict = dict(
                loss=loss.item(),
                action=actor_loss.item(),
                entropy=entropy_loss.item(),
                **{f"reward/{k}": v for k,v in reward_model_log_dict.items()},
            )
            for batch_type, sub_batch in batch.items():
                if sub_batch is not None:
                    from typing import Mapping
                    
                    flattened_batch = TensorDict({
                        k: sub_batch[k]['sample'] if isinstance(sub_batch[k], Mapping) else sub_batch[k]
                        for k in sub_batch.keys()
                    }, batch_size=sub_batch.batch_size).flatten()
                    
                    first_element = {k: v[0] for k, v in flattened_batch.items()}
                    
                    log_dict.update({
                        f"{batch_type}/{k}": v 
                        for k, v in first_element.items()
                    })

        return loss, log_dict
