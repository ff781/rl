import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
import trainer
import actor as actor_module




class SACCriticObjective(nn.Module):
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

    def forward(self, batch):
        with trainer.freeze_params(self.actor, self.env):
            critic_loss = 0
            for batch_type in ['real', 'dream']:
                if batch.get(batch_type) is not None:
                    critic_loss += self.critic.loss(batch[batch_type])

        return critic_loss, dict(
            loss=critic_loss.item(),
        )



class COMBOObjective(nn.Module):
    def __init__(
        self,
        actor,
        critic,
        env,
        scales=None,
        temperature=1.0,
    ):
        super().__init__()
        self.actor = actor
        self.uniform_actor = actor_module.DummyActor(actor.state_space, env.action_space)
        self.critic = critic
        self.env = env
        self.temperature = temperature
        if scales is None:
            scales = {
                'critic': 1.0,
                'conservative': 1.0,
            }
        self.scales = scales

    def calc_pi_values(self, obs, next_obs):
        with torch.no_grad():
            policy_actions = self.actor(obs)['action']
        q1_pi = self.critic.q1(obs, policy_actions)
        q2_pi = self.critic.q2(obs, policy_actions)
        return q1_pi, q2_pi

    def calc_random_values(self, obs, actions):
        random_q1 = self.critic.q1(obs, actions)
        random_q2 = self.critic.q2(obs, actions)
        return random_q1, random_q2

    def forward(self, batch):

        with trainer.freeze_params(self.actor, self.env):
            # Compute TD error
            td_error = 0
            for batch_type in ['real', 'dream']:
                if batch.get(batch_type) is not None:
                    td_error += self.critic.loss(batch[batch_type])

            # Compute conservative loss
            conservative_loss = 0
            if batch.get('dream') is not None:
                dream_batch = batch['dream'].flatten()
                
                # markov_state = self.env.encode_observations(dream_batch['observation'])
                actor_actions = self.actor(markov_state)['action']
                batch = TensorDict(dict(
                    **dream_batch,
                    **self.env.initial_markov_state(batch_size=dream_batch.shape[0]),
                    action=actor_actions,
                ), batch_size=dream_batch.shape)

                print(f"{dream_batch=}")

                conservative_loss += self.critic.cql_loss(batch, temperature=self.temperature)
            
            loss = (
                self.scales['critic'] * td_error
                + self.scales['conservative'] * conservative_loss
            )

        return loss, dict(
            loss=loss.item(),
            td_error=td_error.item(),
            conservative=conservative_loss.item(),
        )