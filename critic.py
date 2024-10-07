import torch
import torch.nn as nn
import numpy as np
import nets



def compute_returns(rewards, values, discount=0.99, lammda=0.95):
    return compute_returns0(rewards=rewards, values=values, discount=discount, lammda=lammda), 0

def compute_returns0(rewards, values, discount=0.99, lammda=0.95):
    next_values = values[1:]
    # values = values[:-1]
    deltas = rewards[:-1] + discount * next_values * (1 - lammda)
    returns = torch.ones_like(deltas)
    last = 0
    for t in reversed(range(returns.shape[0])):
        returns[t] = last = deltas[t] + (discount * lammda * last)
    return returns


class Critic(nn.Module):
    def __init__(self, state_space, action_space, discount_factor=0.99, lammda=0.95):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.lammda = lammda

    def is_v_critic(self):
        return isinstance(self, EnsembleVCritic)

    def actor_values(self, batch, discount=None, lammda=None):
        discount = discount or self.discount_factor
        lammda = lammda or self.lammda
        values = self(batch)  # Shape: (S, B)
        if self.is_v_critic():
            rewards = batch['reward']
            returns, _ = compute_returns(rewards, values, discount, lammda)
        else:
            # Remain unchanged for Q-functions
            returns = values[:-1]
        return returns

    def critic_values(self, batch, discount=None, lammda=None):
        discount = discount or self.discount_factor
        lammda = lammda or self.lammda
        values = self(batch)  # Shape: (S, B)
        if self.is_v_critic():
            rewards = batch['reward']
            returns, _ = compute_returns(rewards, values, discount, lammda)
            return returns
        else:
            # Compute targets for Q-functions
            next_values = values[1:]
            targets = batch['reward'][:-1] + discount * next_values
            return targets


class EnsembleVCritic(Critic):
    def __init__(self, state_space, action_space, hidden_size=256, layer_num=2, ensemble_size=2):
        super().__init__(state_space, action_space)
        self.v_net = nets.Cat(
            nets.Ensemble(
                module_f=lambda: nets.MLP(
                    input_dim=np.prod(state_space.shape),
                    output_dim=1,
                    layer_num=layer_num,
                    layer_size=hidden_size,
                    activation=nn.ELU(inplace=True)
                ),
                ensemble_size=ensemble_size,
                reduction='min'
            ),
            keys=['hidden', 'state'],
        )

    def forward(self, x):
        return self.v_net(x)
    
    def loss(self, batch, discount=None, lammda=None):
        target = self.critic_values(batch, discount=discount, lammda=lammda)
        return self.v_net.loss(batch[:-1], target)



class EnsembleQCritic(Critic):
    def __init__(self, state_space, action_space, hidden_size=256, layer_num=2, ensemble_size=2):
        super().__init__(state_space, action_space, hidden_size, layer_num)
        self.q_net = nets.Cat(
            nets.Ensemble(
                module_f=lambda:nets.MLP(
                    input_dim=np.prod(state_space.shape) + np.prod(action_space.shape),
                    output_dim=1,
                    layer_num=layer_num,
                    layer_size=hidden_size,
                    activation=nn.ELU(inplace=True)
                ),
                ensemble_size=ensemble_size,
                reduction='min',
            ),
            keys=['hidden', 'state', 'action'],
        )
    
    def forward(self, x):
        return self.q_net(x)
    
    def loss(self, batch, discount=None, lammda=None):
        target = self.critic_values(batch, discount=discount, lammda=lammda)
        return self.q_net.loss(batch[:-1], target)

    def cql_loss(self, real_batch, dream_batch, temperature=1.0):

        pi0 = self()
        pi1_batch = TensorDict()
        
        conservative_loss1 = torch.logsumexp(cat_q1 / self.temperature, dim=1).mean() * self.beta * self.temperature - q1.mean() * self.beta
        conservative_loss2 = torch.logsumexp(cat_q2 / self.temperature, dim=1).mean() * self.beta * self.temperature - q2.mean() * self.beta

        obss, actions, next_obss = fake_batch["observation"], \
            fake_batch["action"], fake_batch["next_observation"]
        
        batch_size = len(obss)
        random_actions = torch.FloatTensor(
            batch_size * self.num_repeat_actions, actions.shape[-1]
        ).uniform_(self.env.action_space.low[0], self.env.action_space.high[0]).to(self.actor.device)
        
        tmp_obss = obss.unsqueeze(1).repeat(1, self.num_repeat_actions, 1).view(batch_size * self.num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1).repeat(1, self.num_repeat_actions, 1).view(batch_size * self.num_repeat_actions, obss.shape[-1])
        
        obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss)
        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss)
        random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)

        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)
        return torch.logsumexp(cat_q1 / temperature, dim=1).mean() * cql_weight - self(batch[:-1]).mean() * cql_weight


class DoubleQCritic(EnsembleQCritic):
    def __init__(self, state_space, action_space, hidden_size=256, layer_num=2):
        super().__init__(state_space, action_space, hidden_size, layer_num, ensemble_size=2)


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    # Example reward and values sequence from a dmc task
    rewards = torch.tensor([0.5, 0.7, 0.2, 0.9, 1.1, 0.4, 0.6, 0.8, 1.0, 0.3, 0.5, 0.7, 0.9, 1.2, 0.6, 0.4, 0.8, 1.0, 0.9, 0.7])
    values = torch.tensor([4.5, 4.2, 3.8, 3.5, 3.2, 3.0, 2.8, 2.5, 2.2, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0])
    gamma = 0.99
    lammda = 0.95

    # Compute GAE
    returns, advantages = compute_gae_wrong(rewards, values, gamma, lammda)
    
    # Compute GAE0
    returns0 = compute_returns0(rewards, values, gamma, lammda)

    # Plot the results
    plt.plot(rewards, label='Rewards')
    plt.plot(values, label='Values')
    plt.plot(returns, label='Returns')
    plt.plot(advantages, label='Advantages')
    plt.plot(returns0, label='Returns0')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Rewards, Values, Returns, and Advantages')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
