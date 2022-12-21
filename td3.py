import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._C._onnx as onnx_C

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.layer1 = nn.Linear(state_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.layer1(state))
        a = self.max_action * torch.tanh(a)

        return a

class Critic(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer1 = nn.Linear(state_dim + action_dim, 4)
        self.layer2 = nn.Linear(4, 1)

        self.layer3 = nn.Linear(state_dim + action_dim, 4)
        self.layer4 = nn.Linear(4, 1)

        self.drop = nn.Dropout()

    def forward(self, state, action):

        sa = torch.cat([state, action], 1)


        
        q1 = F.relu(self.layer1(sa))
        q1 = self.drop(q1)
        q1 = F.relu(self.layer2(q1))
        
        q2 = F.dropout(F.relu(self.layer1(sa)))
        q2 = F.relu(self.layer2(q2))

        return q1, q2


class TD3:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        self.actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))

        return self.actor(state)

    def train(self, replay_buffer, batch_size=356):

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = reward + not_done * self.discount * target_Q


        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

state = torch.rand(1, 3)
action = torch.rand(1, 2)

Amodel = Actor(state.shape[1], 2, 2)
Cmodel = Critic(state.shape[1], 2)


torch.onnx.export(Amodel, (state), "Amodel.onnx", input_names=["state"], output_names=["action"])
torch.onnx.export(Cmodel, (state, action), "Cmodel.onnx", input_names=["state", "action"], output_names=["Q1", "Q2"], training=onnx_C.TrainingMode.TRAINING)

