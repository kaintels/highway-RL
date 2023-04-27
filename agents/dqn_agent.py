import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from collections import deque
from models.dqn import DQN


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, eps_end, eps_start, eps_decay, batch_size, gamma) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma

        self.model = DQN(self.state_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.steps_done = 0
        self.memory = deque(maxlen=10000)

    def memorize(self, state, action, reward, next_state):
        self.memory.append(
            (
                state,
                action,
                torch.FloatTensor([reward]),
                torch.FloatTensor(next_state).reshape(-1, 5, 5),
            )
        )

    def action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if random.random() > eps_threshold:
            return self.model(state).data.max(-1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(5)]])

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        current_q = self.model(states).gather(1, actions)
        max_next_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (self.gamma * max_next_q)
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
