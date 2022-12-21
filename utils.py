import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size):

        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = torch.zeros((max_size, state_dim))
        self.action = torch.zeros((max_size, action_dim))
        self.next_state = torch.zeros(max_size, state_dim)
        self.reward = torch.zeros((max_size, 1))
        self.not_done = torch.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):

        ind = torch.randint(0, self.size, size=batch_size)

        return (self.state[ind], self.action[ind], self.next_state[ind], self.reward[ind], self.not_done[ind])