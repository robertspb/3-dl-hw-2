import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

from TicTacToeEnv import TicTacToe

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, exptuple):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exptuple
        self.position = (self.position + 1) % self.capacity
       
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self, conv_size=128):
        nn.Module.__init__(self)
        self.c1 = nn.Conv2d(3, conv_size * 2, (3, 3))
        self.l1 = nn.Linear(conv_size * 2, conv_size)
        self.l2 = nn.Linear(conv_size, 9)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class DuelingNetwork(nn.Module):
    def __init__(self, conv_out=128):
        nn.Module.__init__(self)
        self.c1 = nn.Conv2d(3, conv_out * 2, (3, 3))
        self.l1 = nn.Linear(conv_out * 2, conv_out)
        self.v = nn.Linear(conv_out, 1)
        self.a = nn.Linear(conv_out, 9)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.l1(x))
        v = self.v(x)
        a = self.a(x)
        x = v + (a - a.mean(dim=1, keepdim=True))
        return x


class TicTacToeDQN():
    def __init__(self, env, n_rows=3, n_cols=3, n_win=3, model_class=Network, gamma=0.8, device=device):
        self.device = device
        self.rm = 1000000
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.env = env
        self.models = {1: model_class().to(device),
                       -1: model_class().to(device)}
        self.memories = {1: ReplayMemory(self.rm), 
                         -1: ReplayMemory(self.rm)}
        self.optimizers = {1: optim.Adam(self.models[1].parameters(), lr=0.0001, weight_decay=0.001),
                           -1: optim.Adam(self.models[-1].parameters(), lr=0.0001, weight_decay=0.001)}
        self.prev_states = {1: None,
                            -1: None}
        self.prev_actions = {}
        self.steps_done = 0
        
        self.gamma = gamma
        self.batch_size = 256
        
        self.eps_init, self.eps_final, self.eps_decay = 0.9, 0.05, 1000
        self.num_step = 0

    def select_greedy_action(self, state, cur_turn):
        return self.models[cur_turn](state.unsqueeze(0)).data.max(1)[1].view(1, 1)

    def select_action(self, state, cur_turn):
        sample = random.random()
        self.num_step += 1
        eps_threshold = self.eps_final + (self.eps_init - self.eps_final) * math.exp(-1. * self.num_step / self.eps_decay)
        if sample > eps_threshold:
            return self.select_greedy_action(state, cur_turn)
        else:
            return torch.tensor([[random.randrange(self.n_rows * self.n_cols)]], dtype=torch.int64)
        
    def run_episode(self, e=0, do_learning=True, greedy=False):
        self.env.reset()
        self.prev_states = {1: None,
                            -1: None}
        self.prev_actions = {}
        done = False

        state, _, cur_turn = self.env.getState()
        while not done:
            state_t = self.state_to_tensor(state)
            with torch.no_grad():
                if greedy:
                    action_idx = self.select_greedy_action(state_t.to(self.device), cur_turn).cpu()
                else:
                    action_idx = self.select_action(state_t.to(self.device), cur_turn).cpu()
            self.prev_states[cur_turn] = state_t
            self.prev_actions[cur_turn] = action_idx
            action = self.env.action_from_int(action_idx.numpy()[0][0])
            (next_state, empty_spaces, cur_turn), reward, done, _ = self.env.step(action)
            next_state_t = self.state_to_tensor(next_state)

            if reward == -10:
                transition = (state_t, action_idx, next_state_t, torch.tensor([reward], dtype=torch.float32))
                self.memories[cur_turn].store(transition)
            else:
                if self.prev_states[cur_turn] is not None:
                    if reward == -cur_turn:
                        transition = (self.prev_states[-cur_turn], self.prev_actions[-cur_turn], 
                                      next_state_t, torch.tensor([1.0], dtype=torch.float32))
                        self.memories[-cur_turn].store(transition)

                    transition = (self.prev_states[cur_turn], self.prev_actions[cur_turn], 
                                  next_state_t, torch.tensor([reward * cur_turn], dtype=torch.float32))
                    self.memories[cur_turn].store(transition)

            if do_learning:
                self.learn(cur_turn)

            state = next_state


    def learn(self, cur_turn):
        if np.min([len(self.memories[cur_turn]), len(self.memories[-cur_turn])]) < self.batch_size:
            return
        
        # берём мини-батч из памяти
        transitions = self.memories[cur_turn].sample(self.batch_size)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = Variable(torch.stack(batch_state).to(self.device))
        batch_action = Variable(torch.cat(batch_action).to(self.device))
        batch_reward = Variable(torch.cat(batch_reward).to(self.device))
        batch_next_state = Variable(torch.stack(batch_next_state).to(self.device))
        
        Q = self.models[cur_turn](batch_state)
        Q = Q.gather(1, batch_action).reshape([self.batch_size])
        
        Qmax = self.models[cur_turn](batch_next_state).detach().max(1)[0]
        # Qmax = Qmax.max(1)[0]
        Qnext = batch_reward + (self.gamma * Qmax)

        loss = F.smooth_l1_loss(Q, Qnext)

        self.optimizers[cur_turn].zero_grad()
        loss.backward()
        self.optimizers[cur_turn].step()
        
    def play_game(self, player, episodes=500):
        rewards = []
        for _ in range(episodes):
            self.env.reset()
            state, empty_spaces, cur_turn = self.env.getState()
            done = False
            while not done:
                if cur_turn == player:
                    idx = self.select_greedy_action(self.state_to_tensor(state).to(device), player)
                    action = self.env.action_from_int(idx)
                else:
                    idx = np.random.randint(len(empty_spaces))
                    action = empty_spaces[idx]
                (state, empty_spaces, cur_turn), reward, done, _ = self.env.step(action)
            if reward != -10:
                rewards.append(reward * player)
            else:
                if cur_turn == player:
                    rewards.append(reward)
        return np.array(rewards)

    @staticmethod
    def state_to_tensor(s):
        s = np.array([int(c) for c in s])
        size = int(np.sqrt(len(s)))
        crosses = np.where(s==2, 1, 0).reshape(size, size)
        noughts = np.where(s==0, 1, 0).reshape(size, size)
        empty_spaces = np.where(s==1, 1, 0).reshape(size, size)
        return torch.Tensor(np.stack([crosses, noughts, empty_spaces])).reshape(3, size, size)


class TicTacToeDoubleDQN(TicTacToeDQN):
    def __init__(self, env, n_rows=3, n_cols=3, n_win=3, model_class=Network, gamma=0.8, device=device):
        self.target_models = {1: model_class().to(device), 
                              -1: model_class().to(device)}
        self.episodes_learned = {1: 0, -1: 0}
        super().__init__(self)

    def learn(self, cur_turn):
        if np.min([len(self.memories[cur_turn]), len(self.memories[-cur_turn])]) < self.batch_size:
            return
        
        # берём мини-батч из памяти
        transitions = self.memories[cur_turn].sample(self.batch_size)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = Variable(torch.stack(batch_state).to(self.device))
        batch_action = Variable(torch.cat(batch_action).to(self.device))
        batch_reward = Variable(torch.cat(batch_reward).to(self.device))
        batch_next_state = Variable(torch.stack(batch_next_state).to(self.device))
        
        Q = self.models[cur_turn](batch_state)
        Q = Q.gather(1, batch_action).reshape([self.batch_size])
        
        Qmax = self.target_models[cur_turn](batch_next_state).detach().max(1)[0]
        Qnext = batch_reward + (self.gamma * Qmax)

        loss = F.smooth_l1_loss(Q, Qnext)

        self.optimizers[cur_turn].zero_grad()
        loss.backward()
        self.optimizers[cur_turn].step()
        
        self.episodes_learned[cur_turn] += 1
        if self.episodes_learned[cur_turn] % 500:
            self.target_models[cur_turn].load_state_dict(self.models[cur_turn].state_dict())