import os
import time
import random
import numpy as np
import matplotlib.pyplor as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import  Variable
from collections import deque

class ReplayBuffer(object):
    def __init__(self, max_size = 1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) = self.max_size :
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size) :
        ind = np.random.randint(0, len(self.storage), batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy = False))
            batch_next_states.append(np.array(state, copy = False))
            batch_actions.append(np.array(state, copy = False))
            batch_rewards.append(np.array(state, copy = False))
            batch_dones.append(np.array(state, copy = False))

        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions),\
                np.array(batch_rewards).reshape(-1,1), np.array(batch_dones).reshape(-1,1)


class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dims, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dims)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dims, action_dims):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dims + action_dims, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dims)

        self.layer_4 = nn.Linear(state_dims + action_dims, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, action_dims)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)

        x2 = F.relu(self.layer_4(xu))
        x2 = f.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)

        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class T3D(object):
    def __init__(self, state_dims, action_dims, max_action):
        self.actor = Actor(state_dims, action_dims, max_action).to(device)
        self.actor_target = Actor(state_dims, action_dims, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dims, action_dims).to(device)
        self.critic_target = Critic(state_dims, action_dims, action_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict)

        self.critic_optmizer = torch.optim.Adam(self.critic.parameters())
        self.max_action =  max_action

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1,-1).to(device))
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, iterations, batch_size = 100, discount = 0.99, tau = 0.005, policy_noise = 0.2, noise_clip = 0.5, policy_freq = 2) :
        for it in range(iterations):
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            next_action = self.actor_target.forward(next_state)

            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target.forward(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            current_Q1, current_Q2 = self.critic.forward(state, action)

            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, current_Q)

            self.critic_optmizer.zero_grad()
            critic_loss.backward()
            self.critic_optmizer.step()

            if it % policy_freq == 0:
                actor_loss = -(self.critic.Q1(state, self.actor(state)).mean())
                self.actor_optimizer.grad_zero()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
