# td3_agent.py
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Selecting the device (CPU or GPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS) device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = int(max_size) # Ensure max_size is integer
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[self.ptr] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)
      # Update ptr only after appending if storage is not full yet
      if len(self.storage) == self.max_size:
          self.ptr = 0 # Reset pointer when buffer becomes full
      else:
          self.ptr = len(self.storage) # Pointer is the next index to fill

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind:
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

class Actor(nn.Module):

  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    # Scale the output to [-max_action, +max_action]
    x = self.max_action * torch.tanh(self.layer_3(x))
    return x

class Critic(nn.Module):

  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1

class TD3(object):

  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    # Important: Ensure state is a numpy array here
    state = torch.Tensor(state.reshape(1, -1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

    for it in range(iterations):

      # Step 4: Sample transitions
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)

      # Step 5: Get next action from target actor
      next_action = self.actor_target(next_state)

      # Step 6: Add noise to target action
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

      # Step 7: Get target Q values from target critics
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)

      # Step 8: Keep minimum target Q
      target_Q = torch.min(target_Q1, target_Q2)

      # Step 9: Calculate final target Q
      target_Q = reward + ((1 - done) * discount * target_Q).detach()

      # Step 10: Get current Q values from critics
      current_Q1, current_Q2 = self.critic(state, action)

      # Step 11: Compute Critic loss
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

      # Step 12: Optimize Critics
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

      # Step 13: Delayed policy updates
      if it % policy_freq == 0:
        # Compute Actor loss
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Step 14: Update Actor target network (Polyak Averaging)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Step 15: Update Critic target network (Polyak Averaging)
        # ERROR in original code snippet: Was updating actor target again. Should update critic target.
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

  # Making a save method to save a trained model
  def save(self, filename, directory):
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    torch.save(self.actor.state_dict(), os.path.join(directory, f'{filename}_actor.pth'))
    torch.save(self.critic.state_dict(), os.path.join(directory, f'{filename}_critic.pth'))
    print(f"*** Model Saved to {directory} ***")

  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    actor_path = os.path.join(directory, f'{filename}_actor.pth')
    critic_path = os.path.join(directory, f'{filename}_critic.pth')
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        print(f"Error: Model files not found in {directory}")
        return

    print(f"Loading models from {directory}...")
    # Load actor state potentially mapping location based on device
    self.actor.load_state_dict(torch.load(actor_path, map_location=device))
    self.actor_target.load_state_dict(self.actor.state_dict()) # Update target network

    # Load critic state potentially mapping location based on device
    self.critic.load_state_dict(torch.load(critic_path, map_location=device))
    self.critic_target.load_state_dict(self.critic.state_dict()) # Update target network
    print(f"*** Model Loaded from {directory} ***")