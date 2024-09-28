import random
from collections import namedtuple, deque
import lbforaging
import tqdm
import numpy as np
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
use_wanb = True
if use_wanb:
    wandb.init(project="maddpg-foraging", name = "SEAC")

class ReplayBuffer():
    def __init__(self, capacity = 100000):
        self.buffer = deque(maxlen = capacity)
    
    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)
    
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)   #(6,) tensor
    
class Value(nn.Module):
    def __init__(self, input_size):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SEAC():
    def __init__(self, state_dim, action_dim, n_agents = 2, lr=0.01, gamma=0.95, lambd=0.01):
        self.n_agents = n_agents
        self.gamma = gamma
        self.lambd = lambd
        
        self.Actors = [Actor(state_dim, action_dim) for _ in range(n_agents)]
        self.Values = [Value(state_dim) for _ in range(n_agents)]
        
        self.ActorOptimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.Actors]
        self.ValueOptimizers = [optim.Adam(value.parameters(), lr=lr) for value in self.Values]
        
        
    def obs_to_array(self, obs, agent_idx):
        # obs is a tuple: ((agent1_obs, agent2_obs), {})
        return obs[0][agent_idx].flatten()
    
    
    def select_action(self, obs):
        
        """Select action for the actor P(a_i|o_i, theta_i)

        Args:
            obs (Tuple): Observation by lbforaging

        Returns:
            List : Returns list of actions to be taken [a1, a2]
        """
        actions = []
        for agent_idx, actors in enumerate(self.Actors):
            state_array = self.obs_to_array(obs, agent_idx)
            state = torch.FloatTensor(state_array).unsqueeze(0) 
            action_probs = actors(state).squeeze().detach().numpy()
            action_probs = action_probs / np.sum(action_probs) #Check for probability distribution
            action = np.random.choice(len(action_probs), p=action_probs)
            actions.append(action)
        return actions
    
    def step(self, ReplayBuffers):
        for i in range(self.n_agents):
            
            Memory = ReplayBuffers[i].buffer[-1]
            
            agent_value_loss = self.Values[i](Memory.state) - Memory.reward - self.gamma * self.Values[i](Memory.next_state)
            
            agent_loss = -1 * torch.log(self.Actors[i](Memory.state)[Memory.action])*(-1 * agent_value_loss.detach())
            
            experience_loss = 0
            experience_value_loss = 0
            
            for j in range(self.n_agents):
                if j!= i: 
                    
                    otherMemory = ReplayBuffers[j].buffer[-1]
                    
                    policy_ratio = (self.Actors[i](otherMemory.state)[otherMemory.action]) / (self.Actors[j](otherMemory.state)[otherMemory.action])
                    
                    otherAgents_value_loss = -otherMemory.reward - self.gamma * self.Values[i](otherMemory.next_state) + self.Values[i](otherMemory.state)
                    
                    otherAgents_loss = -1 * torch.log(self.Actors[i](otherMemory.state)[otherMemory.action]) * (-1 * otherAgents_value_loss.detach())
                    
                    experience_loss += policy_ratio * otherAgents_loss
                    
                    experience_value_loss += policy_ratio * otherAgents_value_loss**2
            
            policy_loss = agent_loss + self.lambd * experience_loss
            
            self.ActorOptimizers[i].zero_grad()
            policy_loss.backward()
            self.ActorOptimizers[i].step()
            
            value_loss = agent_value_loss**2 + self.lambd *  otherAgents_value_loss
            
            self.ValueOptimizers[i].zero_grad()
            value_loss.backward()
            self.ValueOptimizers[i].step()
                    
    
def train_seac(env, episodes = 50000, n_steps = 1000):

    start_state = env.reset()
    state_dim = start_state[0][0].shape[0]
    action_dim = env.action_space[0].n
    n_agents = 2
    env_steps = 0 
    
    seac = SEAC(state_dim, action_dim)
    ReplayBuffers = [ReplayBuffer() for _ in range(n_agents)]
    
    for _ in tqdm(range(episodes), desc="Training", leave = True):
        obs = env.reset()
        episode_rewards = [0 for _ in range(n_agents)]
        for timestep in range(n_steps): 
            env_steps += 1
            actions = seac.select_action(obs)
            next_obs, rewards, dones, _ , _ = env.step(actions)
            next_obs = ((next_obs), {})

            if isinstance(dones, bool):  
                dones = [dones] * n_agents
            
            if dones == True: 
                dones = [True] * n_agents
            
            for i in range(n_agents):
                state_array = seac.obs_to_array(obs, i)
                next_state_array = seac.obs_to_array(next_obs, i)
                ReplayBuffers[i].add(torch.tensor(state_array), torch.tensor(actions[i]), torch.tensor(rewards[i]), torch.tensor(next_state_array), torch.tensor(dones))
                episode_rewards[i] += rewards[i]
                
            if use_wanb:
                wandb.log({"Average Reward": np.mean(rewards)}, step = env_steps)
            
            obs = next_obs
            seac.step(ReplayBuffers)
            
            if all(dones):
                break
            
            
if __name__ == "__main__": 
    env = gym.make("Foraging-8x8-2p-2f-v3")
    train_seac(env)