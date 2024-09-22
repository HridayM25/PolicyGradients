import random
from collections import namedtuple, deque
import lbforaging
import tqdm
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
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
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim * n_agents + action_dim * n_agents, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MADDPG:
    def __init__(self, state_dim, action_dim, n_agents=2, lr=0.01, gamma=0.95, tau=0.01):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        
        self.actors = [Actor(state_dim, action_dim) for _ in range(n_agents)]
        self.critics = [Critic(state_dim, action_dim, n_agents) for _ in range(n_agents)]
        self.actors_target = [Actor(state_dim, action_dim) for _ in range(n_agents)]
        self.critics_target = [Critic(state_dim, action_dim, n_agents) for _ in range(n_agents)]
        
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr) for critic in self.critics]
        
        # Initialize target networks
        for actor, actor_target in zip(self.actors, self.actors_target):
            self.soft_update(actor, actor_target, 1.0)
        for critic, critic_target in zip(self.critics, self.critics_target):
            self.soft_update(critic, critic_target, 1.0)
    
    def select_action(self, obs):
        actions = []
        for i, actor in enumerate(self.actors):
            state_array = self.obs_to_array(obs, i)

            state = torch.FloatTensor(state_array).unsqueeze(0) 
            action_probs = actor(state).squeeze().detach().numpy()  

            if action_probs.ndim == 2:
                action_probs = action_probs[0]  
            
            action_probs = action_probs / np.sum(action_probs)
            
            action = np.random.choice(len(action_probs), p=action_probs)
            actions.append(action)
        return actions

    def obs_to_array(self, obs, agent_idx):
        # obs is a tuple: ((agent1_obs, agent2_obs), {})
        return obs[0][agent_idx].flatten()

    def dict_to_array(self, dict_obs):
        return np.concatenate([
            dict_obs['player'].flatten(),
            dict_obs['field'].flatten(),
            dict_obs['food'].flatten(),
            [dict_obs['hunger']]
        ])
    
    def update(self, memories, batch_size):
        for i in range(self.n_agents):
            if len(memories[i]) < batch_size:
                return
            
            states, actions, rewards, next_states, dones = memories[i].sample(batch_size)
            print("CHECK2")
            actions_one_hot = torch.zeros(batch_size, self.n_agents, actions.shape[-1])
            for j in range(self.n_agents):
                actions_one_hot[:, j, actions[:, j].long()] = 1

            target_actions = torch.zeros_like(actions_one_hot)
            for j, actor_target in enumerate(self.actors_target):
                target_actions[:, j] = actor_target(next_states[:, j*states.shape[1]:(j+1)*states.shape[1]])
            
            target_critic_input = torch.cat((next_states, target_actions.view(batch_size, -1)), dim=1)
            current_critic_input = torch.cat((states, actions_one_hot.view(batch_size, -1)), dim=1)
            
            target_q = rewards[:, i] + self.gamma * self.critics_target[i](target_critic_input) * (1 - dones[:, i])
            current_q = self.critics[i](current_critic_input)
            
            critic_loss = nn.MSELoss()(current_q, target_q.detach())
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            pred_actions = torch.zeros_like(actions_one_hot)
            for j in range(self.n_agents):
                if j == i:
                    pred_actions[:, j] = self.actors[j](states[:, j*states.shape[1]:(j+1)*states.shape[1]])
                else:
                    pred_actions[:, j] = self.actors[j](states[:, j*states.shape[1]:(j+1)*states.shape[1]]).detach()
            
            actor_loss = -self.critics[i](torch.cat((states, pred_actions.view(batch_size, -1)), dim=1)).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            
            self.soft_update(self.critics[i], self.critics_target[i], self.tau)
            self.soft_update(self.actors[i], self.actors_target[i], self.tau)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def train_maddpg(env, n_episodes=10000, max_steps=100, batch_size=64):
    n_agents = env.n_agents
    sample_obs = env.reset()
    state_dim = sample_obs[0][0].shape[0] 
    action_dim = env.action_space[0].n
    
    maddpg = MADDPG(state_dim, action_dim, n_agents)
    memories = [ReplayBuffer() for _ in range(n_agents)]
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_rewards = [0 for _ in range(n_agents)]
        # print("The obs is ", obs)
        for step in range(max_steps):
            print(f"Episode : {episode}, Step : {step}")
            actions = maddpg.select_action(obs)
            print("THE ACTIONS ARE ", actions)
            next_obs, rewards, dones, _ , _ = env.step(actions)

            
            if isinstance(dones, bool):  
                dones = [dones] * n_agents
            
            for i in range(n_agents):
                state_array = maddpg.obs_to_array(obs, i)
                next_state_array = maddpg.obs_to_array(next_obs, i)
                print("CHECK1")
                memories[i].add(state_array, actions[i], rewards[i], next_state_array, dones[i])
                episode_rewards[i] += rewards[i]
            
            obs = ((next_obs), {})

            
            maddpg.update(memories, batch_size)
            
            if all(dones):
                break
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Rewards: {np.mean(episode_rewards)}")
    
    return maddpg

if __name__ == "__main__":
    env = gym.make("Foraging-8x8-2p-1f-v3")
    trained_maddpg = train_maddpg(env)
    print("Training completed!")