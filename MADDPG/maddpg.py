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

wandb.init(project="maddpg-foraging", name = "MADDPG")

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
        self.fc1 = nn.Linear((state_dim + action_dim) * n_agents , 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, state, action):
        # print(state.shape, action.shape)
        x = torch.cat((state, action.squeeze(1)), dim=1)
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
        
    def ProcessMemories(self, memories, num_agents, batch_size):
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_target_actions = []
        all_dones = []

        for i in range(num_agents): 
            states_i, actions_i, rewards_i, next_states_i, dones_i = memories[i].sample(batch_size)
            actions_one_hot_i = F.one_hot(actions_i.long(), num_classes=6)
            target_actions_i = self.actors_target[i](next_states_i)
            target_actions_one_hot_i = F.one_hot(target_actions_i.argmax(dim=1), num_classes=6)
            
            all_states.append(states_i)
            all_actions.append(actions_one_hot_i)
            all_rewards.append(rewards_i)
            all_next_states.append(next_states_i)
            all_target_actions.append(target_actions_one_hot_i)
            all_dones.append(dones_i)

        states = torch.cat(all_states, dim=1)
        actions = torch.cat(all_actions, dim=1)
        rewards = torch.cat(all_rewards, dim=1)
        next_states = torch.cat(all_next_states, dim=1)
        target_actions = torch.cat(all_target_actions, dim=1)
        dones = torch.cat(all_dones, dim=1)
        
        return states, actions, rewards, next_states, target_actions, dones
    
    def update(self, memories, batch_size):
        for i in range(self.n_agents):
            if len(memories[i]) < batch_size:
                return
            
            combinedStates, combinedActions, _,  combinedNextStates, combinedTargetActions, _ = self.ProcessMemories(memories, self.n_agents, batch_size)
            # print("HOPEFULLY WORKS")
            # print(combinedStates.shape, combinedActions.shape, combinedNextStates.shape, combinedTargetActions.shape)
            
            combinedActions = combinedActions.view(batch_size, -1)
            
            states, actions, rewards, next_states, dones = memories[i].sample(batch_size)
            
            actions_one_hot = F.one_hot(actions.long(), num_classes=6)
            
            target_actions = self.actors_target[i](next_states)

            target_q = rewards + self.gamma * self.critics_target[i](combinedNextStates, combinedTargetActions) * (1 - dones)

            current_q = self.critics[i](combinedStates, combinedActions) #combined actions is a one hot vector

            critic_loss = F.mse_loss(current_q, target_q.detach())
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            
            actor_loss = -self.critics[i](combinedStates, combinedActions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            
            self.soft_update(self.critics[i], self.critics_target[i], self.tau)
            self.soft_update(self.actors[i], self.actors_target[i], self.tau)

    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def train_maddpg(env, n_episodes=10000, max_steps=1000, batch_size=64):
    n_agents = env.n_agents
    sample_obs = env.reset()
    state_dim = sample_obs[0][0].shape[0] 
    action_dim = env.action_space[0].n
    
    maddpg = MADDPG(state_dim, action_dim, n_agents)
    memories = [ReplayBuffer() for _ in range(n_agents)] #Separate RBs? Will need to verify this.
    
    for episode in tqdm(range(n_episodes), desc="Training", leave = True):
        obs = env.reset()
        episode_rewards = [0 for _ in range(n_agents)]
        # print("The obs is ", obs)
        for step in range(max_steps):
            # print(f"Episode : {episode}, Step : {step}")
            actions = maddpg.select_action(obs)
            next_obs, rewards, dones, _ , _ = env.step(actions)
            next_obs = ((next_obs), {})
            
            if isinstance(dones, bool):  
                dones = [dones] * n_agents
            
            if dones == True: 
                dones = [True] * n_agents
                
            
            for i in range(n_agents):
                state_array = maddpg.obs_to_array(obs, i)
                next_state_array = maddpg.obs_to_array(next_obs, i)
                # print("ACTIONS ", actions[i])
                # print("REWARDS ", rewards[i])
                # try:
                #     print("DONES ", dones[i])
                # except:
                #     print("DONES ", dones)
                memories[i].add(state_array, actions[i], rewards[i], next_state_array, dones[i])
                episode_rewards[i] += rewards[i]
            
            obs = next_obs
            
            maddpg.update(memories, batch_size)
            
            if all(dones):
                break
        
        # if episode % 1 == 0:
        #     print(f"Episode {episode}, Average Rewards: {np.mean(episode_rewards)}")
        wandb.log({"Average Reward": np.mean(episode_rewards)}, step = episode)
    
    return maddpg

if __name__ == "__main__":
    env = gym.make("Foraging-8x8-2p-1f-v3")
    trained_maddpg = train_maddpg(env)
    print("Training completed!")