import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import copy

import torch.nn.functional as F
import torch.optim as optim
import torch as T
import torch.nn as nn

# rainbow:
# distribution olacak
# pre olacak
# noisy nets ?

class ddqn_network(nn.Module):
    def __init__(self, lr, state_dim, action_dim, n1_dim=256, n2_dim=256):
        super(ddqn_network, self).__init__()
        # 🔸 Ortak (feature) katman
        self.feature = nn.Sequential(
            nn.Linear(state_dim, n1_dim),
            nn.ReLU()
        )

        # 🔸 Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(n1_dim, n2_dim),
            nn.ReLU(),
            nn.Linear(n2_dim, 1)
        )

        # 🔸 Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(n1_dim, n2_dim),
            nn.ReLU(),
            nn.Linear(n2_dim, action_dim)
        )

        # 🔸 Optimizasyon
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # 🔸 Cihaz
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        # Ortak temsil
        x = self.feature(state)
        # Ayrı stream’ler
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        # Dueling birleşimi
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


class agent:
    def __init__(
        self,
        gamma=0.99,
        epsilon=1.0,
        lr=1e-3,
        state_dim=8,
        batch_size=64,
        action_dim=4,
        max_mem_size=100000,
        eps_end=0.01,
        eps_dec=5e-4,
        target_update_freq=1000
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(action_dim)]
        self.mem_cntr = 0

        self.online_network = ddqn_network(lr, state_dim, action_dim)
        # target network (copy)
        self.target_network = copy.deepcopy(self.online_network)

        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool_)
        self.state_mem = np.zeros((self.mem_size, state_dim), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, state_dim), dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.reward_mem[index] = reward
        self.state_mem[index] = state
        self.action_mem[index] = action
        self.new_state_mem[index] = state_
        self.terminal_mem[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([np.array(observation)], dtype=T.float32).to(self.online_network.device)
            with T.no_grad():
                actions = self.online_network(state)# get max action with respect to online network
            action = int(T.argmax(actions).item())
        else:
            action = np.random.choice(self.action_space)
        return action



    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.online_network.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)


        state_batch = T.tensor(self.state_mem[batch], dtype=T.float32).to(self.online_network.device)
        new_state_batch = T.tensor(self.new_state_mem[batch], dtype=T.float32).to(self.online_network.device)
        reward_batch = T.tensor(self.reward_mem[batch], dtype=T.float32).to(self.online_network.device)
        terminal_batch = T.tensor(self.terminal_mem[batch], dtype=T.bool).to(self.online_network.device)
        action_batch = T.tensor(self.action_mem[batch], dtype=T.long).to(self.online_network.device)

        # Q(s,a) for chosen actions
        online_network_all = self.online_network(state_batch)                      # shape [batch, action_dim]
        online_network = online_network_all.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # shape [batch]

        # next_state_q from target network (for stability)
        with T.no_grad():
            next_actions = self.online_network(new_state_batch).argmax(dim=1)#[batch]
            next_state_q_all = self.target_network(new_state_batch).gather(1,next_actions.unsqueeze(1))
            next_state_q_all[terminal_batch] = 0.0
            next_state_q = next_state_q_all.squeeze(1)
            

        target_value = reward_batch + self.gamma * next_state_q # R(t+1) + gamma * Q(t+1)target(S',Qmaxonline(S',a')) = target value

        loss = self.online_network.loss(online_network, target_value.detach()) # theta update
        loss.backward()
        self.online_network.optimizer.step()

        # update epsilon
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

        # update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())





env = gym.make('LunarLander-v3')#,render_mode = "human")
agent = agent(
    gamma=0.99,
    epsilon=1.0,
    lr=1e-3,
    state_dim=8,
    batch_size=64,
    action_dim=4,
    eps_end=0.05,
    eps_dec=1e-3,
    target_update_freq=1000
)

reward_per_episode = []
n_games = 300

for i in tqdm.tqdm(range(n_games)):
    score = 0
    done = False
    observation, info = env.reset()
    while not done:
        action = agent.choose_action(observation)
        new_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        agent.store_transition(observation, action, reward, new_observation, done)
        agent.learn()
        observation = new_observation
    reward_per_episode.append(score)

plt.figure(figsize=(12,6))
plt.plot(reward_per_episode, label='Reward per Episode')
window = 20
smoothed_rewards = [np.mean(reward_per_episode[i-window:i+1]) if i >= window else np.mean(reward_per_episode[:i+1]) for i in range(len(reward_per_episode))]
plt.plot(smoothed_rewards, label=f'{window}-episode moving average')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress')
plt.legend()
plt.savefig('dqn.png')
