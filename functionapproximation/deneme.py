import numpy as np
import gymnasium as gym
import tqdm
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

class dqn_network(nn.Module):
    def __init__(self, lr, input_dim, n_actions, n1_dim=256, n2_dim=256):
        super(dqn_network, self).__init__()
        self.n1 = nn.Linear(input_dim, n1_dim)
        self.n2 = nn.Linear(n1_dim, n2_dim)
        self.n3 = nn.Linear(n2_dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.n1(state))
        x = F.relu(self.n2(x))
        actions = self.n3(x)
        return actions

class agent:
    def __init__(
        self,
        gamma=0.99,
        epsilon=1.0,
        lr=1e-3,
        input_dim=8,
        batch_size=64,
        n_actions=4,
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
        self.action_space = [i for i in range(n_actions)]
        self.mem_cntr = 0

        self.q_eval = dqn_network(lr, input_dim, n_actions)
        # target network (copy)
        self.q_target = copy.deepcopy(self.q_eval)
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool_)
        self.state_mem = np.zeros((self.mem_size, input_dim), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, input_dim), dtype=np.float32)

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
            state = T.tensor([np.array(observation)], dtype=T.float32).to(self.q_eval.device)#??
            with T.no_grad():
                actions = self.q_eval(state)
            action = int(T.argmax(actions).item())
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)


        state_batch = T.tensor(self.state_mem[batch], dtype=T.float32).to(self.q_eval.device)
        new_state_batch = T.tensor(self.new_state_mem[batch], dtype=T.float32).to(self.q_eval.device)
        reward_batch = T.tensor(self.reward_mem[batch], dtype=T.float32).to(self.q_eval.device)
        terminal_batch = T.tensor(self.terminal_mem[batch], dtype=T.bool).to(self.q_eval.device)
        action_batch = T.tensor(self.action_mem[batch], dtype=T.long).to(self.q_eval.device)

        # Q(s,a) for chosen actions
        q_eval_all = self.q_eval(state_batch)                      # shape [batch, n_actions]
        q_eval = q_eval_all.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # shape [batch]

        # Q_next from target network (for stability)
        with T.no_grad():
            q_next_all = self.q_target(new_state_batch)           # shape [batch, n_actions]
            q_next_all[terminal_batch] = 0.0
            q_next = T.max(q_next_all, dim=1)[0]

        q_target = reward_batch + self.gamma * q_next

        loss = self.q_eval.loss(q_eval, q_target.detach()) # theta update
        loss.backward()
        self.q_eval.optimizer.step()

        # update epsilon
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

        # update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

# -------- training loop --------
env = gym.make('LunarLander-v3',render_mode = "human")
agent = agent(
    gamma=0.99,
    epsilon=1.0,
    lr=1e-3,
    input_dim=8,
    batch_size=64,
    n_actions=4,
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
