import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import copy
import torch as T
import torch.nn as nn
import torch.optim as optim

# distribution
# multi-step
# noisy nets


# --------- Device ----------
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
print("Device:", device)

# --------- NETWORK (DUELING DDQN) ----------
class DDQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n1=128, n2=128):  # daha küçük ama yeterli
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, n1),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Linear(n2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Linear(n2, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        return V + (A - A.mean(dim=1, keepdim=True))

# --------- PER BUFFER (NumPy / CPU) ----------
class PERBuffer:
    def __init__(self, state_dim, max_size=100000, alpha=0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size,), dtype=np.int64)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.bool_)
        self.priorities = np.ones((max_size,), dtype=np.float32)  # default 1

    def store(self, s, a, r, s_, done):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_
        self.dones[self.ptr] = done
        # set max priority for new sample
        max_prio = self.priorities[:self.size].max() if self.size > 0 else 1.0
        self.priorities[self.ptr] = max_prio
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, beta=0.4):
        # compute probabilities on CPU
        prios = self.priorities[:self.size] ** self.alpha
        total = prios.sum()
        if total == 0:
            probs = np.ones(self.size) / self.size
        else:
            probs = prios / total
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        weights = (self.size * probs[indices]) ** (-beta)
        # normalize by mean (keeps relative importance, avoids tiny values)
        weights = weights / (weights.mean() + 1e-8)
        return indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        # td_errors: numpy array abs values
        self.priorities[indices] = np.abs(td_errors) + epsilon

# --------- AGENT ----------
class Agent:
    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, batch_size=64,
                 eps_start=1.0, eps_end=0.05, eps_dec=5e-4,
                 target_update=500, buffer_size=50000,
                 alpha=0.6, beta_start=0.4, beta_frames=20000):  # beta_frames daha kısa

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta_idx = 1

        self.buffer = PERBuffer(state_dim, max_size=buffer_size, alpha=alpha)

        self.online = DDQNNetwork(state_dim, action_dim).to(device)
        self.target = copy.deepcopy(self.online).to(device)
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='none')  # we'll weight manually

        self.target_update = target_update
        self.learn_step = 0

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = T.tensor(state[None, :], dtype=T.float32, device=device)
        with T.no_grad():
            q_vals = self.online(state_t)
            action = int(T.argmax(q_vals, dim=1).item())
        return action

    def learn(self):
        if self.buffer.size < self.batch_size:
            return

        indices, weights = self.buffer.sample(self.batch_size, beta=self.beta)

        # fetch batch from CPU buffer, then convert to tensors on device once
        states = T.tensor(self.buffer.states[indices], dtype=T.float32, device=device)
        actions = T.tensor(self.buffer.actions[indices], dtype=T.int64, device=device)
        rewards = T.tensor(self.buffer.rewards[indices], dtype=T.float32, device=device)
        next_states = T.tensor(self.buffer.next_states[indices], dtype=T.float32, device=device)
        dones = T.tensor(self.buffer.dones[indices].astype(np.uint8), dtype=T.float32, device=device)
        weights_t = T.tensor(weights, dtype=T.float32, device=device)

        # Q(s,a)
        q_pred = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with T.no_grad():
            next_actions = self.online(next_states).argmax(dim=1)
            q_next = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_next = q_next * (1.0 - dones)
            target = rewards + self.gamma * q_next

        td_errors = (target - q_pred).detach()

        # update priorities on CPU to avoid GPU<->CPU sync
        self.buffer.update_priorities(indices, td_errors.abs().cpu().numpy())

        # weighted loss
        loss_per_sample = weights_t * self.loss_fn(q_pred, target)
        loss = loss_per_sample.mean()

        self.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.optimizer.step()

        # epsilon decay (per learn step)
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

        # beta annealing (faster)
        self.beta = min(1.0, self.beta_start + self.beta_idx * (1.0 - self.beta_start) / max(1, self.beta_frames))
        self.beta_idx += 1

        # target update
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

# --------- TRAIN ----------
def train(n_games=300, render=False):
    env = gym.make('LunarLander-v3', render_mode=None)
    agent = Agent(state_dim=8, action_dim=4,
                  lr=3e-4, batch_size=64, target_update=1000,
                  buffer_size=50000, alpha=0.6, beta_start=0.4, beta_frames=15000)

    reward_history = []

    for ep in tqdm.tqdm(range(n_games)):
        state, _ = env.reset()
        done = False
        score = 0
        steps = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.store(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            score += reward
            steps += 1
            if render:
                env.render()
        reward_history.append(score)
    env.close()
    return reward_history

if __name__ == "__main__":
    rewards = train(n_games=300)
    # PLOT
    plt.figure(figsize=(12,6))
    plt.plot(rewards, label='Reward per Episode')
    window = 20
    smoothed = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
    plt.plot(smoothed, label='20-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Dueling DDQN + PER (optimized sampling on CPU)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("dueling_ddqn_per")
