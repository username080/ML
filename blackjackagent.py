import gymnasium as gym
from collections import defaultdict
import numpy as np

class BlackjackAgent:
    def __init__(self,
                env: gym.Env,
                learning_rate: float,
                epsilon: float,
                epsilon_decay: float,
                final_epsilon: float,
                discount_factor: float = 0.95):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.training_error = []

    def choose_action(self, obs: tuple[int,int,bool]) ->int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_values[obs])
    
    def update_q_value(self,
                    action: int,
                    reward: float,
                    obs: tuple[int,int,bool],
                    terminated: bool,
                    next_obs: tuple[int,int,bool]):
        
        next_q = 0 if terminated else np.max(self.q_values[next_obs])

        second_part = reward + self.discount_factor * next_q
        temporal_difference = second_part - self.q_values[obs][action]
        third_part = self.learning_rate * temporal_difference
        final_part = self.q_values[obs][action] + third_part

        self.q_values[obs][action] = final_part
        self.training_error.append(abs(temporal_difference))

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



learning_rate = 0.05        
n_episodes = 100_000      
start_epsilon = 1.0         
epsilon_decay = start_epsilon / (n_episodes / 2) 
final_epsilon = 0.1
total_reward = 0
env = gym.make("Blackjack-v1", sab=True)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

from tqdm import tqdm

agent = BlackjackAgent(env,learning_rate,start_epsilon,epsilon_decay,final_epsilon)

for episode in tqdm(range(n_episodes)):
    obs, _ = env.reset()
    terminated = False

    while not terminated:
        action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update_q_value(action,reward,obs,terminated,next_obs)

        obs = next_obs

    agent.decay_epsilon()


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.savefig("blackjack_agent_performance.png")

print(f"total_Reward: {total_reward}")