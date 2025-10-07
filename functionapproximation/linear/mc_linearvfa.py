import gymnasium as gym
import numpy as np
import tqdm 
import matplotlib.pyplot as plt
#sklearn

class linear_vfa_agent:
    def __init__(self, alpha = 0.1, gamma = 0.99, epsilon = 0.1 ):
        self.alpha = alpha #learning rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.w = None

    def in_w(self, n_actions, n_features):
        self.w = np.zeros((n_actions,n_features))

    def calculate_q(self,state):
        features = np.array(state)
        return np.dot(self.w,features)
    
    def choose_action(self, state, action_samples):
        if np.random.random() < self.epsilon:
            return np.random.choice(action_samples)
        return np.argmax(self.calculate_q(state))
    
    def update(self, state, action, reward, next_state, done):
        features = np.array(state)
        q = self.calculate_q(state)
        q_next = 0 if done else np.max(self.calculate_q(next_state))
        td_target = reward + self.gamma * q_next 
        td_error = td_target - q[action]
        self.w[action] += self.alpha *td_error*features 


env = gym.make("MountainCar-v0") #render_mode="human")

agent = linear_vfa_agent()
agent.in_w(env.action_space.n, env.observation_space.shape[0])

num_episodes = 1000
reward_per_episode = []
possible_actions = np.arange(env.action_space.n)

for _ in tqdm.tqdm(range(num_episodes)):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state, possible_actions)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    reward_per_episode.append(total_reward)


plt.plot(reward_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("MountainCar Linear VFA Learning")
plt.savefig("mountain_car.png")




