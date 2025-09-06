
import numpy as np

from tqdm import tqdm

num_episodes = 10000


class environment():
    def __init__(self):
        self.random_std = 0.1
        self.q_true = np.zeros(10)
        self.walk_std = 0.01
    def step(self, action):
        reward = np.random.normal(self.q_true[action], 1.0)
        self.q_true += np.random.normal(0, self.walk_std, 10)

        return reward
    

    
class agent_constant():
    def __init__(self):
        self.a = 0.1
        self.q_values = np.zeros(10)
        self.epsilon=0.1

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 10)
        else:
            return np.argmax(self.q_values)

        
    
    def update(self, action, reward):
        self.q_values[action] += self.a * (reward - self.q_values[action])


class agent_sample_average():
    def __init__(self):
        self.q_values = np.zeros(10)
        self.action_count = np.zeros(10)
        self.epsilon=0.1

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 10)
        else:
            return np.argmax(self.q_values)



    def update(self, action, reward):
        self.action_count[action] += 1
        self.q_values[action] = self.q_values[action] + 1/self.action_count[action] * (reward - self.q_values[action])


env = environment()
agent1 = agent_constant()
agent2 = agent_sample_average()

total_reward = 0
for episode in tqdm(range(num_episodes)):
    action = agent1.select_action()
    reward = env.step(action)
    total_reward += reward
    agent1.update(action,reward) 


for episode in tqdm(range(num_episodes)):
    action = agent2.select_action()
    reward = env.step(action)
    agent2.update(action,reward)

print(total_reward)
#matplotlib kısmını atladım
