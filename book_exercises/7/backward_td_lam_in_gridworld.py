import numpy as np
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
class GridWorld:
    def __init__(self, size=5, start=(0,0), goal=(4,4)):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.action_space = [0, 1, 2, 3]
        self.n_actions = len(self.action_space)
        self.n_states = size * size

    def reset(self):
        self.state = self.start
        return self.state
    
    def get_state(self):
        return self.state

    def step(self, action):
        x, y = self.state

        if action == 0:   # up
            x = max(0, x - 1)
        elif action == 1: # right
            y = min(self.size - 1, y + 1)
        elif action == 2: # down
            x = min(self.size - 1, x + 1)
        elif action == 3: # left
            y = max(0, y - 1)

        self.state = (x, y)

        # Reward: -1 each step, +10 at goal
        if self.state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        return self.state, reward, done

    def render(self):
        grid = np.full((self.size, self.size), " . ")
        x, y = self.state
        gx, gy = self.goal
        grid[gx, gy] = " G "
        grid[x, y] = " A "
        for row in grid:
            print("".join(row))
        print()

class backward_TD_lam:
    def __init__(self, alpha = 0.1, learning_rate = 0.3, lam = 0.1, gamma = 0.1):
        
        self.lam = lam
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.V = defaultdict(float)
        self.E = defaultdict(float)

    def choose_action(self,state):
       return np.random.choice([0,1,2,3])

    def get_v(self,state):
        return self.V[state]

    def reset_E(self):
        self.E.clear()

    def update(self, state, delta):
        self.E[state] = 1
        for k in self.V.keys():
            self.V[k] = self.V[k] + self.learning_rate * delta * self.E[k]
        for k in self.E.keys():
            self.E[k] = self.lam * self.gamma * self.E[k] 


env = GridWorld()
agent = backward_TD_lam()

num_episode = 500
gamma = 0.1

rewards_per_episode = []
for _ in tqdm.tqdm(range(num_episode)):
    
    agent.reset_E()
    state = env.get_state()
    done = False
    total_reward = 0
    while not done:

        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        delta = reward + agent.get_v(next_state) * gamma - agent.get_v(state)
        agent.update(state, delta)
        total_reward += reward
        state = next_state
        
    rewards_per_episode.append(total_reward)

# --- Chart ---
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Backward TD(λ) Learning Curve")
plt.savefig("tdbackview.png")
