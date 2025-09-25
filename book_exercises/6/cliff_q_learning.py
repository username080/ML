import numpy as np
from collections import defaultdict
import tqdm
import matplotlib.pyplot as plt

class CliffWalkingEnv:
    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start = (rows - 1, 0)      # (3,0)
        self.goal = (rows - 1, cols-1)  # (3,11)

        # Cliff bölgeleri: alt satır, start ve goal hariç
        self.cliffs = [(rows - 1, c) for c in range(1, cols - 1)]

        self.state = None

    def start_episode(self):
        """Episode başlatır ve start state döner"""
        self.state = self.start
        return self.state

    def step(self, action):
        """Bir adım atar. Aksiyonlar: 0=up, 1=right, 2=down, 3=left"""
        r, c = self.state

        if action == 0:   # up
            r = max(r - 1, 0)
        elif action == 1: # right
            c = min(c + 1, self.cols - 1)
        elif action == 2: # down
            r = min(r + 1, self.rows - 1)
        elif action == 3: # left
            c = max(c - 1, 0)

        next_state = (r, c)
        reward = -1
        done = False

        # Cliff kontrolü
        if next_state in self.cliffs:
            reward = -100
            next_state = self.start
        elif next_state == self.goal:
            reward = 10
            done = True

        self.state = next_state
        return next_state, reward, done

class Q_Agent:
    def __init__(self, discount=1, a=0.2, epsilon = 0.1):
        self.a = a
        self.epsilon = epsilon
        self.discount = discount
        self.Q = defaultdict(float)
        self.p_actions = [0,1,2,3]
    def choose_Action(self,state):
        rand = np.random.random()
        q_values = np.array([self.Q[(state, a)] for a in self.p_actions])
        max_q = np.max(q_values)
        greedy = [a for a, q in zip(self.p_actions, q_values) if q == max_q]
        greedy = greedy[0]
        if rand < 1 - self.epsilon:
            return greedy
        
        return np.random.choice([x for x in self.p_actions if x != greedy]) 
    
    def get_max_q(self,next_state):
        q_vals = np.array([self.Q[(next_state,a)] for a in self.p_actions])
        return np.max(q_vals)


    def update(self, state, action, reward, next_state): # next actionu iterationda seçtir
        sa_pair = (state,action)
        q_max = self.get_max_q(next_state)
        self.Q[sa_pair] += self.a * ( reward + self.discount * (q_max - self.Q[sa_pair])) 
        


env = CliffWalkingEnv()
agent = Q_Agent()

reward_per_episode = []
num_ep = 100
for _ in tqdm.tqdm(range(num_ep)):
    done = False
    state = env.start_episode()
    total_reward = 0

    while not done:
        action = agent.choose_Action(state)
        next_state, reward, done = env.step(action)
        agent.update(state,action,reward,next_state)
        state = next_state
        total_reward += reward

    reward_per_episode.append(total_reward)



    # Plotting
plt.figure(figsize=(10,6))
plt.plot(reward_per_episode, marker='o')
plt.ylim(-100, 10)
plt.title('Total Reward per Episode in Cliff Walking with Q_Learning')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.savefig("q.png")