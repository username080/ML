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

class sarsa_Agent:
    def __init__(self, discount=1, a=0.2, epsilon = 0.1):
        self.a = a
        self.epsilon = epsilon
        self.discount = discount
        self.Q = defaultdict(float)

    def choose_Action(self,state):
        p_actions = [0,1,2,3]
        rand = np.random.random()
        q_values = np.array([self.Q[(state, a)] for a in p_actions])
        max_q = np.max(q_values)
        greedy = [a for a, q in zip(p_actions, q_values) if q == max_q]
        greedy = greedy[0]
        if rand < 1 - self.epsilon:
            return greedy
        
        return np.random.choice([x for x in p_actions if x != greedy]) 

    def update(self, state, action, reward, next_state, next_action): # next actionu iterationda seçtir
        sa_pair = (tuple(state),action)
        next_sa_pair = (tuple(next_state),next_action)
        self.Q[(sa_pair)] += self.a * ( reward + self.discount * (self.Q[(next_sa_pair)] - self.Q[(sa_pair)])) 
        


env = CliffWalkingEnv()
agent = sarsa_Agent()

reward_per_episode = []
num_ep = 100
for _ in tqdm.tqdm(range(num_ep)):
    done = False
    state = env.start_episode()
    action = agent.choose_Action(state)
    total_reward = 0
    while not done:
        next_state, reward, done = env.step(action)
        next_action = agent.choose_Action(next_state)
        agent.update(state,action,reward,next_state,next_action)
        action = next_action
        state = next_state
        total_reward += reward
    reward_per_episode.append(total_reward)



    # Plotting
plt.figure(figsize=(10,6))
plt.plot(reward_per_episode, marker='o')
plt.ylim(-100, 10)
plt.title('Total Reward per Episode in Cliff Walking with SARSA')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.savefig("sarsa.png")
