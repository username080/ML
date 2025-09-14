import numpy as np
import random as random
import collections
import matplotlib.pyplot as plt

class gamblerenv():
    def __init__(self):
        self.states = np.arange(1,101)
        self.p_h = 0.4

    def get_probabilities(self, state, action):
        return [(min(100,state + action), self.p_h), (max(0,state - action), 1-self.p_h)]

    def get_Reward(self, state):
        if state == 100:
            return 1
        return 0
    
    def step(self, stake, n1):
        coin = 'tail'

        x = random.random()
        if x < self.p_h:
            coin = 'head'


        done = False
        reward = 0
        new_state = 0
        if coin == 'head':
            new_state = self.states[min(100,n1 + stake)]
            if new_state == 100:
                done = True
                reward = 1

        else:
            new_state = self.states[max(0,n1-stake)]
            if new_state == 0:
                done = True
        

        return new_state,reward,done


class gamblerAgent:
    def __init__(self, mdp, discount=1, theta = 1e-3, sweep=100):
        self.V = np.zeros(101)
        self.policy =  collections.defaultdict(int)
        self.lam= discount
        self.theta = 1e-3
        self.sweep = sweep
        self.mdp = mdp # = env
    
    def possibleActions(self,state):
        return np.arange(1,state+1)
        
    def get_V(self):
        return self.V
    
    def compute_q_values(self, env, state, action):
        q = 0.0

        for next_state, prob in env.get_probabilities(state, action):
            reward = env.get_Reward(next_state)
            q += prob * (reward + self.lam * self.V[next_state])        
        
        return q

    def valueEvaluation(self, states):
        q_values = {}
        for s in range(self.sweep):
            delta = 0
            for state in states:
                old_v = self.V[state]
                q_values = {}
                for action in self.possibleActions(state):
                    q_values[action] = self.compute_q_values(self.mdp, state, action)
                best_action = max(q_values, key=q_values.get)
                self.policy[state] = best_action
                self.V[state] = q_values[best_action]
                delta = max(delta, abs(old_v - self.V[state]))
            if delta < self.theta:
                break
        return self.policy
    


env = gamblerenv()
agent = gamblerAgent(mdp=env)


states = env.states


policy = agent.valueEvaluation(states)



plt.figure(figsize=(10,6))

# V(s) grafiği
plt.subplot(2,1,1)
plt.plot(agent.get_V(), label="Value Function V(s)")
plt.xlabel("Capital (state)")
plt.ylabel("Value")
plt.title("Gambler's Problem - Value Function")
plt.legend()

# Policy grafiği
plt.subplot(2,1,2)
plt.scatter(policy.keys(), policy.values(), s=10, c="red")
plt.xlabel("Capital (state)")
plt.ylabel("Stake (policy)")
plt.title("Optimal Policy")

plt.tight_layout()
plt.savefig('x.png')
