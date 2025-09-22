from collections import defaultdict
import numpy as np
import tqdm
import matplotlib.pyplot as plt
class BlackjackEnv:# state = [dealer_sum,dealer_usable_ace,player_sum,player_usable_ace]
    def __init__(self):
        self.deck = np.clip(np.arange(1,14),0,10) # 1 as ve 13 tane kart var burada
        self.reset()
        
    def reset(self):
        self.get_start_values()

    def sample(self,size):
        x = []
        for _ in range(size): 
            x.append(np.random.randint(0,13))
        return x


    def get_start_values(self):
        state = defaultdict(int)
        
        sample = self.sample(2)#random cards for dealer
        self.hidden = sample[0]

        if sample[1] == 1:
            state[1] += 1
            sample[1] = 11
        state[0] += sample[1]

        sample = self.sample(2)
        for x in sample:
            if x == 1 and state[2] < 11:        
                state[3] += 1
                x = 11

            state[2] += x
        self.state = state
        return state # buradan sonra agent politikaya göre aksiyon seçecek

    def do_action(self,state,action): # 0 stick, 1 hit
        done = False
        reward = 0
        if action:
            new = self.sample(1)
            new = new[0]
            if new == 1 and state[2] < 11:
                state[3] += 1
                new = 11
            
            state[2] += new

            if state[2] == 21 or (state[2] > 21 and not state[3]):
                done = True

            elif(state[2] > 21 and state[3] > 0):
                state[2] -= 10
                state[3] -= 1
            return state, reward, done
        
        else:
            done = True

        if done:
            if state[2] > 21:
                reward = -1
                return state, reward, done
             # artık oyuncu sabit rewardu buradan döndürebilirim
            hidden = self.hidden
            if hidden == 1 and state[0] < 10:
                state[1] += 1
                hidden = 11

            state[0] += hidden  

            while state[0] < 17:
                new = self.sample(1)
                new = new[0]
                if new == 1 and state[0] < 10:
                    state[1] += 1
                    new = 11

                state[0] += new

                if state[0] > 21 and state[1]:
                    state[0] -= 10
                    state[1] -= 1
                
             
            if state[0] > 21:
                reward = 1
            #there is no need to check 21 equality because reward already 0
            elif state[2] > state[0]:
                reward = 1
            elif state[0] > state[2]:
                reward = -1

            return state, reward, done 

class MCagent:
    def __init__(self,discount = 1, epsilon = 0.2):
        self.discount = discount
        self.epsilon = epsilon
        self.policy = defaultdict(lambda: np.random.randint(0,2))

        self.Q = defaultdict(float)
        self.returns = defaultdict(list) #[total_reward,visit_count]

    def choose_action(self,state):
        rand = np.random.random()
        state_key = (state[0],state[1],state[2],state[3])

        if rand < 1 - self.epsilon:
            return self.policy[state_key]
        
        return np.random.randint(0,2)
    
    def update(self,episode):#episode = [state,reward,action]
        G = 0
        for state,reward,action in reversed(episode):
            state_key = (state[0],state[1],state[2],state[3])
            sa_pair = (state_key,action)
            G = reward + G * self.discount 
            self.returns[sa_pair].append(G)
            self.Q[sa_pair] = np.mean(self.returns[sa_pair])
            q_vals = [self.Q[sa_pair] for a in [0,1]]
            best_action = int(np.argmax(q_vals))
            self.policy[state_key] = best_action

    #eğer first visit yapmak istersen bir visited seti tutucaksın ve G yi güncelledikten sonra onu kontrol edeceksin ama gerek yok şuan


num_episodes = 500000

env = BlackjackEnv()
agent = MCagent()



for _ in tqdm.tqdm(range(num_episodes)):
    done = False
    state = env.get_start_values()
    episode = []
    while not done: 
        action = agent.choose_action(state)
        next_state, reward, done = env.do_action(state,action)
        episode.append((state,reward,action))
        state = next_state

    agent.update(episode)


#-------------------------------------------------------------
player_sums = np.arange(12, 22)
dealer_cards = np.arange(1, 11)
policy_usable_ace = np.zeros((len(player_sums), len(dealer_cards)))
policy_no_usable_ace = np.zeros((len(player_sums), len(dealer_cards)))

for pi, ps in enumerate(player_sums):
    for di, dc in enumerate(dealer_cards):
        # With usable ace
        key_ua = (dc, 0, ps, 1)
        policy_usable_ace[pi, di] = agent.policy[key_ua]
        # Without usable ace
        key_nua = (dc, 0, ps, 0)
        policy_no_usable_ace[pi, di] = agent.policy[key_nua]

fig, ax = plt.subplots(1, 2, figsize=(14,6))
im0 = ax[0].imshow(policy_usable_ace, cmap="YlGn", origin='lower', aspect='auto',
                   extent=[1, 10, 12, 21])
ax[0].set_title("Policy with Usable Ace (1=Hit, 0=Stick)")
ax[0].set_xlabel("Dealer Showing")
ax[0].set_ylabel("Player Sum")
fig.colorbar(im0, ax=ax[0], ticks=[0,1])

im1 = ax[1].imshow(policy_no_usable_ace, cmap="YlGn", origin='lower', aspect='auto',
                   extent=[1, 10, 12, 21])
ax[1].set_title("Policy without Usable Ace (1=Hit, 0=Stick)")
ax[1].set_xlabel("Dealer Showing")
ax[1].set_ylabel("Player Sum")
fig.colorbar(im1, ax=ax[1], ticks=[0,1])

plt.tight_layout()
plt.savefig("a.png")