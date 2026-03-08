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

    def do_action(self,state,action): # 1 stick, 2 hit
        done = False
        reward = 0
        if action == 2:
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
                state[2] = 22
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


#state defaultdict dönüyor!


class off_policy_agent:
    def __init__(self, discount=1, epsilon=0.2):
        self.target_policy = defaultdict(lambda: 1)  # Always pick action 2 initially
        self.discount = discount
        self.epsilon = epsilon
        self.Q = defaultdict(float)
        self.C = defaultdict(float)
    
    def get_behavior_prob(self, state_key, action):
        greedy = self.target_policy[state_key]
        if action == greedy:
            return 1 - self.epsilon + self.epsilon / 2 # b_policy eğer greedy seçerse 0.9 prob
        else:
            return self.epsilon / 2 # eğer explore ederse 0.1 prob
    
    def choose_action(self, state):
        state_key = (state[0], state[1], state[2], state[3])
        greedy = self.target_policy[state_key]
        if np.random.random() < 1 - self.epsilon + self.epsilon / 2:
            return greedy
        else:
            return np.random.choice([a for a in [1, 2] if a != greedy])
    
    def update(self, episode):
        G = 0
        W = 1
        for state, action, reward in reversed(episode):
            state_key = (state[0], state[1], state[2], state[3])
            sa_pair = (state_key, action)
            G = reward + self.discount * G
            self.C[sa_pair] += W
            self.Q[sa_pair] += (W / self.C[sa_pair]) * (G - self.Q[sa_pair])
            # Update target policy to be greedy
            q1 = self.Q[(state_key, 1)]
            q2 = self.Q[(state_key, 2)]
            self.target_policy[state_key] = 1 if q1 >= q2 else 2# ++
            # Importance sampling ratio
            if action != self.target_policy[state_key]:
                break  # 
            W = W / self.get_behavior_prob(state_key, action)
            if W == 0:
                break  # Can stop early if W hits zero  




env = BlackjackEnv()
agent = off_policy_agent(epsilon=0.2)

num_episode = 1000000

for _ in tqdm.tqdm(range(num_episode)):
    episode = []
    states = env.get_start_values()
    done = False
    while not done:
        action = agent.choose_action(states)
        next, reward, done = env.do_action(states, action)
        episode.append(((states[0], states[1], states[2], states[3]), action, reward))
        states = next

    agent.update(episode)

# --- Policy Visualization ---
def plot():
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    policy_usable_ace = np.zeros((len(player_sums), len(dealer_cards)))
    policy_no_usable_ace = np.zeros((len(player_sums), len(dealer_cards)))

    for pi, ps in enumerate(player_sums):
        for di, dc in enumerate(dealer_cards):
            key_ua = (dc, 0, ps, 1)
            key_nua = (dc, 0, ps, 0)
            policy_usable_ace[pi, di] = agent.target_policy[key_ua]
            policy_no_usable_ace[pi, di] = agent.target_policy[key_nua]

    fig, ax = plt.subplots(1, 2, figsize=(14,6))
    im0 = ax[0].imshow(policy_usable_ace, cmap="YlGn", origin='lower', aspect='auto',
                extent=[1, 10, 12, 21])
    ax[0].set_title("Policy with Usable Ace (2=Hit, 1=Stick)")
    ax[0].set_xlabel("Dealer Showing")
    ax[0].set_ylabel("Player Sum")
    fig.colorbar(im0, ax=ax[0], ticks=[1,2])

    im1 = ax[1].imshow(policy_no_usable_ace, cmap="YlGn", origin='lower', aspect='auto',
                extent=[1, 10, 12, 21])
    ax[1].set_title("Policy without Usable Ace (2=Hit, 1=Stick)")
    ax[1].set_xlabel("Dealer Showing")
    ax[1].set_ylabel("Player Sum")
    fig.colorbar(im1, ax=ax[1], ticks=[1,2])

    plt.tight_layout()
    plt.savefig("b.png")

plot()
