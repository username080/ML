import gymnasium as gym
import numpy as np
from agent import agent
import tqdm
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')


N = 2048
batch_size = 128
n_epochs = 4
alpha = 1e-3

ppo_agent = agent(gamma = 0.99, lam=0.95, n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs, beta=0.01)

n_games = 200
learn_iters = 0
n_steps = 0

scores = []
avg_scores = []

for i in tqdm.tqdm(range(n_games)):
    state, _ = env.reset()
    done = False
    score = 0
    while not done:
        log_prob, action, val = ppo_agent.choose_action(state)
        state_, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        n_steps += 1
        score += reward
        ppo_agent.store_transition(state, action, reward, log_prob, done, val)

                
       
        if  ppo_agent.memory.size > N:
            ppo_agent.learn()
            learn_iters += 1



        state = state_

    scores.append(score)
    avg_score = np.mean(scores[-50:])
    avg_scores.append(avg_score)

plt.figure(figsize=(10,6))
plt.plot(scores, label='Score per episode')
plt.plot(avg_scores, label='Average Score (50 episodes)', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('PPO Learning Progress on CartPole-v1')
plt.legend()
plt.grid(True)
plt.savefig("PPO.png")
