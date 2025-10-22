import gymnasium as gym
import numpy as np
from agent import agent
import tqdm
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')#,render_mode='human')


N = 2048
batch_size = 64
n_epochs = 15
alpha = 3e-4
epsilon = 0.25
ppo_agent = agent(gamma = 0.99, lam=0.95, n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs, beta=0.005, epsilon=epsilon  )

n_games = 500
learn_iters = 0
n_steps = 0

scores = []
avg_scores = []
mean_scores = 0

for i in tqdm.tqdm(range(n_games)):
    state, _ = env.reset()
    done = False
    score = 0
    quit = 0
    while not done:
        log_prob, action, val = ppo_agent.choose_action(state)
        state_, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        n_steps += 1
        score += reward
        ppo_agent.store_transition(state, action, reward, log_prob, done, val)

                
       
        if  ppo_agent.memory.size >= N:
            old_params = np.concatenate([p.numpy().flatten() for p in ppo_agent.actor.trainable_variables])
            ppo_agent.learn() # sorun learnde. Öğrenmiyor orospu evladı amın oğlu piç
            new_params = np.concatenate([p.numpy().flatten() for p in ppo_agent.actor.trainable_variables])
            print("Average param change:", np.mean(np.abs(new_params - old_params)))
            learn_iters += 1
            if np.mean(np.abs(new_params - old_params)) < 0.002:
                quit = 1
            ppo_agent.memory.clear_memory()

        state = state_


    #print('neext')
    mean_scores += score
    if i % 10 == 0:
        print('mean_scores',mean_scores / 10)
        mean_scores = 0
    scores.append(score)
    avg_score = np.mean(scores[-50:])
    avg_scores.append(avg_score)
    if quit:
        break

plt.figure(figsize=(10,6))
plt.plot(scores, label='Score per episode')
plt.plot(avg_scores, label='Average Score (50 episodes)', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('PPO Learning Progress on CartPole-v1')
plt.legend()
plt.grid(True)
plt.savefig("PPO.png")

