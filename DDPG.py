import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Buffer:
    def __init__(self, n_actions, input_dim, max_size):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.states_mem = np.zeros((self.mem_size, *input_dim))
        self.actions_mem = np.zeros((self.mem_size, n_actions))
        self.rewards_mem = np.zeros(self.mem_size)
        self.states__mem = np.zeros((self.mem_size, *input_dim))
        self.terminal_mem = np.zeros(self.mem_size)

    def store_transitions(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size

        self.states_mem[index] = state
        self.actions_mem[index] = action
        self.rewards_mem[index] = reward
        self.states__mem[index] = state_
        self.terminal_mem[index] = done

        self.mem_cntr += 1

    def get_size(self):
        return self.mem_cntr

    def get_Sample(self, batch_size):
        
        memory = min(self.mem_size, self.mem_cntr)

        batch = np.random.choice(memory, batch_size, replace=False)

        states = self.states_mem[batch]
        actions = self.actions_mem[batch]
        rewards = self.rewards_mem[batch]
        states_ = self.states__mem[batch]
        terminal = self.terminal_mem[batch]

        return states, actions, rewards, states_, terminal


class Critic_Network(keras.Model):
    def __init__(self, l1_dim=256, l2_dim= 256, name = 'critic', chkpt_dir = 'temp/ddpg' ):
        super(Critic_Network,self).__init__()
        self.l1_dim = l1_dim
        self.l2_dim = l2_dim
        self.model_name = name
        self.chkpt_dir = chkpt_dir

        self.checkpoint_file = os.path.join(self.chkpt_dir, self.model_name + "ddpg.h5")

        self.l1 = Dense(self.l1_dim, activation='relu')
        self.l2 = Dense(self.l2_dim, activation = 'relu')
        self.endl = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.l1(tf.concat([state, action],axis=1))
        action_value = self.l2(action_value)
        
        q = self.endl(action_value)
        
        return q


class Actor_Network(keras.Model):
    def __init__(self, n_actions, l1_dim=256, l2_dim=256,  name='actor', chkpt_dir ="'temp/ddpg"):
        super(Actor_Network,self).__init__()

        self.n_actions = n_actions
        self.l1_dim = l1_dim
        self.l2_dim = l2_dim
        self.model_name = name
        self.chkpt_dir = chkpt_dir

        self.checkpoint_file = os.path.join(self.chkpt_dir,self.model_name,"ddpg.h5")

        self.l1 = Dense(self.l1_dim, activation = 'relu')
        self.l2 = Dense(self.l2_dim, activation = 'relu')

        self.endl = Dense(self.n_actions, activation = 'tanh')

    def call(self, state):
        action = self.l1(state)
        action = self.l2(action)

        action = self.endl(action)

        return action    
    


class DDPG_agent:
    def __init__(self, n_actions, input_dims, alpha=0.001, beta = 0.002, env=None, gamma=0.99, 
                 max_size=1000000, tau=0.005, batch_size=64, noise=0.1):
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.max_size = max_size
        self.memory = Buffer(n_actions=self.n_actions, input_dim=self.input_dims, max_size=self.max_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.noise = noise
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = Actor_Network(n_actions=self.n_actions)
        self.target_actor = Actor_Network(n_actions=self.n_actions, name="target_Actor")
        self.critic = Critic_Network()
        self.target_critic = Critic_Network(name="target_Critic")

        self.actor.compile(optimizer=Adam(learning_rate = alpha))
        self.target_actor.compile(optimizer=Adam(learning_rate = alpha))
        self.critic.compile(optimizer=Adam(learning_rate = beta))
        self.target_critic.compile(optimizer=Adam(learning_rate = beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
                                                    #loss üzerinden update kısmı yok gibi burda
        for i,weight in enumerate(self.actor.weights):# bütün weigthler üzerinden iterate edip weigth ve index dönüyü (weightleri liste gibi düşün)
            weights.append(weight*tau + targets[i]*(1-tau))#yeni weight elemanına tau ile ne kadar güncelleneceğini belirleyip güncelleyerek weights listesine atıyo
        self.target_actor.set_weights(weights)#targetı buna göre güncelliyo         off policy algoritma

        weights = []
        targets = self.target_critic.weights

        for i,weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i] * 1-tau)
        self.target_critic.set_weights(weights)

    def save_models(self):
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self,obs,evaluate=False):
        state = tf.convert_to_tensor([obs],dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],stddev=self.noise) 
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 
        
        state,action,reward,state_,done = self.memory.get_Sample(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        states_ = tf.convert_to_tensor(state_, dtype=tf.float32)


        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)# next state için targettan action alıyor
            critic_value_ = tf.squeeze(self.target_critic(states_,target_actions), 1)#t+1 için action ve space le Q dönüyor
            critic_value = tf.squeeze(self.critic(states,actions), 1)# 
            target = rewards + self.gamma * critic_value_ * (1-done)
            critic_loss = keras.losses.MSE(target, critic_value)    

        Critic_network_gradient = tape.gradient(critic_loss,self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(Critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states,new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
        
        self.update_network_parameters()


        # ---------------------------------------------------------------

import gymnasium as gym
import matplotlib.pyplot as plt
import tqdm

env = gym.make("Pendulum-v1")#, render_mode="human")
agent = DDPG_agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])

n_episode = 175
score_history = []
for _ in tqdm.tqdm(range(n_episode)):
    state, _= env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(state)
        state_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        agent.memory.store_transitions(state,action,reward,state_,done)
        agent.learn()
        state = state_
    score_history.append(score)


plt.plot(np.arange(n_episode), score_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DDPG Learning Curve")
plt.grid(True)
plt.savefig("DDPG.png")