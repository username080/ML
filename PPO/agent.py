import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow as keras
from tensorflow.keras.optimizers import Adam

from networks import actor_network, critic_network
from memory import PPOmemory

class agent:
    def __init__(self, n_actions, beta, alpha=5e-4, lam=0.95, gamma=0.99, epsilon=0.2, batch_size=64, n_epochs=10):
        
        self.lam = lam
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta # for entropy
        self.n_epochs = n_epochs
        
        self.memory = PPOmemory(batch_size)

        self.actor = actor_network(n_actions)
        self.actor.compile(optimizer = Adam(learning_rate = alpha))
        self.critic = critic_network()
        self.critic.compile(optimizer = Adam(learning_rate = alpha))


    def store_transition(self, state, action, reward, prob, done, value):
        self.memory.store(state, action, prob, value, reward, done)

    def choose_action(self, state): # learn how to use entropy then implement this function
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.expand_dims(state, axis=0)


        probs = self.actor(state)
        values = self.critic(state)
        
        dist = tfp.distributions.Categorical(probs)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)

        actions = actions.numpy()[0]
        values = values.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return log_prob, actions, values # where to use log_prob ??
        
    def learn(self):
        for _ in range(self.n_epochs):
            states, actions, old_log_probs, values, rewards, dones, batches = self.memory.get_sample()

            # --- 1. Advantage hesaplama ---
            advantage = np.zeros_like(rewards, dtype=np.float32)
            
            for t in range(len(rewards)):
                discount, a_t = 1, 0
                for k in range(t, len(rewards) - 1):
                    
                    delta = rewards[k] + self.gamma * values[k+1] * (1 - dones[k]) - values[k]
                    a_t += discount * delta
                    discount *= self.gamma * self.lam
                advantage[t] = a_t

            # --- 2. Her minibatch için PPO update ---
            for batch in batches:
                states_b   = tf.convert_to_tensor(states[batch], dtype=tf.float32)
                actions_b  = tf.convert_to_tensor(actions[batch], dtype=tf.int32)
                old_logp_b = tf.convert_to_tensor(old_log_probs[batch], dtype=tf.float32)
                adv_b      = tf.convert_to_tensor(advantage[batch], dtype=tf.float32)
                vals_b     = tf.convert_to_tensor(values[batch], dtype=tf.float32)

                with tf.GradientTape(persistent=True) as tape:
                    


                    probs = self.actor(states_b)
                    dist = tfp.distributions.Categorical(probs=probs)
                    #entropy = dist.entropy()
                    new_logp = dist.log_prob(actions_b)

                    # Ratio ve clip
                    ratio = tf.exp(new_logp - old_logp_b)
                    weighted = ratio * adv_b
                    clipped = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    weighted_clipped = clipped * adv_b

                    # Actor loss
                    actor_loss = -tf.reduce_mean(tf.minimum(weighted, weighted_clipped))# - self.beta * entropy)

                    # Critic loss
                    critic_value = tf.squeeze(self.critic(states_b), 1)
                    returns = adv_b + vals_b
                    critic_loss = tf.reduce_mean(tf.square(returns - critic_value))

                # Gradient update
                actor_params = self.actor.trainable_variables
                critic_params = self.critic.trainable_variables

                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_grads = tape.gradient(critic_loss, critic_params)
                
                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

        self.memory.clear_memory()
