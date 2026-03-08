import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow as keras
from tensorflow.keras.optimizers import Adam

from networks import actor_network, critic_network
from memory import PPOmemory

class agent:#losslara odaklan
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


    def store_transition(self, state, action, reward, log_prob, done, value):
        self.memory.store(state, action, reward, log_prob, done, value)

    def choose_action(self, state): 
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

        return log_prob, actions, values 
        
    def learn(self):
        for _ in range(self.n_epochs):  
            states, actions, old_log_probs, values, rewards, dones, batches = self.memory.get_sample()
            # Compute GAE advantage
            advantage = np.zeros(len(rewards), dtype=np.float32)
            gae = 0
            for t in reversed(range(len(rewards) - 1)):
                delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
                advantage[t] = gae


            returns = advantage + values  # Critic target, NOT normalized

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    # Convert batch to tensors
                    states_b = tf.convert_to_tensor(states[batch], dtype=tf.float32)
                    actions_b = tf.convert_to_tensor(actions[batch])
                    old_log_probs_b = tf.convert_to_tensor(old_log_probs[batch], dtype=tf.float32)
                    adv_b = advantage[batch]
                    
                    # Normalize advantage only for actor
                    adv_b = (adv_b - tf.reduce_mean(adv_b)) / (tf.math.reduce_std(adv_b) + 1e-8)

                    # Actor
                    probs = self.actor(states_b)
                    #print(probs)
                    dist = tfp.distributions.Categorical(probs)
                    new_log_probs = dist.log_prob(actions_b)
                    entropy = tf.reduce_mean(dist.entropy())
                    
                    prob_ratio = tf.exp(new_log_probs - old_log_probs_b)
                    
                    weighted_probs = adv_b * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.epsilon, 1 + self.epsilon)
                    weighted_and_clipped = clipped_probs * adv_b
                    actor_loss = -tf.reduce_mean(tf.minimum(weighted_probs, weighted_and_clipped))
                    #print(actor_loss)
                    # Critic
                    critic_value = self.critic(states_b)
                    returns_b = tf.convert_to_tensor(returns[batch], dtype=tf.float32)

                    critic_loss = tf.reduce_mean(tf.square(returns_b - critic_value))
                    #print("critic",critic_loss)

                    # Total loss   critic - entropyi + yaptım(original paperda öyle gösteriyodu)
                    total_loss = actor_loss - 0.5 * critic_loss + self.beta * entropy

                # Apply gradients
                all_vars = self.actor.trainable_variables + self.critic.trainable_variables
                grads = tape.gradient(total_loss, all_vars)
                n_actor = len(self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(grads[:n_actor], self.actor.trainable_variables))
                self.critic.optimizer.apply_gradients(zip(grads[n_actor:], self.critic.trainable_variables))
            print('learned')

