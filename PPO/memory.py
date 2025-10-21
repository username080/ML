import numpy as np

class PPOmemory:
    def __init__(self, batch_size):
    
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.dones = []
        self.values = []
        self.size = 0

    def get_sample(self):
        
        n_states = len(self.states)
        indices = np.arange(0, n_states)
        
        batches = [indices[i:i + self.batch_size] for i in range(0, n_states, self.batch_size)] # ??
        
        return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.values),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches

    def store(self, state, action, reward, prob, done, value):
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.probs.append(prob)
        self.dones.append(done)
        self.values.append(value)
        self.size += 1

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.dones = []
        self.values = []
        self.size = 0
        