from tensorflow import keras
from tensorflow.keras.layers import Dense


class actor_network(keras.Model):
    def __init__(self, n_actions, l1_dims = 256, l2_dims = 256):
        super(actor_network, self).__init__()

        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.n_actions = n_actions

        self.l1 = Dense(l1_dims, activation = 'relu')
        self.l2 = Dense(l2_dims, activation = 'relu')
        self.l3 = Dense(n_actions, activation = 'softmax')

    def call(self, state):
        nn = self.l1(state)
        nn = self.l2(nn)
        nn = self.l3(nn)

        return nn
    

class critic_network(keras.Model):
    def __init__(self, l1_dims = 256, l2_dims = 256):
        super(critic_network, self).__init__()

        self.l1_dims = l1_dims
        self.l2_dims = l2_dims

        self.l1 = Dense(l1_dims, activation = 'relu')
        self.l2 = Dense(l2_dims, activation = 'relu')
        self.l3 = Dense(1, activation = None)

    def call(self, state):
        v = self.l1(state)
        v = self.l2(v)
        v = self.l3(v)

        return v
    
    