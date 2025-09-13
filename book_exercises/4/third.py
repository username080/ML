import tqdm._tqdm as tqdm
import numpy as np
import scipy.stats.poisson as poisson

#true but terrible solution

discount_rate = 0.9
max_move_at_night = 5
max_cars = 20
rent_lam = [3,4]
return_lam = [3,2]

class env:
    def __init__(self):
        self.states = np.array([[10,10]])
    
    def get_states(self):
        return self.states
    
    def rents(self,lam):
        rent1 = np.random.poisson(lam[0])
        rent2 = np.random.poisson(lam[1])
        return np.array([rent1,rent2])

    def returns(self,lam):
        return1 = np.random.poisson(lam[0])
        return2 = np.random.poisson(lam[1])
        return np.array([return1,return2])
        

    def step(self,action,reward):
        rented = self.rents(rent_lam)
        returned = self.returns(return_lam)

        self.states = self.states - rented + returned

        self.states[0] += action
        self.states[1] += action 

        done = False
        if self.states[0] <= 0 or self.states[1] <= 0:
            done = True
        
        reward = ( rented[0] + rented[1] ) * 10 - 2* action.num

        return self.states, reward, done
         

class agent:
    def __init__(self): 
        self.V = np.zeros((max_cars+1,max_cars+1))
        self.policy = np.zeros((max_cars+1,max_cars+1))


    def expected_Return(self, V, n1, n2, action ):
        max_n = 11

        new_n1 = min(max_cars,max(0,n1+action))
        new_n2 = min(max_cars,max(0,n2+action))

        expected = 0.0

        for rent1 in range(max_n):
            for rent2 in range(max_n):

                realrent1 = min(new_n1,rent1)
                realrent2 = min(new_n2,rent2)

                reward = 10 * ( realrent1 + realrent2 ) - 2 * abs(action)

                p_rent = poisson.pmf(rent1, rent_lam[0]) * poisson.pmf(rent2 * rent_lam[1])

                for ret1 in range(max_n):
                    for ret2 in range(max_n):
                        n1_next = min(max_cars, new_n1 + ret1 - realrent1)
                        n2_next = min(max_cars,new_n2 + ret2 - realrent2)

                        p_ret = poisson.pmf(ret1, return_lam[0]) * poisson.pmf(ret2, return_lam[1])

                        probability = p_ret * p_rent

                        expected += probability * (reward + discount_rate * V[n1_next,n2_next])
        return expected    

    
    def policy_evaluation(self,V,policy):
        delta = 0
        theta = 1e-3

        while True:
            delta = 0
            for n1 in range(max_cars+1):
                for n2 in range (max_cars+1):
                    v = V[n1,n2]
                    action = policy[n1,n2]
                    V[n1,n2] = self.expected_Return(V,n1,n2,action)
                    delta = max(delta,abs(v - V[n1,n2])) 
            if delta < theta:
                    break    
        return V
    
    def policy_Improvement(self,policy,V):
        policy_Stable = True
        max_act = 5
        for n1 in range(max_cars+1):
            for n2 in range (max_cars+1):
                old_action = policy[n1,n2]
                possible_Actions = []
                for action in range(-max_act, max_act+1):
                    if 0<= n1+action <= max_cars and 0<= n2 + action <= max_cars:
                        possible_Actions.append((self.expected_Return(V,n1,n2,action),action))                
                x,action =max(possible_Actions)
                policy[n1,n2] = action
                if action != old_action:
                    policy_Stable = False


        return policy_Stable




    
    






