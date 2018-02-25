import numpy as np
import math

"""
Contains the definition of the agent that will run in an
environment.
"""

class epsilon_greedy:
    def __init__(self):
        """Init a new agent.
            """
        self.counts = [0] * 10
        self.values = [0] * 10
        self.epsilon = 0.1
    def act(self, observation):
        """Acts given an observation of the environment.
            
            Takes as argument an observation of the current state, and
            returns the chosen action.
            """
        if np.random.random() < self.epsilon:
            return np.random.randint(0,9)
        else:
            return np.argmax(self.values)
            
    
    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
            given observation.
            
            This is where your agent can learn.
            """
        self.counts[action] = self.counts[action] + 1
        n = self.counts[action]
        value = self.values[action]
        
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[action] = new_value
        
        pass

class optimisitic_ep_greedy:
    def __init__(self):
        """Init a new agent.
            """
        self.counts = [0] * 10
        self.values = [2000] * 10
        self.epsilon = 0.1
    
    def act(self, observation):
        """Acts given an observation of the environment.
            
            Takes as argument an observation of the current state, and
            returns the chosen action.
            """
        if np.random.random() < self.epsilon:
            return np.random.randint(0,9)
        else:
            return np.argmax(self.values)

    
    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
            given observation.
            
            This is where your agent can learn.
            """
        self.counts[action] = self.counts[action] + 1
        n = self.counts[action]
        value = self.values[action]
        
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[action] = new_value
        
        pass

class softmax:
    def __init__(self):
        """Init a new agent.
            """
        self.counts = [0] * 10
        self.values = [0] * 10
        self.softmaxvalues = [0] * 10
        self.t = 0.3
    
    def act(self, observation):
        """Acts given an observation of the environment.
            
            Takes as argument an observation of the current state, and
            returns the chosen action.
            """
        
        for i in range(len(self.counts)):
            if self.counts[i] == 0:
                return i
    
        exp_val = [math.exp(val / self.t) for val in self.values]
        tot_exp_val = np.sum(exp_val)
        self.softmaxvalues = exp_val / tot_exp_val

        l = np.random.random()
        probacumul = 0
        for i in range(len(self.softmaxvalues)):
            probacumul += self.softmaxvalues[i]
            if probacumul > l:
                return i
    
    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
            given observation.
            
            This is where your agent can learn.
            """
        self.counts[action] = self.counts[action] + 1
        n = self.counts[action]
        value = self.values[action]
        
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[action] = new_value
        
        pass

class UCB:
    def __init__(self):
        """Init a new agent.
            """
        self.counts = [0] * 10
        self.values = [0] * 10
        self.ucb_values = [0] * 10
        self.minmax = 0
    
    def act(self, observation):
        """Acts given an observation of the environment.
            
            Takes as argument an observation of the current state, and
            returns the chosen action.
            """
        for i in range(len(self.counts)):
            if self.counts[i] == 0:
                return i
        
        total_counts = np.sum(self.counts)
        for action1 in range(len(self.counts)):
            add = math.sqrt((self.minmax/2 * math.log(total_counts)) / float(self.counts[action1]))
            self.ucb_values[action1] = self.values[action1] + add
        return np.argmax(self.ucb_values)

    
    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
            given observation.
            
            This is where your agent can learn.
            """
        self.counts[action] = self.counts[action] + 1
        n = self.counts[action]
        value = self.values[action]
        
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[action] = new_value
        self.minmax = max(self.values) - min(self.values)
        
        
        pass

# Choose which Agent is run for scoring
Agent = UCB
