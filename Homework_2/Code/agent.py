import numpy as np
import math
"""
Contains the definition of the agent that will run in an
environment.
"""

ACT_UP    = 1
ACT_DOWN  = 2
ACT_LEFT  = 3
ACT_RIGHT = 4
ACT_TORCH_UP    = 5
ACT_TORCH_DOWN  = 6
ACT_TORCH_LEFT  = 7
ACT_TORCH_RIGHT = 8

class qlearning_epsilongreedy:
    
    def __init__(self):
        """Init a new agent.
            """
        
        self.s = (10,10,2,2,4,8)
        self.score = np.zeros(self.s)

        self.totcount = 0
        self.previous_situation = (0,0,0,0,0,0)
        self.previous_score = 0
        self.previous_reward = 0
        
        self.epsilon = 0.15
        self.alpha = 0.75
        self.gamma = 1
    
    def reset(self):
        """Reset the internal state of the agent, for a new run.
            
        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.
        
        You must **not** reset the learned parameters.
        """
        self.totcount += 1
        
        
        pass
    
    def act(self, observation):
        """Acts given an observation of the environment.
            
        Takes as argument an observation of the current state, and
        returns the chosen action.
        
        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """
        
        
        if self.totcount < 700 and np.random.random() < self.epsilon:
            return np.random.randint(0,8) + 1
        else:
            return np.argmax(self.score[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],] ) + 1


    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.
        
        This is where your agent can learn.
        """
        
    
        future_value = self.score[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action-1]
        value = self.previous_score
        
        new_value = value + self.alpha * (self.previous_reward  + self.gamma * future_value - value)
        
        self.score[self.previous_situation] = new_value
    
        #update previous situation
        self.previous_situation = (observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action-1)
        
        self.previous_score = self.score[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action-1]

        self.previous_reward = reward
        pass


class qlearning_UCB:
    
    def __init__(self):
        """Init a new agent.
            """
        
        self.s = (10,10,2,2,4,8)
        self.score = np.zeros(self.s)
        
        self.ucb_values = np.zeros(self.s)
        self.counts = np.zeros(self.s)
        
        self.previous_situation = (0,0,0,0,0,0)
        self.previous_score = None
        self.previous_reward = 0
        
        self.alpha = 0.75
        self.gamma = 1

    
    def reset(self):
        """Reset the internal state of the agent, for a new run.
            
            You need to reset the internal state of your agent here, as it
            will be started in a new instance of the environment.
            
            You must **not** reset the learned parameters.
            """
        #self.previous_score = None
        
        
        pass
    
    def act(self, observation):
        """Acts given an observation of the environment.
            
            Takes as argument an observation of the current state, and
            returns the chosen action.
            
            Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
            """
        total_counter = np.sum(self.counts) +1
        total_counts = np.sum(self.counts[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],]) +1
        
        if total_counter < 900:
            for action1 in range(8):
                add = math.sqrt((2 * math.log(total_counts)) / (float(self.counts[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action1])+1))
                self.ucb_values[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action1] = self.score[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action1] + add
            
            return np.argmax(self.ucb_values[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],]) + 1
    
        return np.argmax(self.score[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],] ) + 1


    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.
        
        This is where your agent can learn.
        """

        self.counts[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action-1] = self.counts[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action-1]  +1
        
        if (self.previous_score is not None):
            future_value = self.score[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action-1]
            value = self.previous_score
            
            new_value = value + self.alpha * (self.previous_reward + self.gamma * future_value - value)
            
            self.score[self.previous_situation] = new_value
            #update previous situation
            
        self.previous_situation = (observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action-1)
        
        self.previous_score = self.score[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action-1]
        
        self.previous_reward = reward
        pass



class qlearning1:
    
    def __init__(self):
        """Init a new agent.
            """
        
        self.s = (10,10,2,2,4,8)
        self.score = np.zeros(self.s)
        
        self.ac = 0
        self.previous_situation = (0,0,0,0,0,0)
        self.previous_reward = 0
        
        self.alpha = 0.75
        self.gamma = 1
    
    def reset(self):
        """Reset the internal state of the agent, for a new run.
            
            You need to reset the internal state of your agent here, as it
            will be started in a new instance of the environment.
            
            You must **not** reset the learned parameters.
            """
        
        pass
    
    def act(self, observation):
        """Acts given an observation of the environment.
            
            Takes as argument an observation of the current state, and
            returns the chosen action.
            
            Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
            """
        
        return np.argmax(self.score[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],] ) + 1
    
    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
            given observation.
            
            This is where your agent can learn.
            """
        
        future_value = self.score[observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action-1]
        value = self.score[self.previous_situation]
        
        new_value = value + self.alpha * (self.previous_reward  + self.gamma * future_value - value)
        
        self.score[self.previous_situation] = new_value
        
        #update previous situation
        self.previous_situation = (observation[0][0],observation[0][1],int(observation[1]),int(observation[2]),observation[3],action-1)
        
        self.previous_reward = reward
        pass

Agent = qlearning_epsilongreedy
