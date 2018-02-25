import numpy as np
import math
"""
Contains the definition of the agent that will run in an
environment.
"""
class Qlearning:
    
    def __init__(self):
        """Init a new agent.
            """
        self.p = 20
        self.k = 20
        self.f = (self.p + 1,self.k +1,3)
        self.e = np.zeros(self.f)
        self.Q = np.ones(self.f)*5
        
        self.g = (self.p + 1,self.k +1)
        self.locx = np.ones(self.g)
        self.locvx = np.ones(self.g)
        self.phi = np.zeros(self.g)
        self.previous_phi = np.ones(self.g)
        
        self.s = [0,0]
        self.previous_observation = None
        self.previous_situation = None
        self.previous_reward = None
        self.previous_action = None
        self.count_episodes = 0
        self.count_steps = 0
        self.avg_steps = 0
        self.count_victory = 0

        self.totcount = 0
        self.choice = [0]*3
        
        self.Lambda = 0.5
        self.alpha = 0.8
        self.gamma = 0.8
        self.epsilon = 0.1

    
    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.
            
            Parameters of the environment do not change when starting a new
            episode of the same game, but your initial location is randomized.
            
            x_range = [xmin, xmax] contains the range of possible values for x
            
            range for vx is always [-20, 20]
            """
        self.xmin = x_range[0]
        self.xmax = x_range[1]
        self.totcount += 1
        self.locx = [self.xmin + float(i * (self.xmax - self.xmin)) / float(self.p) for i in range (self.p +1)]
        self.locvx = [-20 + float(j * 40)/float(self.k) for j in range (self.k + 1)]
        self.count_episodes += 1
        self.count_steps = 0

        pass
    
    def act(self, observation):
        """Acts given an observation of the environment.
            
            Takes as argument an observation of the current state, and
            returns the chosen action.
            
            observation = (x, vx)
            """
        self.s = [math.floor(observation[0]*float(self.p)/(self.xmax-self.xmin) - self.xmin *float(self.p)/(self.xmax-self.xmin)), math.floor(observation[1]*float(self.k)/40 + 20*float(self.k)/40)]
        
        
        if self.totcount < 180:
            rand = np.random.random() 
            if rand < self.epsilon:
                return np.random.randint(0,3) - 1
    
        self.choice = [np.sum(self.Q[:,:,i] * self.phi) for i in range(3)]
        return np.argmax(self.choice ) - 1
    

    def reward(self, observation, action, reward):

        for i in range (self.p + 1):
            for j in range(self.k +1):
               self.phi[i,j] = np.exp(-(np.power((observation[0]-self.locx[i]),2)))*np.exp(-(np.power((observation[1]-self.locvx[j]),2)))

        #print(self.phi)
        #print(np.max(self.phi))

        if self.previous_observation is not None:
            self.delta = reward + self.gamma *  self.Q[self.s[0], self.s[1], action +1]  - self.Q[self.previous_situation]
            self.e[self.previous_situation] = self.e[self.previous_situation] + 1


            self.Q = self.Q + self.alpha * self.delta * self.e
            self.e = self.gamma * self.Lambda * self.e

 
        #Update values

        self.previous_phi = self.phi
        self.previous_action = action
        self.previous_reward = reward
        self.previous_observation = observation
        self.previous_situation = (self.s[0], self.s[1], action +1)
        self.count_steps += 1

        if reward > 0:
            if self.count_episodes > 180:
                self.count_victory += 1
                self.avg_steps += self.count_steps
                # print(self.W)
                print("Solved episode %s (steps: %s)"
                % (self.count_episodes, self.count_steps))
            if self.count_episodes == 200:
                print("Test time finished (%i / 20 games), avg step %.1f => score: %s"
                % (self.count_victory,
                    float(self.avg_steps) / float(self.count_victory),
                    float(self.count_victory)*50-0.1*self.avg_steps))

        pass


Agent = Qlearning
