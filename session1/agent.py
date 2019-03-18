import random
import numpy
import copy

class Agent:

    def __init__(self, nr_actions):
        self.nr_actions = nr_actions
        
    def policy(self, state):
        pass
        
    def update(self, state, action, reward, next_state):
        pass
        

class RandomAgent:

    def __init__(self, nr_actions):
        self.nr_actions = nr_actions
        
    def policy(self, state):
        return random.choice(range(self.nr_actions))
        
    def update(self, state, action, reward, next_state):
        pass
        