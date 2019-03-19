import random
import numpy
import copy

# Q-Matrix:
# Q(s, a) = reward + gamma * max(Q(r', a'))
# Reward, den ich mit der besten aktion danach erwarte


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


class QAgent:

    def __init__(self, nr_actions):
        self.nr_actions = nr_actions
        self.discount = 0.9
        self.greedy = 0.1
        self.learning_speed = 0.2
        self.random_agent = RandomAgent(nr_actions)
        self.qmatrix = {}
        # dictionary

    def initialize_action_row(self):
        return { x : 0 for x in range(self.nr_actions) }


    def ensure_matrix_actions(self, state):
        if state not in self.qmatrix:
            self.qmatrix[state] = self.initialize_action_row()

    def get_matrix_actions(self, state):
        self.ensure_matrix_actions(state)
        return self.qmatrix[state]

    def get_best_action_for_state(self, state):
        cur_state_actions = self.get_matrix_actions(state)

        max_reward = max(cur_state_actions.values())
        best_actions = [ key for key, value in cur_state_actions.items() if value == max_reward ]

        return random.choice(best_actions)


    def policy(self, state):
        if numpy.random.rand() < self.greedy:
            return self.random_agent.policy(state)
        return self.get_best_action_for_state(state)

    def update(self, state, action, reward, next_state):
        # alpha = learning_speed, gamma = discount
        # Q(s, a)_neu = (1 - alpha) * q(s, a)_alt + alpha * ( r + gamma * max(Q(r, a)))
        self.ensure_matrix_actions(state)
        self.qmatrix[state][action] = (1 - self.learning_speed) * self.qmatrix[state][action] + self.learning_speed * ( reward + self.discount * self.qmatrix[next_state][self.get_best_action_for_state(next_state)])
