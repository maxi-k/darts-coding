#!/usr/bin/env python3
import rooms
import agent as a
import random
import matplotlib.pyplot as plot

small_room = 'layouts/rooms_9_9_4.txt'
large_room = 'layouts/rooms_17_17_4.txt'


class Episode:
    def __init__(self, agent):
        self.filename = small_room
        self.movie = '../result.mp4'
        self.time_limit = 800
        self.stochastic = False
        self.agent = agent
        self.learn_count = 100

    def create_default_env(self):
        return rooms.load_env(self.filename, self.movie, self.time_limit, self.stochastic)

    def get_env_state(self, env):
        return str(env.agent_position)

    def episode(self):
        env = self.create_default_env()
        done = False
        discounted_reward = 0
        steps = 0
        while not done:
            action = self.agent.policy(self.get_env_state(env))
            old_state = self.get_env_state(env)
            agent_position, reward, done, other = env.step(action)
            next_state = self.get_env_state(env)
            self.agent.update(old_state, action, reward, next_state)
            discounted_reward += self.agent.discount ** (reward * steps)
            steps += 1
        return env, discounted_reward

    def run(self):
        env = None
        history = []
        for x in range(0, self.learn_count):
            env, discounted_reward = self.episode()
            history.append(discounted_reward)
        env.save_video()
        self.show_plot(history)

    def show_plot(self, history):
        plot.bar(range(len(history)), history, 1 / 1.5, color = 'blue')
        plot.show()



agent = a.QAgent(len(rooms.ROOMS_ACTIONS))
episode = Episode(agent)
episode.run()
