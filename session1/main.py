#!/usr/bin/env python3
import multirooms as rooms
import agent as a
import random
import matplotlib.pyplot as plot
import numpy

small_room = 'layouts/rooms_9_9_4.txt'
large_room = 'layouts/rooms_17_17_4.txt'


class Episode:
    def __init__(self, agents):
        self.filename = small_room
        self.movie = '../result.mp4'
        self.time_limit = 800
        self.stochastic = False
        self.agents = agents
        self.learn_count = 1

    def create_default_env(self):
        return rooms.load_env(self.filename, self.movie, len(self.agents), self.time_limit, self.stochastic)

    def get_env_state(self, env, agent):
        return str(env.agent_position[agent])

    def episode(self):
        env = self.create_default_env()
        done = False
        discounted_rewards = numpy.full(len(self.agents), 0)
        steps = 0
        while not done:
            actions = [agent.policy(self.get_env_state(env, index)) for index, agent in enumerate(self.agents)]

            old_states = [self.get_env_state(env, index) for index in range(len(self.agents))]
            agent_positions, rewards, done, other = env.step(actions)
            next_states = [self.get_env_state(env, index) for index in range(len(self.agents))]

            updates = [agent.update(old_states[index], actions[index], rewards[index], next_states[index]) for index, agent in enumerate(self.agents)]
            discounted_rewards = [(discounted_rewards[index] + agent.discount ** (rewards[index] * steps)) for index, agent in enumerate(self.agents)]
            steps += 1
        return env, discounted_rewards

    def run(self):
        env = None
        history = []
        for x in range(0, self.learn_count):
            env, discounted_rewards = self.episode()
            history.append(discounted_rewards)
        self.show_plot(history)
        env.save_video()

    def show_plot(self, history):
        fig, ax = plot.subplots()
        index = numpy.arange(len(self.agents))
        colors = ['r', 'g', 'b']
        bar_width = 0.35
        opacity = 0.8
        for index, agent in enumerate(self.agents):
            plot.bar(index + bar_width, list(map(lambda h: h[index], history)), bar_width, color = colors[index])
        plot.show()


agent_count = 2
agents = [a.QAgent(len(rooms.ROOMS_ACTIONS)) for x in range(agent_count)]
episode = Episode(agents)
episode.run()
