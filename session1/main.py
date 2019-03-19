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

    def create_default_env(self):
        return rooms.load_env(self.filename, self.movie, self.time_limit, self.stochastic)


    def run(self):
        env = self.create_default_env()
        done = False
        while not done:
            action = self.agent.policy(env.state_summary)
            old_state = env.state_summary
            agent_position, reward, done, other = env.step(action)
            next_state = env.state_summary
            self.agent.update(old_state, action, reward, next_state)
        env.save_video()


agent = a.QAgent(len(rooms.ROOMS_ACTIONS))
episode = Episode(agent)
episode.run()
