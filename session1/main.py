#!/usr/bin/env python3
import rooms
import agent as a
import random
import matplotlib.pyplot as plot

small_room = 'layouts/rooms_9_9_4.txt'
large_room = 'layouts/rooms_17_17_4.txt'


def create_default_env():
    filename = small_room
    movie = '../result.mp4'
    return rooms.load_env(filename, movie)

def episode():
    env = create_default_env()
    done = False
    while not done:
        action = random.choice(rooms.ROOMS_ACTIONS)
        agent_position, reward, done, other = env.step(action)
    env.save_video()

episode()
