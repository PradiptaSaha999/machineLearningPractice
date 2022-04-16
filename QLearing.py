# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:20:06 2020

@author: z00424nd
"""


import gym
import numpy as np


LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY=2000

env = gym.make("MountainCar-v0")
print(env.action_space.n)
print(env.observation_space.high)
print(env.observation_space.low)

DISCRETE_OS_SIZE = [20,20]
print(DISCRETE_OS_SIZE)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE


print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# print(q_table)


def get_discrete_state(state):
    discrete_state=(state-env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))   # we use this tuple to look up the 3 Q values for the available actions in the q-table



discrete_state=get_discrete_state(env.reset())
action = np.argmax(q_table[discrete_state]) # always go right!   
# print (discrete_state)

# print(np.argmax(q_table[discrete_state]))  ##np.argmax returns the position of maximum value
# print(np.max(q_table[discrete_state])) 
# print((q_table[discrete_state])) 
# print(q_table[discrete_state+ (action,)])
# print(env.goal_position)

# env.reset()
for episode in range(EPISODES):
    if episode % SHOW_EVERY==0:
        print(episode)
        render=True
    else:
        render=False
    discrete_state=get_discrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state]) # always go right!
        new_state, reward, done, _ = env.step(action)
        new_discrete_state=get_discrete_state(new_state)
        # print(new_state, reward,_)
        if render:
            env.render()
        if not done: # If simulation did not end yet after last step - update Q table
            max_future_q=np.max(q_table[new_discrete_state])  # Maximum possible Q value in next step (for new state)
            current_q=q_table[discrete_state+ (action,)] # Current Q value (for current state and performed action)
            new_q=(1- LEARNING_RATE)*current_q+LEARNING_RATE*(reward+DISCOUNT*max_future_q)# And here's our equation for a new Q value for current state and action
            q_table[discrete_state+(action, )]=new_q   # Update Q table with new Q value
        elif new_state[0]>= env.goal_position:  # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directl
            print(f"we made it on episode{episode}" )
            q_table[discrete_state+(action,)]=0
            
        discrete_state=new_discrete_state
    
env.close()  
