# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 09:35:31 2022

@author: Yx
"""
import gym
from gym import spaces
import numpy as np
import os
import sys
ROOT_DIR = os.path.abspath('./env')
sys.path.append(ROOT_DIR)
from .utils.action_space import MultiAgentActionSpace
from .utils.observation_space import MultiAgentObservationSpace
from envs import fixed_wing_model
# Leader-Follower gym Environment

class CS_flocking:
    metadata = {'render.modes': ['human']}
    def __init__(self,n_agents,args):
        
        '''
        state:
           [x,
            y,
            psi_f-l,
            roll_f,
            roll_l,
            v_f,
            v_l]
        '''

        #Input fixed-wing UAV's physical parameters
        self.n = n_agents
        self.beta = args.beta
        self.sigma = args.sigma
        self.Cr = args.Cr
        self.Cv = args.Cv
        self.wing_length = args.wing_length
        self.air_length = args.air_length
        self.theta = args.theta
        self.delta_r_0 = 40
        self.vmax=args.vmax
        self.rollmax=args.rollmax
        self.alpha_g=9.8
        self.coll_rate = 0
        self.args = args
        self.time_delay = 0
        self.x_grid = 235.0
        self.y_grid = 235.0
        #Initialize flocking environment
        self.action_space = MultiAgentActionSpace(
            [spaces.Box(low = -1,high = 1,shape=(2,)) for _ in range(self.n)])
        self.state_space = MultiAgentActionSpace(
            [spaces.Box(low=np.array([-15000,-15000,-4*np.pi,-np.pi/12,
                                      -np.pi/6,12,12]),
                        high=np.array([15000,15000,4*np.pi,np.pi/12,
                                       np.pi/6,18,18]), shape=(7,)) for _ in range(self.n)])
        self.observation_space = MultiAgentActionSpace(
            [spaces.Box(low=np.array([-15000,-15000,-4*np.pi,-np.pi/12,
                                      -np.pi/12,12,12]),
                        high=np.array([15000,15000,4*np.pi,np.pi/12,
                                       np.pi/12,18,18]), shape=(7,)) for _ in range(self.n)])
        self.reset()



    def reset(self):
        '''
        output:
        [x,  (relative coordinate)
         y,  (relative coordinate)
         psi_f-psi_l,
         roll_f,
         roll_l,
         v_l,
         v_f]
        '''
        self.state, Rstate = fixed_wing_model.reset(self.x_grid, self.y_grid, self.delta_r_0, self.n)
        return Rstate
    
    def Convert(self):
        # TODO: Covert UAVs' state to leader-relative state
        Rstate = fixed_wing_model.Convert(self.state, self.n)
        return fixed_wing_model.double_array_to_nparray(Rstate)

    def step(self, actions):
        self.state, reward, done, info = fixed_wing_model.step(self.state, self.n, actions, self.rollmax, self.vmax, self.wing_length, self.air_length,
                              self.Cv, self.sigma, self.beta, self.Cr, self.theta, self.alpha_g)
        Rstate = self.Convert()
        if done == True and self.args.evaluate == False:
            done = False
        return Rstate, reward, done, info

if __name__ == '__main__':
    env=CS_flocking()
    rst=env.reset()
    buffer=[]
    done=False
    i=0
    while done==False:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        info=np.array(info)
        buffer.append(i)
        buffer[i]=reward
        i+=1
        if i==200:
            done=True
        #buffer[i]=info
        #info=[]
    
    